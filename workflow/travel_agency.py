# A travel agency as a workflow: five subagents, each with its own workflow,
# passing the caller along concierge -> flights -> hotels -> activities -> booking.
# The runtime runs the workflow; this file runs the tools and hooks. Prices are in
# Indian rupees. end_call is the agent's own tool, so every subagent can hang up.
#
# Each step's answer is a dataclass (its Step's `result`), read back by step id:
# self.workflow.search here, self.workflow.concierge.trip from another subagent.
# Supplier searches take seconds (simulated with asyncio.sleep), so each one says
# a short line while it works instead of leaving dead air. Payment can't be cut
# off halfway, so it is cancel_on_interrupt=False.
# Hotels are skipped for a day trip, the visa check and the extras are optional
# (the caller can decline them), and a sold-out fare is retried up to max_attempts.
#
# Plans can change at any step: change_trip goes back to that concierge step.
# If the new answer differs, everything after the concierge is done again in
# order (flights, hotels, activities, booking); if it's the same, nothing is,
# and the caller passes straight back to where they were.

import asyncio
import random
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import zeroruntime
from zeroruntime import Agent, AgentContext, Pipeline, Room, Step, SubAgent, current_session, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD
from zeroruntime.workflow import on_workflow

load_dotenv(override=True)

IST = ZoneInfo("Asia/Kolkata")
STYLES = ("budget", "comfort", "luxury")


# Each step's answer

@dataclass
class Trip:
    origin: str
    destination: str
    depart: str
    return_date: str
    nights: int


@dataclass
class Travellers:
    lead_name: str
    party_size: int


@dataclass
class Budget:
    rupees: float
    style: str


@dataclass
class Visa:
    passport: str
    destination: str
    rule: str


@dataclass
class Fares:
    fares: list[dict[str, Any]]


@dataclass
class HeldFare:
    fare_id: str
    airline: str
    seat: str
    total_rupees: int


@dataclass
class Area:
    neighbourhood: str


@dataclass
class HotelOptions:
    hotels: list[dict[str, Any]]
    budget_left_rupees: int


@dataclass
class HotelRoom:
    hotel_id: str
    hotel: str
    room: str
    total_rupees: int


@dataclass
class Interests:
    interests: list[str]


@dataclass
class Itinerary:
    plan: list[dict[str, Any]]


@dataclass
class Extras:
    transfer_rupees: int
    insurance_rupees: int


@dataclass
class Quote:
    total_rupees: int


@dataclass
class Payment:
    payment_id: str
    email: str


@dataclass
class Documents:
    booking_ref: str
    emailed_to: str


def today() -> date:
    return datetime.now(IST).date()


@function_tool
async def end_call() -> dict:
    """Hang up, with the goodbye. Call only once the caller has said they have nothing else."""
    await current_session().agent.hangup(reason="caller done")
    return {"status": "ending"}


async def working(agent: SubAgent, lines: list[str], seconds: float = 2.5) -> None:
    """A slow supplier call, with a short line per step so the caller isn't left in silence."""
    for line in lines:
        await agent.session.say(line, add_to_chat_context=False)
        await asyncio.sleep(seconds)


async def change_trip(agent: SubAgent, part: str) -> dict:
    if part not in ("trip", "travellers", "budget"):
        return {"error": "part is trip, travellers or budget."}
    return await agent.go_to_step(f"concierge.{part}", reason=f"caller wants to change their {part}")


class Concierge(SubAgent):
    name = "concierge"
    instructions = (
        "You're the front desk. Get the trip, who's travelling and the budget, one thing at a time, "
        "like a friendly travel agent would. Confirm dates back in words, like 'Friday the sixth of "
        "November'. If the caller changes an earlier answer, call change_answer for that part."
    )
    workflow = [
        Step(id="trip", description="where and when",
             instructions="Find out where they're flying from and to, and their travel dates. Work out exact dates "
             "from what they say, relative to today, then call record_trip.",
             opening="Ask where they'd like to go, and when.", result=Trip),
        Step(id="travellers", description="who's travelling",
             instructions="Find out who's travelling: the lead traveller's name and how many people in total. The lead "
             "is usually the caller, so if you know their name, just confirm it. Then call record_travellers.",
             opening="Ask who's coming along.", result=Travellers),
        Step(id="budget", description="budget and style",
             instructions="Find out their total budget in rupees for the whole trip, and whether they'd like a budget, "
             "comfort or luxury trip. Never assume a style; ask. Then call record_budget.",
             opening="Ask roughly what they'd like to spend on the whole trip, and what kind of trip they have in mind.",
             result=Budget, max_attempts=3),
        Step(id="visa", description="visa check",
             instructions="If they're travelling abroad, ask which passport they hold and call check_visa. If they "
             "already have a visa or don't need one, the step can be skipped.",
             opening="Ask which passport they'll be travelling on, so you can check the entry rules.",
             result=Visa, optional=True),
    ]

    async def on_enter(self, reason: str):
        if reason == "call started":
            name = self.userdata.get("caller_name")
            hello = f"Greet {name} by name" if name else "Greet the caller"
            return f"{hello}, warmly and briefly, as Wanderlust Travel, and ask how you can help plan their trip."
        if reason.startswith("caller wants to change"):
            return "Say of course, and ask what they'd like it to be now. Everything else they told you stays."
        return None

    @function_tool
    async def record_trip(self, origin: str, destination: str, depart: str, return_date: str) -> dict:
        """Record the route and dates.

        Args:
            origin: The city they're flying from.
            destination: The city they're flying to.
            depart: Departure date, YYYY-MM-DD.
            return_date: Return date, YYYY-MM-DD; the same as depart for a day trip.
        """
        try:
            out, back = date.fromisoformat(depart), date.fromisoformat(return_date)
        except ValueError:
            return {"error": "Dates must be YYYY-MM-DD. Work them out from what the caller said and today's date."}
        if out < today():
            return {"error": f"{depart} has already passed; today is {today().isoformat()}. Check the dates with the caller."}
        if back < out:
            return {"error": "The return date is before the departure. Check both dates with the caller."}
        return await self.complete(Trip(origin.strip(), destination.strip(), depart, return_date, (back - out).days))

    @function_tool
    async def record_travellers(self, lead_name: str, party_size: int) -> dict:
        """Record the lead traveller and how many people are travelling in total.

        Args:
            lead_name: The lead traveller's full name.
            party_size: Everyone travelling, the lead included.
        """
        if party_size > 9:
            return {"error": "Groups of ten or more go to our group desk; offer to have them call back."}
        return await self.complete(Travellers(lead_name.strip(), party_size))

    @function_tool
    async def record_budget(self, rupees: float, style: str) -> dict:
        """Record the total budget in Indian rupees, and the travel style.

        Args:
            rupees: The total budget for the whole trip in rupees, as a number: 5 lakh is 500000, 1 crore is 10000000.
            style: budget, comfort or luxury.
        """
        style = style.strip().lower()
        if style not in STYLES:
            return {"error": "Style is budget, comfort or luxury. Ask which fits them best."}
        return await self.complete(Budget(rupees, style))

    @function_tool
    async def check_visa(self, passport_country: str) -> dict:
        """Check the entry rules for the caller's passport at their destination. Takes a few seconds.

        Args:
            passport_country: The country that issued their passport.
        """
        await working(self, ["Let me check the entry rules for you."], seconds=3)
        return await self.complete(Visa(passport_country, self.workflow.trip.destination,
                                        "e-visa needed; apply at least 7 days before travel"))

    @function_tool
    async def change_answer(self, step_id: str) -> dict:
        """Go back to change one earlier answer; the others are kept.

        Args:
            step_id: trip, travellers, budget or visa.
        """
        return await self.go_to_step(step_id, reason="caller correction")

    async def on_exit(self):
        # Flights first; a subagent with nothing left to redo passes the caller straight on.
        return await self.transfer_to("flights", reason="trip details ready")


class Flights(SubAgent):
    name = "flights"
    instructions = (
        "You find and hold flights. Read out at most three options, briefly: airline, stops, and price "
        "per person in rupees. Only quote what search_fares returned; never invent airlines or prices."
    )
    workflow = [
        Step(id="search", description="search fares", instructions="Search fares for their trip, then read out the options.",
             opening="Tell them you'll look for flights now, then call search_fares.", result=Fares),
        Step(id="choose", description="choose a fare", instructions="Help them pick a fare and a seat preference, then call hold_fare.",
             opening="Ask which option they'd like, and whether they prefer a window or an aisle.",
             result=HeldFare, max_attempts=2),
    ]

    async def on_enter(self, reason: str):
        trip = self.workflow.concierge.trip
        return f"Say you'll look for flights from {trip.origin} to {trip.destination}."

    @function_tool(cancel_on_interrupt=False)
    async def search_fares(self) -> dict:
        """Search the airlines for fares on the caller's route and dates. Takes about ten seconds."""
        await working(self, ["Checking the airlines now.", "Just comparing a few fares."], seconds=3)
        base = {"budget": 18000, "comfort": 32000, "luxury": 150000}[self.workflow.concierge.budget.style]
        fares = [
            {"fare_id": f"F{i}", "airline": airline, "stops": stops, "per_person_rupees": base + extra}
            for i, (airline, stops, extra) in enumerate([("SkyLine", 0, 9500), ("AeroJet", 1, 0), ("BlueWing", 1, 3500)])
        ]
        return await self.complete(Fares(fares))

    @function_tool
    async def hold_fare(self, fare_id: str, seat_preference: str) -> dict:
        """Hold the chosen fare for 24 hours.

        Args:
            fare_id: The fare they picked, e.g. F1.
            seat_preference: window, aisle, or no preference.
        """
        fares = {f["fare_id"]: f for f in self.workflow.search.fares}
        if fare_id not in fares:
            return {"error": f"Pick one of the fares you read out: {', '.join(fares)}."}
        await working(self, ["Holding those seats for you."])
        if random.random() < 0.3:
            return {"error": "That fare just sold out. Apologise and offer one of the others."}
        fare = fares[fare_id]
        total = fare["per_person_rupees"] * self.workflow.concierge.travellers.party_size
        return await self.complete(HeldFare(fare_id, fare["airline"], seat_preference, total))

    @function_tool
    async def change_trip(self, part: str) -> dict:
        """The caller wants to change their trip: where or when, who's travelling, or the budget.

        Args:
            part: trip (where or when), travellers, or budget.
        """
        return await change_trip(self, part)

    async def on_exit(self):
        if self.workflow.concierge.trip.nights > 0:
            return await self.transfer_to("hotels", reason="flight held")
        return await self.transfer_to("activities", reason="flight held, day trip")

    async def on_step_failed(self, step_id: str, reason: str):
        return {"error": "Fares keep selling out. Apologise, and offer to call them back when prices refresh."}


class Hotels(SubAgent):
    name = "hotels"
    instructions = (
        "You find hotels that suit their style and what's left of the budget. Offer at most three, "
        "with the price per night in rupees. Only quote what search_hotels returned."
    )
    workflow = [
        Step(id="area", description="neighbourhood", instructions="Ask what kind of area they'd like to stay in, then call record_area.",
             opening="Ask where in the city they'd like to stay: near the beach, the old town, somewhere quiet?", result=Area),
        Step(id="search", description="search hotels", instructions="Search hotels, then offer the options.",
             opening="Say you'll find some places to stay, then call search_hotels.", result=HotelOptions),
        Step(id="choose", description="book a room", instructions="Help them pick a hotel and a room type, then call book_room.",
             opening="Ask which hotel they like the sound of.", result=HotelRoom),
    ]

    async def on_enter(self, reason: str):
        return "Say the flight is held, and that now you'll find them somewhere to stay."

    @function_tool
    async def record_area(self, neighbourhood: str) -> dict:
        """Record where they'd like to stay.

        Args:
            neighbourhood: The area or kind of area, in their words.
        """
        return await self.complete(Area(neighbourhood.strip()))

    @function_tool(cancel_on_interrupt=False)
    async def search_hotels(self) -> dict:
        """Search hotels for every night of the trip. Takes about eight seconds."""
        await working(self, ["Looking at hotels in that area.", "Checking they're free every night."], seconds=3)
        flight = self.workflow.flights.choose
        left = self.workflow.concierge.budget.rupees - flight.total_rupees
        per_night = max(3500, left * 0.6 / max(self.workflow.concierge.trip.nights, 1))
        hotels = [{"hotel_id": f"H{i}", "name": name, "per_night_rupees": round(per_night * k, -2)}
                  for i, (name, k) in enumerate([("Harbour View", 1.0), ("Old Town Inn", 0.7), ("The Grand", 1.4)])]
        return await self.complete(HotelOptions(hotels, round(left)))

    @function_tool
    async def book_room(self, hotel_id: str, room_type: str) -> dict:
        """Reserve a room at the chosen hotel.

        Args:
            hotel_id: The hotel they picked, e.g. H1.
            room_type: e.g. double, twin, suite.
        """
        hotels = {h["hotel_id"]: h for h in self.workflow.search.hotels}
        if hotel_id not in hotels:
            return {"error": f"Pick one of the hotels you offered: {', '.join(hotels)}."}
        hotel = hotels[hotel_id]
        total = round(hotel["per_night_rupees"] * self.workflow.concierge.trip.nights)
        return await self.complete(HotelRoom(hotel_id, hotel["name"], room_type, total))

    @function_tool
    async def change_trip(self, part: str) -> dict:
        """The caller wants to change their trip: where or when, who's travelling, or the budget.

        Args:
            part: trip (where or when), travellers, or budget.
        """
        return await change_trip(self, part)

    async def on_exit(self):
        return await self.transfer_to("activities", reason="hotel reserved")

    @on_workflow("step_end")
    def on_step_end(self, data: dict):
        self.log.info("%s.%s done; to do: %s", data["agent"], data["step"], data["progress"]["to_do"])


class Activities(SubAgent):
    name = "activities"
    instructions = "You plan a relaxed day-by-day itinerary around what the caller loves doing."
    workflow = [
        Step(id="interests", description="interests", instructions="Find out what they love doing on holiday, then call record_interests.",
             opening="Ask what they most enjoy on a trip: food, the outdoors, museums, a bit of everything?", result=Interests),
        Step(id="itinerary", description="build itinerary", instructions="Build the plan, then give a short summary rather than every day.",
             opening="Say you'll sketch out a plan, then call build_itinerary.", result=Itinerary),
        Step(id="extras", description="add-ons", instructions="Offer an airport transfer and travel insurance; if they want either, call add_extras.",
             opening="Ask if they'd like an airport transfer or travel insurance.", result=Extras, optional=True),
    ]

    # async def on_enter(self, reason: str):
    #     return "Say that's the travel sorted, and now for the fun part: what they'll do there."

    @function_tool
    async def record_interests(self, interests: list[str]) -> dict:
        """Record what they enjoy, e.g. food, hiking, museums.

        Args:
            interests: What they enjoy, in a few words each.
        """
        return await self.complete(Interests(interests))

    @function_tool(cancel_on_interrupt=False)
    async def build_itinerary(self) -> dict:
        """Plan each day of the trip around their interests. Takes a little while for longer trips."""
        days = max(self.workflow.concierge.trip.nights, 1)
        interests = self.workflow.interests.interests
        plan = []
        for day in range(1, days + 1):
            if day == 1 or day % 4 == 0:
                await self.session.say("Starting on day one." if day == 1 else f"Putting together day {day}.",
                                       add_to_chat_context=False)
            await asyncio.sleep(1.0)
            plan.append({"day": day, "activity": interests[(day - 1) % len(interests)], "estimate_rupees": 6500})
        return await self.complete(Itinerary(plan))

    @function_tool
    async def add_extras(self, transfer: bool = False, insurance: bool = False) -> dict:
        """Add an airport transfer and/or travel insurance.

        Args:
            transfer: Add a return airport transfer.
            insurance: Add travel insurance.
        """
        return await self.complete(Extras(5000 if transfer else 0, 3500 if insurance else 0))

    @function_tool
    async def change_trip(self, part: str) -> dict:
        """The caller wants to change their trip: where or when, who's travelling, or the budget.

        Args:
            part: trip (where or when), travellers, or budget.
        """
        return await change_trip(self, part)

    async def on_exit(self):
        return await self.transfer_to("booking", reason="itinerary ready")


class Booking(SubAgent):
    name = "booking"
    instructions = (
        "You go over the quote, take payment through a secure link, and send the confirmations. "
        "Never ask for card numbers over the phone. Once they have their booking reference, ask if "
        "there's anything else; when there isn't, call end_call."
    )
    workflow = [
        Step(id="review", description="review quote", instructions="Give the total, and the main parts of it if they ask, then get a clear yes.",
             opening="Walk them through the total for the trip.", result=Quote),
        Step(id="pay", description="payment", instructions="Ask for their email, send the secure payment link, and wait while they pay.",
             opening="Ask for the best email address for the payment link and their tickets.", result=Payment, max_attempts=2),
        Step(id="confirm", description="confirmations", instructions="Issue the tickets and vouchers, then give them their booking reference.",
             opening="Say you're issuing everything now, then call issue_documents.", result=Documents),
    ]

    async def on_enter(self, reason: str):
        return f"Say everything's ready, and give them the total of {self.quote():,} rupees for the whole trip."

    def quote(self) -> int:
        flight = self.workflow.flights.choose
        room = self.workflow.hotels.choose if self.workflow.concierge.trip.nights else None
        itinerary, extras = self.workflow.activities.itinerary, self.workflow.activities.extras
        return round(flight.total_rupees + (room.total_rupees if room else 0)
                     + sum(d["estimate_rupees"] for d in itinerary.plan)
                     + (extras.transfer_rupees + extras.insurance_rupees if extras else 0))

    @function_tool
    async def accept_quote(self, accepted: bool) -> dict:
        """Record whether the caller is happy with the total.

        Args:
            accepted: True once they've clearly said yes.
        """
        if not accepted:
            return {"next_step": "Ask what they'd like to change. For where or when, travellers or budget, call change_trip."}
        return await self.complete(Quote(self.quote()))

    @function_tool(cancel_on_interrupt=False)
    async def take_payment(self, email: str) -> dict:
        """Email a secure payment link and wait while the caller pays. Must not be cut off halfway.

        Args:
            email: Where to send the link and the tickets. Read it back to confirm before calling.
        """
        if "@" not in email:
            return {"error": "That isn't an email address. Ask them to spell it out."}
        await working(self, ["I've sent the link to your email. Take your time; I'll wait here."], seconds=6)
        return await self.complete(Payment(f"pay_{uuid.uuid4().hex[:8]}", email.strip()))

    @function_tool(cancel_on_interrupt=False)
    async def issue_documents(self) -> dict:
        """Ticket the flight, confirm the hotel and tours, and email everything."""
        await working(self, ["Payment's through. Issuing your tickets.", "And confirming the hotel and tours."])
        return await self.complete(Documents(f"WL-{uuid.uuid4().hex[:6].upper()}", self.workflow.pay.email))

    @function_tool
    async def change_trip(self, part: str) -> dict:
        """The caller wants to change their trip: where or when, who's travelling, or the budget.

        Args:
            part: trip (where or when), travellers, or budget.
        """
        return await change_trip(self, part)

    async def on_step_failed(self, step_id: str, reason: str):
        return {"error": "The payment didn't go through twice. Say their holds last 24 hours, and they can call back to pay."}

    async def on_exit(self):
        return "Have a wonderful trip, and thanks for choosing Wanderlust Travel."


def make_host(job: AgentContext) -> Agent:
    now = datetime.now(IST)
    host = Agent(
        instructions=(
            "You are Wanderlust Travel's phone assistant: warm, relaxed and efficient. "
            f"Today is {now:%A, %d %B %Y}, and it's {now:%I:%M %p} in India (IST). Work out dates the "
            "caller gives, like 'next Friday' or 'the second week of December', from today. "
            "This is a phone call: keep each reply to a sentence or two, ask one thing at a time, and never "
            "use lists or symbols. All prices are in Indian rupees: say them the way people do in India, "
            "rounded, like 'about one lakh twenty thousand rupees'. Only give prices, airlines, hotels and "
            "rules that a tool returned. Never guess a detail the caller hasn't given; ask for it. If they "
            "want to change their dates, destination, travellers or budget, call change_trip. Call end_call "
            "only when the caller is done, never in the middle of planning."
        ),
        pipeline=Pipeline(stt=DeepgramSTT(), llm=GoogleLLM(), tts=CartesiaTTS(),
                          vad=SileroVAD(), turn_detector=TurnDetector()),
        tools=[end_call],  # shared: every subagent can use it
        subagents=[Concierge(), Flights(), Hotels(), Activities(), Booking()],  # the concierge opens the call
    )
    host.on("agent_handoff", lambda e: print("handoff:", e))
    host.on("step_end", lambda e: print("step done:", e))
    return host


def on_ready() -> None:
    zeroruntime.invoke(room=Room(name="Travel Agency", playground=True), metadata={"caller_name": "Asha"})


if __name__ == "__main__":
    zeroruntime.serve(make_host,
    on_ready=on_ready
    )
