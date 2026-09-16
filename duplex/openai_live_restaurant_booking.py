# Restaurant booking on GPT-Live: a caller books a table at Saffron House.
#
# GPT-Live talks to the caller and hands booking requests to a backend OpenAI
# model, which calls the tools below. Your code steers the call through the
# session: when a full slot the caller wanted opens up, the caller hears about it.
#
# Saturday at 8 PM starts full and opens up SLOT_OPENS_AFTER_S into the call.

import asyncio
import itertools
import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, function_tool
from zeroruntime.plugins import OpenAIBackendConfig, OpenAILive

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "openai-live-restaurant-booking")

# the full Saturday 8 PM slot opens up this many seconds into the call
SLOT_OPENS_AFTER_S = float(os.getenv("SLOT_OPENS_AFTER_S", "45"))

DAYS = ["today", "tomorrow", "saturday", "sunday"]
TIMES = ["7 PM", "7:30 PM", "8 PM", "8:30 PM", "9 PM"]
FULL = {("saturday", "8 PM"), ("saturday", "8:30 PM")}
MAX_PARTY = 8
BOOKINGS: dict[str, dict] = {}
_booking_ids = itertools.count(101)


def _slot(day: str, time: str) -> tuple[str, str]:
    """"Saturday", "8:00 pm" -> ("saturday", "8 PM")."""
    time = " ".join(time.upper().replace(":00", "").replace("PM", " PM").split())
    return day.strip().lower(), time


class HostAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the friendly host at Saffron House, an Indian restaurant in "
                "Bengaluru. Keep replies short. We are open 7 PM to 11 PM and serve North "
                "and South Indian food, with plenty of vegetarian dishes. To book, you "
                "need the day, time, number of people and a name.\n"
                "Delegation policy:\n"
                "Backend tools:\n"
                "- Check if a table is free\n"
                "- Book a table\n"
                "- Cancel a booking\n"
                "Delegate to the backend when:\n"
                "- The caller asks for a table, gives booking details, or wants to cancel.\n"
                "Do not delegate to the backend when:\n"
                "- The caller asks about the menu or opening hours, or is just chatting.\n"
                "Never confirm a booking before the backend confirms it."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                llm=OpenAILive(
                    model="gpt-live-1",
                    voice="marin",
                    # the backend OpenAI model that calls the tools
                    config=OpenAIBackendConfig(
                        model="gpt-5.6-terra",
                        instructions=(
                            "Check availability before booking. If the time is full, offer "
                            "the other times. Read the details back and book only after the "
                            "customer says yes. Reply in one or two short sentences."
                        ),
                        tool_choice="auto", 
                        parallel_tool_calls=False,
                        reasoning_effort="low",
                    ),
                ),
            ),
        )
        # the full slot the caller asked for, if any
        self.wanted: tuple[str, str] | None = None
        self._slot_task: asyncio.Task | None = None

    async def on_enter(self) -> None:
        await self.session.say("Hello, thank you for calling Saffron House. How can I help you today?")
        self._slot_task = asyncio.create_task(self._open_slot_later())

    async def on_exit(self) -> None:
        if self._slot_task is not None:
            self._slot_task.cancel()

    async def _open_slot_later(self) -> None:
        """Stands in for another guest cancelling during the call."""
        await asyncio.sleep(SLOT_OPENS_AFTER_S)
        FULL.discard(("saturday", "8 PM"))
        print("[RESTAURANT] a Saturday 8 PM table just opened up")
        if self.wanted == ("saturday", "8 PM"):
            # said aloud at the next natural pause, in the model's own words
            await self.session.append_commentary(
                "A table for Saturday at 8 PM just opened up. Offer it to the caller."
            )

    @function_tool
    async def check_availability(self, day: str, time: str, party_size: int) -> dict:
        """Check if a table is free.

        Args:
            day: today, tomorrow, Saturday or Sunday.
            time: 7 PM, 7:30 PM, 8 PM, 8:30 PM or 9 PM.
            party_size: Number of people.
        """
        print(f"[TOOLCALL] check_availability({day}, {time}, {party_size})")
        day, time = _slot(day, time)
        if day not in DAYS or time not in TIMES:
            return {"available": False, "reason": "we take bookings from today to Sunday, 7 PM to 9 PM"}
        if party_size > MAX_PARTY:
            return {"available": False, "reason": f"we book up to {MAX_PARTY} people; bigger groups call the manager"}
        if (day, time) in FULL:
            self.wanted = (day, time)
            return {"available": False, "other_times": [t for t in TIMES if (day, t) not in FULL]}
        return {"available": True}

    @function_tool
    async def book_table(self, name: str, day: str, time: str, party_size: int, customer_said_yes: bool) -> dict:
        """Book a table, only after reading the details back and the customer saying yes.

        Args:
            name: The name for the booking.
            day: today, tomorrow, Saturday or Sunday.
            time: 7 PM, 7:30 PM, 8 PM, 8:30 PM or 9 PM.
            party_size: Number of people.
            customer_said_yes: True only if the customer confirmed these details.
        """
        print(f"[TOOLCALL] book_table({name}, {day}, {time}, {party_size}, yes={customer_said_yes})")
        day, time = _slot(day, time)
        if day not in DAYS or time not in TIMES or (day, time) in FULL or party_size > MAX_PARTY:
            return {"booked": False, "reason": "that table is not free; check availability first"}
        if not customer_said_yes:
            return {"booked": False, "reason": "read the details back and ask the customer to confirm"}
        booking_id = str(next(_booking_ids))
        BOOKINGS[booking_id] = {"name": name, "day": day, "time": time, "party_size": party_size}
        self.wanted = None
        # quiet context, so the model can answer questions about it later
        await self.session.append_thinking(
            f"Booking {booking_id} is confirmed: {name}, {party_size} people, {day} at {time}."
        )
        return {"booked": True, "booking_id": booking_id}

    @function_tool
    async def cancel_booking(self, booking_id: str) -> dict:
        """Cancel a booking by its booking number.

        Args:
            booking_id: The booking number, such as 101.
        """
        print(f"[TOOLCALL] cancel_booking({booking_id})")
        if BOOKINGS.pop(booking_id.strip(), None) is None:
            return {"cancelled": False, "reason": "no booking with that number"}
        return {"cancelled": True}


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Saffron House Bookings", playground=True))

if __name__ == "__main__":
    zeroruntime.serve(HostAgent, on_ready=on_ready)
