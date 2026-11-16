# Booking: an agent with a workflow. Its steps run in the runtime's workflow
# engine, which asks each in order; a tool's self.complete(...) finishes a step.
#
#   trip -> travellers -> confirm
#
# Run alone it takes its own calls; when the front desk transfers to it, it takes
# over that call in place, with its own model and voice (OpenAI + ElevenLabs).
# Handed back and forth, it picks the workflow up where it left off.
#
#   python booking.py     (with front_desk.py and travel_support.py; see README.md)

import logging
import os
import zlib
from dataclasses import dataclass

import zeroruntime
from dotenv import load_dotenv
from zeroruntime import Agent, Pipeline, RemoteAgent, RemoteAgentError, Step, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import DeepgramSTT, ElevenLabsTTS, OpenAILLM, SileroVAD
from zeroruntime.workflow import on_workflow

load_dotenv(override=True)
logger = logging.getLogger("booking")

FRONT_DESK = "front_desk"
FARES = {"goa": 6500, "delhi": 5200, "mumbai": 4800}


@dataclass
class Trip:
    destination: str
    date: str


@dataclass
class Travellers:
    count: int
    quote: int


@dataclass
class Confirmed:
    reference: str


STEPS = [
    Step(id="trip", description="where and when", result=Trip,
         instructions="Find out where they'd like to go and when, then call record_trip.",
         opening="Ask where they're thinking of going, and when."),
    Step(id="travellers", description="how many people", result=Travellers,
         instructions="Find out how many people are going, call record_travellers, and share the price.",
         opening="Ask who's coming along, or how many people in total."),
    Step(id="confirm", description="confirm", result=Confirmed,
         instructions="Recap the trip and price, and ask if they'd like to go ahead. If they do, call "
                      "confirm_booking, give them the reference, then call transfer_to_front_desk.",
         opening="Recap the trip and the price, and ask if they'd like to book it."),
]


class BookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="booking",
            instructions=(
                "You are Leo from the Skyline Travel booking team, on the phone with a caller. "
                "Be upbeat and easy-going, a sentence or two at a time, no lists or markdown. "
                "Repeat key details back naturally so they know you've got them right."
            ),
            pipeline=Pipeline(stt=DeepgramSTT(), llm=OpenAILLM(), tts=ElevenLabsTTS(),
                              vad=SileroVAD(), turn_detector=TurnDetector()),
            workflow=STEPS,
        )

    async def on_enter(self, reason: str):
        logger.info("[booking] on_enter (%s)", reason)
        if reason == "call started":  # run alone, the call starts here
            return "Greet them as Leo from Skyline Travel bookings."
        return f"Introduce yourself as Leo from bookings, and say you'd be happy to help with {reason}."

    async def on_exit(self):
        logger.info("[booking] every step is done")

    # Hooks that only log: each runs here while booking holds the call.
    async def on_user_turn_start(self, transcript: str) -> None:
        logger.info("[booking] caller said: %s", transcript)

    async def on_agent_turn_end(self) -> None:
        logger.info("[booking] finished speaking")

    async def on_metrics_llm(self, data: dict) -> None:
        logger.info("[booking] llm metrics: %s", data)

    async def on_error(self, data: dict) -> None:
        logger.error("[booking] pipeline error: %s", data)

    # The workflow's own hooks, called by the engine.
    @on_workflow("step_start")
    def step_started(self, data: dict) -> None:
        logger.info("[booking] step started: %s", data.get("step"))

    @on_workflow("step_end")
    def step_done(self, data: dict) -> None:
        logger.info("[booking] step done: %s; still to do: %s", data.get("step"),
                    (data.get("progress") or {}).get("to_do"))

    @function_tool
    async def record_trip(self, destination: str, date: str) -> dict:
        """Record where and when they're travelling.

        Args:
            destination: The city.
            date: The date, as they said it.
        """
        if destination.strip().lower() not in FARES:
            return {"error": f"We don't fly to {destination} yet. We fly to {', '.join(FARES)}."}
        return await self.complete(Trip(destination.strip().title(), date))

    @function_tool
    async def record_travellers(self, count: int) -> dict:
        """Record how many people are going, and price the trip.

        Args:
            count: 1 to 9.
        """
        if not 1 <= count <= 9:
            return {"error": "We can book 1 to 9 people at a time."}
        fare = FARES[self.workflow.trip.destination.lower()]
        quote = fare * count - (500 if count >= 4 else 0)  # a group discount
        return await self.complete(Travellers(count, quote))

    @function_tool
    async def confirm_booking(self) -> dict:
        """They want to go ahead: book it."""
        trip = self.workflow.trip
        reference = f"SKY{zlib.crc32(f'{trip.destination}|{trip.date}'.encode(
        )) % 10000:04d}"
        logger.info("[booking] booked %s on %s for %d: %s", trip.destination, trip.date,
                    self.workflow.travellers.count, reference)
        return await self.complete(Confirmed(reference))

    @function_tool
    async def transfer_to_front_desk(self) -> dict:
        """The booking is done, or they need something else."""
        await self.session.say("Lovely. I'll pass you back to Ava now.", add_to_chat_context=False)
        try:
            return await self.transfer_to(RemoteAgent(FRONT_DESK), reason="booking finished")
        except RemoteAgentError:
            # No one from the front desk could join: the caller stays here.
            await self.session.say("I'm sorry, our front desk isn't available right now.",
                                   add_to_chat_context=False)
            return {"transferred": False,
                    "note": "The front desk is unavailable. You've apologised; offer to help another way."}


if __name__ == "__main__":
    # The id other agents transfer to; serve() registers and takes transfers under it.
    os.environ.setdefault("ZERORUNTIME_AGENT_ID", "booking")
    zeroruntime.serve(BookingAgent)
