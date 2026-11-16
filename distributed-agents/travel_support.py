# Travel support: a plain agent, with a refunds subagent in its own process.
# Refunds can hand the caller back to support, or straight to the front desk.
#
# Run alone it takes its own calls; when another agent transfers to it, it takes
# over that call in place.
#
# A quiet office ambience plays while support holds the call (and under refunds,
# the same team), and stops when the caller is handed elsewhere. It plays on the
# room's mixing track, which the room that starts the call opens:
# Room(background_audio=True).
#
#   python travel_support.py     (with front_desk.py and booking.py; see README.md)

import logging
import os

import zeroruntime
from dotenv import load_dotenv
from zeroruntime import Agent, Pipeline, RemoteAgent, RemoteAgentError, Room, SubAgent, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import DeepgramSTT, GoogleLLM, SarvamAITTS, SileroVAD

load_dotenv(override=True)
logger = logging.getLogger("travel-support")

FRONT_DESK = "front_desk"
BOOKING = "booking"

TRIPS = {"AB12": {"flight": "AI 101",
                  "status": "delayed by 2 hours", "fare": 6500, "days_out": 10}}


class Refunds(SubAgent):
    """A subagent of travel support, with no pipeline of its own: it runs on the call's."""

    name = "refunds"
    instructions = (
        "You are Sam from the Skyline Travel refunds team, on the phone. Be reassuring and clear, "
        "a sentence or two at a time, no lists or markdown. Work out what they'd get back with "
        "refund_quote and explain it simply, including any fee. If they're happy, call "
        "issue_refund. Afterwards, if they need more help with their trip call transfer_to_support; "
        "if they're all done call transfer_to_front_desk."
    )

    async def on_enter(self, reason: str):
        logger.info("[refunds] on_enter (%s)", reason)
        return f"Introduce yourself as Sam from refunds, and say you'll sort out the refund for {reason}."

    # Hooks that only log, while refunds holds the call.
    async def on_user_turn_start(self, transcript: str) -> None:
        logger.info("[refunds] caller said: %s", transcript)

    async def on_metrics_llm(self, data: dict) -> None:
        logger.info("[refunds] llm metrics: %s", data)

    @function_tool
    async def refund_quote(self, booking_reference: str) -> dict:
        """What they would get back for a trip.

        Args:
            booking_reference: Their booking reference, e.g. AB12.
        """
        trip = TRIPS.get(booking_reference.strip().upper())
        if trip is None:
            return {"error": "No booking with that reference. Ask them to read it again."}
        fee = 0 if trip["days_out"] > 14 else round(trip["fare"] * 0.2)
        return {"reference": booking_reference.upper(), "refund": trip["fare"] - fee, "fee": fee}

    @function_tool
    async def issue_refund(self, booking_reference: str) -> dict:
        """Refund the trip; they've agreed to the amount.

        Args:
            booking_reference: Their booking reference.
        """
        logger.info("[refunds] refunded %s", booking_reference)
        return {"reference": booking_reference.upper(), "status": "refunded"}

    @function_tool
    async def transfer_to_support(self) -> dict:
        """They need more help with their trip."""
        return await self.transfer_to("travel_support", reason="refund done")

    @function_tool
    async def transfer_to_front_desk(self) -> dict:
        """They're all done."""
        await self.session.say("All sorted. I'll pass you back to Ava.", add_to_chat_context=False)
        try:
            handoff = await self.transfer_to(RemoteAgent(FRONT_DESK), reason="refund done")
        except RemoteAgentError:
            # No one from the front desk could join: the caller stays here.
            await self.session.say("I'm sorry, our front desk isn't available right now.",
                                   add_to_chat_context=False)
            return {"transferred": False,
                    "note": "The front desk is unavailable. You've apologised; offer to help another way."}
        await self.session.stop_background_audio()  # the front desk has no ambience
        return handoff


class TravelSupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="travel_support",
            instructions=(
                "You are Maya from Skyline Travel support, on the phone with someone whose trip has "
                "hit a snag. Be calm and empathetic, a sentence or two at a time, no lists or "
                "markdown. Ask for their booking reference and check it with lookup_trip. If they "
                "want a different date, use rebook_flight. If they'd rather have their money back, "
                "call transfer_to_refunds. If they want to book something new, call transfer_to_booking. "
                "When they're all set, call transfer_to_front_desk."
            ),
            pipeline=Pipeline(stt=DeepgramSTT(), llm=GoogleLLM(), tts=SarvamAITTS(),
                              vad=SileroVAD(), turn_detector=TurnDetector()),
        )

    async def on_enter(self, reason: str):
        logger.info("[support] on_enter (%s)", reason)
        if reason != "refund done":  # back from refunds, it is still playing
            await self.session.play_background_audio(volume=0.3, looping=True)
        if reason == "call started":  # run alone, the call starts here
            return "Greet them as Maya from Skyline Travel support and ask how you can help."
        if reason == "refund done":
            return "Check whether there's anything else you can help with on their trip."
        return f"Introduce yourself as Maya from support, acknowledge {reason}, and ask for their booking reference."

    async def on_exit(self) -> None:
        logger.info("[support] call ended in travel support")

    # Hooks that only log: each runs here while travel support holds the call.
    async def on_user_turn_start(self, transcript: str) -> None:
        logger.info("[support] caller said: %s", transcript)

    async def on_agent_turn_end(self) -> None:
        logger.info("[support] finished speaking")

    async def on_metrics_llm(self, data: dict) -> None:
        logger.info("[support] llm metrics: %s", data)

    async def on_error(self, data: dict) -> None:
        logger.error("[support] pipeline error: %s", data)

    @function_tool
    async def lookup_trip(self, booking_reference: str) -> dict:
        """Look up a trip they already have.

        Args:
            booking_reference: Their booking reference, e.g. AB12.
        """
        trip = TRIPS.get(booking_reference.strip().upper())
        return {"found": False} if trip is None else {"found": True, **trip}

    @function_tool
    async def rebook_flight(self, booking_reference: str, new_date: str) -> dict:
        """Move their trip to another date.

        Args:
            booking_reference: Their booking reference.
            new_date: The date they'd prefer.
        """
        return {"reference": booking_reference.upper(), "flight": "AI 205", "date": new_date, "status": "rebooked"}

    @function_tool
    async def transfer_to_refunds(self, summary: str) -> dict:
        """Hand them to Sam in refunds, in this process.

        Args:
            summary: What they want refunded, in a few words.
        """
        return await self.transfer_to(Refunds(), reason=summary)

    @function_tool
    async def transfer_to_booking(self, summary: str):
        """Hand them to our booking team for something new.

        Args:
            summary: What they'd like to book, in a few words.
        """
        await self.session.say("Happy to help with that. Let me get you to our booking team.",
                               add_to_chat_context=False)
        try:
            handoff = await self.transfer_to(RemoteAgent(BOOKING), reason=summary)
        except RemoteAgentError:
            # No one from the booking team could join: the caller stays here.
            await self.session.say("I'm sorry, our booking team isn't available right now.",
                                   add_to_chat_context=False)
            return {"transferred": False,
                    "note": "The booking team is unavailable. You've apologised; offer to help another way."}
        # Only now that booking has joined: a transfer that fails keeps the ambience.
        await self.session.stop_background_audio()
        return handoff

    @function_tool
    async def transfer_to_front_desk(self) -> dict:
        """They're all set."""
        await self.session.say("Glad we could sort that out. I'll pass you back to Ava.", add_to_chat_context=False)
        try:
            handoff = await self.transfer_to(RemoteAgent(FRONT_DESK), reason="support finished")
        except RemoteAgentError:
            # No one from the front desk could join: the caller stays here.
            await self.session.say("I'm sorry, our front desk isn't available right now.",
                                   add_to_chat_context=False)
            return {"transferred": False,
                    "note": "The front desk is unavailable. You've apologised; offer to help another way."}
        await self.session.stop_background_audio()
        return handoff


if __name__ == "__main__":
    # The id other agents transfer to; serve() registers and takes transfers under it.
    os.environ.setdefault("ZERORUNTIME_AGENT_ID", "travel_support")
    # Run alone, its own calls need the mixing track too.
    zeroruntime.serve(TravelSupportAgent, room=Room(background_audio=True))
