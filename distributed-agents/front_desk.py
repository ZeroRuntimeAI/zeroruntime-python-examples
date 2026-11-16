# The front desk: starts the call and owns it. It hands the caller to the policy
# desk in this process, and to bookings and travel support, which run in their
# own processes, possibly on other machines.
#
# Every agent here is served with zeroruntime.serve(): run alone, it takes its own
# calls; when another agent transfers to its id, the same process takes over that
# call in place (same session, same participant).
#
#   python booking.py & python travel_support.py & python front_desk.py
#
# front_desk.py starts the call and prints the playground link. See README.md.

import logging
import os

import zeroruntime
from dotenv import load_dotenv
from zeroruntime import Agent, Pipeline, RemoteAgent, RemoteAgentError, Room, SubAgent, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

load_dotenv(override=True)
logger = logging.getLogger("front-desk")

# The ids transfers name: each agent is served under its own (ZERORUNTIME_AGENT_ID).
FRONT_DESK = "front_desk"
BOOKING = "booking"
TRAVEL_SUPPORT = "travel_support"

MEMBERS = {
    "9876543210": {"name": "Asha Rao", "tier": "gold"},
    "9123456780": {"name": "Ravi Menon", "tier": "silver"},
}


class PolicyDesk(SubAgent):
    """A subagent in this process: transfer_to(PolicyDesk()) hands the call to it."""

    name = "policy"
    instructions = (
        "You are Nina from Skyline Travel, and you know our policies well. You're on the "
        "phone, so keep it conversational: a sentence or two at a time, no lists or markdown. "
        "Use cancellation_policy for the details, explain them simply, and check whether that "
        "answers their question. Once it does, call transfer_to_front_desk."
    )

    async def on_enter(self, reason: str):
        logger.info("[policy] on_enter (%s)", reason)
        return f"Introduce yourself as Nina, and help with their question about {reason}."

    async def on_user_turn_start(self, transcript: str) -> None:
        logger.info("[policy] caller said: %s", transcript)

    async def on_metrics_llm(self, data: dict) -> None:
        logger.info("[policy] llm metrics: %s", data)

    @function_tool
    async def cancellation_policy(self, kind: str) -> dict:
        """The cancellation rules for a booking.

        Args:
            kind: "flight" or "hotel".
        """
        rules = {"flight": "free within 24 hours of booking, then a 20% fee",
                 "hotel": "free until 48 hours before check-in"}
        return {"kind": kind, "rule": rules.get(kind.lower(), "ask whether it's a flight or a hotel")}

    @function_tool
    async def transfer_to_front_desk(self) -> dict:
        """Their policy question is answered."""
        return await self.transfer_to("front_desk", reason="policy question answered")


class FrontDesk(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="front_desk",
            instructions=(
                "You are Ava, the front desk at Skyline Travel, talking with a caller on the phone. "
                "Sound like a friendly person, not a script: warm, relaxed and brief, a sentence or "
                "two at a time, no lists or markdown. Find out what they need. If they give you "
                "their phone number, look them up with lookup_member and use their name. "
                "To book a new trip, call transfer_to_booking. For a problem with a trip they "
                "already have, like a delay, a change or a refund, call transfer_to_travel_support. "
                "For questions about our policies, call transfer_to_policy. Pass a short summary of "
                "what they want, so they don't have to repeat themselves."
            ),
            pipeline=Pipeline(stt=DeepgramSTT(), llm=GoogleLLM(), tts=CartesiaTTS(),
                              vad=SileroVAD(), turn_detector=TurnDetector()),
        )

    async def on_enter(self, reason: str):
        logger.info("[front desk] on_enter (%s)", reason)
        if reason == "call started":
            name = self.userdata.get("caller_name", "there")
            return f"Warmly greet {name}, introduce yourself as Ava from Skyline Travel, and ask what brings them in today."
        return "Welcome them back warmly and ask if there's anything else you can help with."

    async def on_exit(self) -> None:
        logger.info("[front desk] call ended")

    # Hooks that only log: each runs here while the front desk holds the call.
    async def on_user_turn_start(self, transcript: str) -> None:
        logger.info("[front desk] caller said: %s", transcript)

    async def on_agent_turn_end(self) -> None:
        logger.info("[front desk] finished speaking")

    async def on_metrics_llm(self, data: dict) -> None:
        logger.info("[front desk] llm metrics: %s", data)

    async def on_error(self, data: dict) -> None:
        logger.error("[front desk] pipeline error: %s", data)

    async def on_participant_joined(self, participant) -> None:
        logger.info("[front desk] joined: %s", participant)

    async def on_participant_left(self, participant) -> None:
        logger.info("[front desk] left: %s", participant)

    async def on_agent_activated(self, payload: dict) -> None:
        # Sent to every agent on the call, active or not.
        logger.info("[front desk] active agent: %s (from %s: %s)", payload.get("agent_id"),
                    payload.get("from_agent_id"), payload.get("reason"))

    @function_tool
    async def lookup_member(self, phone: str) -> dict:
        """Find the caller's membership.

        Args:
            phone: Their 10-digit phone number.
        """
        digits = "".join(c for c in phone if c.isdigit())[-10:]
        member = MEMBERS.get(digits)
        return {"found": False} if member is None else {"found": True, **member}

    @function_tool
    async def transfer_to_policy(self, question: str) -> dict:
        """Hand the caller to Nina, who knows our policies.

        Args:
            question: What they asked, in a few words.
        """
        return await self.transfer_to(PolicyDesk(), reason=question)

    @function_tool
    async def transfer_to_booking(self, summary: str):
        """Hand the caller to our booking team.

        Args:
            summary: What they'd like to book, in a few words.
        """
        # Covers the moment while the booking agent joins the call.
        await self.session.say("Sure, let me get you over to our booking team.", add_to_chat_context=False)
        try:
            return await self.transfer_to(RemoteAgent(BOOKING), reason=summary)
        except RemoteAgentError:
            # No one from the booking team could join: the caller stays here.
            await self.session.say("I'm sorry, our booking team isn't available right now.",
                                   add_to_chat_context=False)
            return {"transferred": False,
                    "note": "The booking team is unavailable. You've apologised; offer to help another way."}

    @function_tool
    async def transfer_to_travel_support(self, summary: str):
        """Hand the caller to travel support.

        Args:
            summary: What's wrong with their trip, in a few words.
        """
        await self.session.say("I'm sorry to hear that. Let me get you to our support team.",
                               add_to_chat_context=False)
        try:
            return await self.transfer_to(RemoteAgent(TRAVEL_SUPPORT), reason=summary)
        except RemoteAgentError:
            # No one from the support team could join: the caller stays here.
            await self.session.say("I'm sorry, our support team isn't available right now.",
                                   add_to_chat_context=False)
            return {"transferred": False,
                    "note": "The support team is unavailable. You've apologised; offer to help another way."}


def on_ready() -> None:
    # background_audio opens the room's mixing track, for travel support's ambience.
    started = zeroruntime.invoke(room=Room(name="Skyline Travel", playground=True, background_audio=True),
                                 metadata={"caller_name": "Asha"})
    print(f"call: {started.get('playground_url')}")


if __name__ == "__main__":
    # The id other agents transfer to; serve() registers and takes transfers under it.
    os.environ.setdefault("ZERORUNTIME_AGENT_ID", "front_desk")
    zeroruntime.serve(FrontDesk, on_ready=on_ready)
