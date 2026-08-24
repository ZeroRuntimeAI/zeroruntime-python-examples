# Handing a live call to a human, having first briefed them on it.
#
# The agent puts the caller on hold, dials a supervisor into a second room,
# reads them a summary of the call so far, and only then bridges the two.
#
# Requires a live SIP leg: warm transfer reads the caller's SIP call id off the
# room, so a playground-only session cannot reach the first phase. Start this,
# note the room id it prints, and point an inbound call at that room (or place
# an outbound one) before asking for a supervisor.

import os

import zeroruntime
from zeroruntime import Agent,Pipeline,Room,SIPDestination,WarmTransferConfig,function_tool
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD
from zeroruntime.inference import TurnDetector

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "warm-transfer-agent"
SUPERVISOR_JOIN_TIMEOUT = 120.0
BRIEFING_TIMEOUT = 180.0
TRANSFER_BUDGET = SUPERVISOR_JOIN_TIMEOUT + BRIEFING_TIMEOUT + 60.0
TOOL_TIMEOUT = int(TRANSFER_BUDGET) + 30


def _pipeline() -> Pipeline:
    return Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )


class CustomerServiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful customer service agent. If the caller asks to "
                "speak to a manager or supervisor, or their issue needs a human, "
                "call the escalate_to_human tool. Do not promise a transfer before "
                "the tool has come back."
            ),
            pipeline=_pipeline(),
            tool_timeout_seconds=TOOL_TIMEOUT,
        )

    async def on_enter(self) -> None:
        self.session.on_warm_transfer(callback=self._on_transfer_phase)
        await self.session.say("Hi, how can I help you today?")
        
        await self.session.play_background_audio(
                file=os.getenv(
                    "BACKGROUND_AUDIO_URL",
                    "https://cdn.zeroruntime.ai/zrt/bg-audio/bg-noise-1.ogg",
                ),
                volume=0.5,
                looping=True,
        )

    async def on_exit(self) -> None:
        pass

    def _on_transfer_phase(self, payload: dict) -> None:
        phase = getattr(payload["phase"], "value", payload["phase"])
        print(f"[warm transfer] {phase} {payload['data']}")

    @function_tool
    async def escalate_to_human(self, reason: str) -> str:
        """Escalate this call to a human supervisor with a warm transfer.

        Args:
            reason: Short description of why the escalation is happening.
        """
        routing_rule_id = os.getenv("WARM_TRANSFER_ROUTING_RULE_ID", "")
        call_to = os.getenv("WARM_TRANSFER_TO", "")
        call_from = os.getenv("WARM_TRANSFER_FROM", "")
        if not (routing_rule_id and call_to and call_from):
            return (
                "Transfer is not configured, so I cannot reach a supervisor. "
                "Keep helping the caller."
            )

        config = WarmTransferConfig(
            destination=SIPDestination(
                routing_rule_id=routing_rule_id,
                sip_call_to=call_to,
                sip_call_from=call_from,
            ),
            summary_llm=GoogleLLM(),
            briefing_pipeline=_pipeline(),
            supervisor_join_timeout=SUPERVISOR_JOIN_TIMEOUT,
            briefing_timeout=BRIEFING_TIMEOUT,
        )

        result = await self.session.warm_transfer(config, timeout=TRANSFER_BUDGET)
        if result.success:
            return "Connected to a supervisor."
        print(f"[warm transfer] failed at {result.phase}: {result.error}")
        return (
            "I couldn't reach a supervisor right now. "
            "Let me keep helping you in the meantime."
        )


def invoke_agent() -> None:
    started = zeroruntime.invoke(
        AGENT_ID, room=Room(name="Warm Transfer Demo", playground=True)
    )
    print(f"room_id={started['room_id']} -- point a SIP call at this room")


if __name__ == "__main__":
    zeroruntime.serve(CustomerServiceAgent,room=Room(recording=True,background_audio=True))
