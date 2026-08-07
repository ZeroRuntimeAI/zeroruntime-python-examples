# Handing the call to a human after briefing them: summarise the call, dial the
# supervisor into a second room, brief them out loud, then bridge the caller
# across. The result carries the phase it reached, since any of them can fail.

import os

import zrt
from zrt import (
    Agent,
    Pipeline,
    Room,
    SIPDestination,
    WarmTransferConfig,
    WarmTransferPhase,
    function_tool,
)
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "warm-transfer")

ROUTING_RULE_ID = os.getenv("SIP_ROUTING_RULE_ID", "rr_xxxxxxxx")
SUPERVISOR_NUMBER = os.getenv("SUPERVISOR_NUMBER", "+1XXXXXXXXXX")
CALLER_ID = os.getenv("SIP_CALLER_ID", "+1XXXXXXXXXX")

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)


class CustomerServiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful customer service agent. If the caller asks to "
                "speak to a manager or supervisor, or their issue requires a "
                "human, call the escalate_to_human tool."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    @function_tool
    async def escalate_to_human(self, reason: str) -> str:
        """Escalate this call to a human supervisor with a warm transfer.

        Args:
            reason: Short description of why the escalation is happening.
        """
        logger.info("escalating: %s", reason)

        result = await self.session.warm_transfer(
            WarmTransferConfig(
                destination=SIPDestination(
                    routing_rule_id=ROUTING_RULE_ID,
                    sip_call_to=SUPERVISOR_NUMBER,
                    sip_call_from=CALLER_ID,
                ),
                summary_llm=GoogleLLM(model="gemini-2.5-flash"),
                summary_prompt=(
                    "Brief the supervisor in under 60 words: who is calling, "
                    "what they want, and what has been tried. No pleasantries."
                ),
                briefing_pipeline=Pipeline(
                    stt=DeepgramSTT(),
                    llm=GoogleLLM(),
                    tts=CartesiaTTS(),
                    vad=SileroVAD(),
                    turn_detector=TurnDetector(),
                ),
            )
        )

        if result.success:
            logger.info("transferred; briefing was: %s", result.summary)
            return "Connected to a supervisor."

        logger.warning("transfer stopped at %s: %s",
                       result.phase, result.error)
        return (
            "I couldn't reach a supervisor right now. "
            "Let me keep helping you in the meantime."
        )

    async def on_enter(self) -> None:
        session = self.session

        @session.on_warm_transfer()
        def any_phase(payload: dict) -> None:
            phase = payload["phase"]
            logger.info(
                "[warm transfer] %s %s",
                getattr(phase, "value", phase),
                payload["data"] or "",
            )

        @session.on_warm_transfer(WarmTransferPhase.SUMMARY_READY)
        def briefed(payload: dict) -> None:
            logger.info("[warm transfer] briefing: %s",
                        payload["data"].get("summary", ""))

        @session.on_warm_transfer(WarmTransferPhase.TRANSFER_COMPLETE)
        def done(payload: dict) -> None:
            logger.info("[warm transfer] -> complete")

        await session.say("Hi, how can I help you today?")

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    result = zrt.invoke(AGENT_ID, room=Room(
        name="Warm Transfer Demo", playground=True))
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    zrt.serve(CustomerServiceAgent, on_ready=on_ready)
