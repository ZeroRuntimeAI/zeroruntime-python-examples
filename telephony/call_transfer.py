# Moving the caller to another number with a transfer_call function tool.
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "call-transfer-agent"


class CallTransferAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are the Call Transfer Agent which helps transfer an ongoing "
                "call to a new number. Use the transfer_call tool to transfer."
            ),
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello Buddy, How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye Buddy, Thank you for calling!")

    @function_tool
    async def transfer_call(self) -> dict:
        """Transfer the call to the configured number."""
        transfer_to = os.getenv("CALL_TRANSFER_TO", "")
        if not transfer_to:
            return {"ok": False, "reason": "CALL_TRANSFER_TO is not set"}
        try:
            result = await self.session.transfer_call(transfer_to)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "result": result}


def invoke_agent() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Call Transfer Agent", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(CallTransferAgent, on_ready=invoke_agent)
