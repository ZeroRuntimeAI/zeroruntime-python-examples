# The agent ends the call itself: an end_call function tool the model invokes
# when the caller asks to hang up.
import asyncio

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "agent-hangup"


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant that can answer questions. "
                "When the user asks to hang up, end the call, or stop the "
                "conversation, call the end_call function tool."
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
        await self.session.say("Hello, how can I help you today?")

    @function_tool
    async def end_call(self) -> dict:
        """End the call when the user asks to hang up or says goodbye."""
        asyncio.create_task(self._announce_and_hangup())
        return {"status": "ending_call"}

    async def _announce_and_hangup(self) -> None:
        await self.session.interrupt()
        await asyncio.sleep(1)
        handle = await self.session.say(
            "I am ending the call now.", interruptible=False
        )
        await handle
        await self.hangup(farewell="Goodbye!")


def invoke_agent() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Agent Hangup", playground=True))


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=invoke_agent)
