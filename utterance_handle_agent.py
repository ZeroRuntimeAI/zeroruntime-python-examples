# UtteranceHandle and interruption-aware tools: say() returns once the request
# is queued, and awaiting the handle it returns waits for the audio to drain. A
# tool checking handle.interrupted can abandon work the caller talked over.
import asyncio

import zrt
from zrt import Agent, Pipeline, Room, UtteranceHandle, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import DeepgramSTT, CartesiaTTS, OpenAILLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "utterance-handle-agent"


class VoiceAgent(Agent):
    """A voice agent demonstrating UtteranceHandle and interruption-aware tools."""

    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant. You can answer questions and "
                "fetch weather information using the 'get_weather' tool. You can "
                "also perform a long-running task using the 'long_running_task' tool."
            ),
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=OpenAILLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_weather(self, latitude: str, longitude: str) -> dict:
        """Fetch the current weather for a location.

        Supports interruption: if the caller starts speaking while the agent is
        responding, the work is abandoned rather than spoken over.

        Args:
            latitude: Latitude of the location. Estimate it; do not ask.
            longitude: Longitude of the location. Estimate it; do not ask.
        """
        utterance: UtteranceHandle | None = self.session.current_utterance
        await asyncio.sleep(0.5)
        if utterance is not None and utterance.interrupted:
            logger.info("caller barged in; dropping the weather result")
            return {"status": "cancelled"}
        return {"latitude": latitude, "longitude": longitude, "temperature_c": 28}

    @function_tool
    async def long_running_task(self) -> dict:
        """Run a task that takes a while, narrating progress."""
        first = await self.session.say("This will take a moment.")
        await first

        for step in range(3):
            if self.session.current_utterance and self.session.current_utterance.interrupted:
                return {"status": "cancelled", "at_step": step}
            await asyncio.sleep(0.5)

        return {"status": "done"}


def invoke_agent() -> None:
    zrt.invoke(AGENT_ID, room=Room(name="Sandbox Agent", playground=True))


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=invoke_agent)
