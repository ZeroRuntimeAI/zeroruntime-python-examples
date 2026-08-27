# The turn lifecycle of a realtime call, through the same observation hooks a
# cascade pipeline reports. Handlers run in this process and their return value
# is discarded, so a slow one cannot stall a call.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.plugins import GeminiLiveConfig, GeminiRealtime


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "realtime-hooks-agent")

pipeline = Pipeline(
    realtime=GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        config=GeminiLiveConfig(
                        voice="Leda", response_modalities=["AUDIO"]
                    ),
    )
)


@pipeline.on("user_turn_start")
async def on_user_turn_start(transcript: str) -> None:
    logger.info("[USER TURN START] %s", transcript)


@pipeline.on("user_turn_end")
async def on_user_turn_end() -> None:
    logger.info("[USER TURN END]")


@pipeline.on("agent_turn_start")
async def on_agent_turn_start() -> None:
    logger.info("[AGENT TURN START]")


@pipeline.on("agent_turn_end")
async def on_agent_turn_end() -> None:
    logger.info("[AGENT TURN END]")


@pipeline.on("llm")
async def on_agent_text(data: dict) -> None:
    """The answer as text, even though the model is speaking it.

    A realtime model produces audio; this is the transcript of what it said.
    Useful for logging a call whose audio you are not keeping.
    """
    text = (data or {}).get("text", "")
    if text:
        logger.info("agent said: %s", text)


class MyVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Realtime Pipeline Hooks", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(MyVoiceAgent, on_ready=on_ready)
