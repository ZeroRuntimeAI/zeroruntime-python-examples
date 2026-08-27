# Voice in, text out -- Pipeline(stt=, llm=, vad=, turn_detector=) infers
# STT_LLM_ONLY. No TTS, so the agent listens and answers in text without ever
# speaking into the room.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, PubSubPublishConfig, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)

AGENT_ID = os.getenv("AGENT_ID", "voice-to-text-agent")
OUT_TOPIC = "AGENT_RESPONSE"

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)

@pipeline.on("user_turn_start")
async def on_user_turn_start(transcript: str) -> None:
    """What the caller said, as soon as they stopped saying it."""
    logger.info("heard: %s", transcript)


class VoiceToTextAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant listening to a call. Answer "
                "concisely in text."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        logger.info("listening")

    async def on_llm(self, data: dict) -> None:
        text = (data or {}).get("text", "")
        if not text.strip():
            return
        logger.info("answer: %s", text)
        await self.session.publish_to_pubsub(
            PubSubPublishConfig(topic=OUT_TOPIC, message=text)
        )

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Voice to Text", playground=True))
    logger.info("speak in the room; answers arrive on %r", OUT_TOPIC)


if __name__ == "__main__":
    zeroruntime.serve(VoiceToTextAgent, on_ready=on_ready)
