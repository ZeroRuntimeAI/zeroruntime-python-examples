# Voice in, text out -- Pipeline(stt=, llm=, vad=, turn_detector=) infers
# STT_LLM_ONLY. No TTS, so the agent listens and answers in text without ever
# speaking into the room.

import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = os.getenv("AGENT_ID", "voice-to-text-agent")
OUT_TOPIC = "AGENT_RESPONSE"

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)

session: "zrt.Session | None" = None


@pipeline.on("user_turn_start")
async def on_user_turn_start(transcript: str) -> None:
    """What the caller said, as soon as they stopped saying it."""
    logger.info("heard: %s", transcript)


@pipeline.on("llm")
async def on_llm(data: dict) -> None:
    """The agent's answer. With no TTS this is the only output there is --
    without publishing it somewhere, this agent would think in silence."""
    text = (data or {}).get("text", "")
    if not text.strip() or session is None:
        return
    logger.info("answer: %s", text)
    await session.publish(OUT_TOPIC, text)


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
        global session
        session = self.session
        logger.info("listening")

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Voice to Text", playground=True))
    logger.info("speak in the room; answers arrive on %r", OUT_TOPIC)


if __name__ == "__main__":
    zrt.serve(VoiceToTextAgent, on_ready=on_ready)
