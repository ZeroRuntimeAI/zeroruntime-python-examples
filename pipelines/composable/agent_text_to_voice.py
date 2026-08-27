# Text in, voice out -- Pipeline(llm=..., tts=...) infers LLM_TTS_ONLY. No STT
# and no VAD, because the input is typed rather than spoken.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, PubSubSubscribeConfig, Room
from zeroruntime.plugins import CartesiaTTS, GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "text-to-voice-agent")
IN_TOPIC = "CHAT"


room = Room(name="Text to Voice", playground=True)


class TextToVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant. Keep spoken answers short -- they "
                "are read aloud."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(llm=GoogleLLM(), tts=CartesiaTTS()),
        )

    async def on_enter(self) -> None:
        await self.session.subscribe_to_pubsub(
            PubSubSubscribeConfig(topic=IN_TOPIC, cb=self.on_chat)
        )
        await self.session.say("Hello. Type something and I will read my answer aloud.")

    async def on_chat(self, frame: dict, backlog: bool) -> None:
        text = str(frame.get("message") or "")
        if backlog or not text.strip():
            return

        logger.info("user typed: %s", text)
        await self.session.process_text(text)

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zeroruntime.invoke(
        AGENT_ID,
        room=room,
    )
    logger.info(
        "publish text on the %r topic to hear it answered", IN_TOPIC)


if __name__ == "__main__":
    zeroruntime.serve(TextToVoiceAgent, on_ready=on_ready)
