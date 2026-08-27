# Fixed phrases synthesised once and replayed: say(text, audio_data=pcm) plays
# the bytes and skips the TTS round trip. The audio must be PCM in the room's
# format, and the text still travels because it goes into the chat context.

import logging
import os
import pathlib

import zeroruntime
from zeroruntime import Agent, InterruptConfig, Pipeline, Room, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "cached-tts-agent")

GREETING = "Hi, you've reached support. How can I help?"
HOLD = "Let me check that for you, one moment."
GOODBYE = "Thanks for calling. Goodbye."

CACHE_DIR = pathlib.Path(os.getenv("TTS_CACHE_DIR", "tts_cache"))


class PhraseCache:
    """Raw PCM per phrase, read once at startup.

    Deliberately dumb: a dict and some files. The point of the example is
    ``audio_data=``, not the cache -- swap this for your own TTS vendor's SDK,
    or for a CDN fetch, and nothing else changes.
    """

    def __init__(self, directory: pathlib.Path) -> None:
        self._directory = directory
        self._audio: dict[str, bytes] = {}

    def preload(self, phrases: list[str]) -> None:
        for phrase in phrases:
            path = self._directory / f"{abs(hash(phrase))}.pcm"
            if path.exists():
                self._audio[phrase] = path.read_bytes()
                logger.info("cached %d bytes for %r", len(
                    self._audio[phrase]), phrase[:32])
            else:
                logger.info(
                    "no cache for %r -- it will be synthesised", phrase[:32])

    def fetch(self, phrase: str) -> bytes | None:
        """The bytes, or None to let the runtime synthesise it as usual."""
        return self._audio.get(phrase)


cache = PhraseCache(CACHE_DIR)
cache.preload([GREETING, HOLD, GOODBYE])


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a support agent. Answer questions about orders. Use "
                "check_order_status rather than guessing."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
                interrupt_config=InterruptConfig(mode="HYBRID", interrupt_min_words=2),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(GREETING, audio_data=cache.fetch(GREETING))

    async def on_exit(self) -> None:
        await self.session.say(GOODBYE, audio_data=cache.fetch(GOODBYE))

    @function_tool
    async def check_order_status(self, order_id: str) -> dict:
        """Look up an order.

        Args:
            order_id: The order number the caller gives you.
        """
        await self.session.say(HOLD, audio_data=cache.fetch(HOLD))

        return {"order_id": order_id, "status": "shipped", "eta": "Tuesday"}


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(name="Cached TTS", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(SupportAgent, on_ready=on_ready)
