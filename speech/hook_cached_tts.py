"""Cached TTS, with the cache decision written here in the agent.

The runtime owns the audio: it stores the bytes, serves them, and synthesizes
when there is nothing to serve. It does not own the *rule*. Before a turn is
spoken it hands the text to the `tts` hook below, and whatever phrase that hook
names is the phrase whose stored audio is played. Return None and the turn is
synthesized as usual. No audio crosses -- text out, a phrase back.

Run the runtime with a cache enabled:

    ZERORUNTIME_TTS_CACHE_REDIS_URL=redis://localhost:6379/0
    ZERORUNTIME_TTS_CACHE_SHARED_POOL=demo

Seed the same phrases with the same TTS config first, or the first call of each
is a miss that seeds itself:

    python tools/seed_tts_cache.py --pool demo --plugin cartesia \
        --model sonic-2 --voice-id <id> --phrases phrases.txt
"""

import logging
import os
import unicodedata
import re

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)

AGENT_ID = os.getenv("AGENT_ID", "hook-cached-tts")

MODEL = "sonic-2"
VOICE_ID = "f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d"

PHRASES = [
    "Thanks for calling Northwind support. How can I help you today?",
    "Sure, let me check that for you.",
    "One moment please.",
    "Thanks for calling Northwind. Have a great day!",
]


def normalize(text: str) -> str:
    """Fold the ways a model respells a phrase it was told to say verbatim.

    Observed drift: `...` emitted as U+2026, and the space after a full stop
    dropped ("support.How"). Whitespace is removed entirely rather than
    collapsed, because a missing space cannot be restored by collapsing. This
    only loosens *matching* -- `match()` hands back the phrase as written, so
    the cache key stays exact and two phrases that differ only in spacing would
    have to be spelled apart some other way.
    """
    text = unicodedata.normalize("NFKC", text).replace("\u2026", "...")
    return "".join(text.split()).casefold()


PHRASES_BY_NORM = {normalize(phrase): phrase for phrase in PHRASES}


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(model=MODEL, voice=VOICE_ID),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    @pipeline.on("tts")
    async def cache_lookup(text: str) -> str | None:
        """Return the canonical phrase to serve from cache, or None to synthesize."""
        return PHRASES_BY_NORM.get(normalize(text))

    return pipeline


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a phone support agent for Northwind. Keep replies short. "
                "When one of these lines fits, say it exactly as written, with no "
                "words before or after:\n" + "\n".join(PHRASES)
            ),
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say(PHRASES[0])

    async def on_exit(self) -> None:
        await self.session.say(PHRASES[-1])


def invoke_agent() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(name="Hook Cached TTS", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(
        lambda job=None: SupportAgent(),
        on_ready=invoke_agent,
        capacity=2,
        room=Room(name="Hook Cached TTS", playground=True),
    )
