# The mirror of cascade_to_realtime_handoff.py: a live call moving from a
# realtime model back to stt/llm/tts, to buy back a specific voice, a specialist
# STT or a cheaper model. Same two traps -- detached, and idempotent.

import asyncio
import logging
import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GeminiRealtime, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)

AGENT_ID = os.getenv("AGENT_ID", "realtime-support")


def make_cascade_pipeline() -> Pipeline:
    """The whole pipeline. Every slot named -- omitting one empties it."""
    return Pipeline(
        stt=DeepgramSTT(model="nova-2"),
        llm=GoogleLLM(model="gemini-2.5-flash"),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )


class RealtimeSupportAgent(Agent):
    def __init__(self) -> None:
        self._switched = False
        self._switch_task: "asyncio.Task | None" = None
        super().__init__(
            instructions=(
                "You are a support agent. If the caller asks for a different "
                "voice, or for a cheaper or more configurable mode, call "
                "switch_to_cascade."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                realtime=GeminiRealtime(
                    model="gemini-3.1-flash-live-preview",
                    config={"voice": "Leda", "response_modalities": ["AUDIO"]},
                )
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi, you've reached support. How can I help?")

    async def on_exit(self) -> None:
        logger.info("call finished (cascade=%s)", self._switched)

    @function_tool
    async def switch_to_cascade(self) -> dict:
        """Switch to a cascade pipeline with a configurable voice.

        Call this when the caller asks for a different voice, or for a cheaper
        or more configurable mode.
        """
        if self._switched:
            logger.info("already on cascade; ignoring repeat switch")
            return {"status": "already on the cascade pipeline"}
        self._switched = True

        async def _do_switch() -> None:
            mode = await self.session.change_pipeline(make_cascade_pipeline())
            logger.info("now on %s", mode)
            await self.session.say(
                "Done -- I've switched. I still have our whole conversation."
            )

        self._switch_task = asyncio.create_task(_do_switch())
        return {"status": "switching to the cascade pipeline"}


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Realtime to Cascade", playground=True))


if __name__ == "__main__":
    zrt.serve(RealtimeSupportAgent, on_ready=on_ready)
