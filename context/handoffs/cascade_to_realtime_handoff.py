# Switching a live call from a cascade pipeline to a realtime model. The swap
# must be detached, since it tears down the pipeline running the tool that
# called it, and idempotent, since the new pipeline still has that tool.

import asyncio
import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GeminiRealtime, GoogleLLM, SileroVAD
from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "support")


def make_realtime_pipeline() -> Pipeline:
    """The whole pipeline, not a patch -- realtime replaces stt/llm/tts."""
    return Pipeline(
        realtime=GeminiRealtime(
            model="gemini-3.1-flash-live-preview",
            config={"voice": "Leda", "response_modalities": ["AUDIO"]},
        )
    )


class SupportAgent(Agent):
    def __init__(self) -> None:
        self._switched = False
        self._switch_task: "asyncio.Task | None" = None
        super().__init__(
            instructions=(
                "You are a support agent. Answer questions about orders. If the "
                "caller asks for faster or more natural responses, call "
                "switch_to_realtime."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi, you've reached support. How can I help?")

    async def on_exit(self) -> None:
        logger.info("call finished (realtime=%s)", self._switched)

    @function_tool
    async def lookup_order(self, order_id: str) -> dict:
        """Look up the status of an order.

        Args:
            order_id: The order number the caller gives you.
        """
        return {"order_id": order_id, "status": "shipped", "eta": "Tuesday"}

    @function_tool
    async def switch_to_realtime(self) -> dict:
        """Switch the conversation to a low-latency realtime voice model.

        Call this when the caller asks for faster or more natural responses.
        """
        if self._switched:
            logger.info("already on realtime; ignoring repeat switch")
            return {"status": "already on the realtime pipeline"}
        self._switched = True

        async def _do_switch() -> None:
            mode = await self.session.change_pipeline(make_realtime_pipeline())
            logger.info("now on %s", mode)
            await self.session.say(
                "Done -- I've switched to realtime mode. I still have our whole "
                "conversation, so let's keep going."
            )

        self._switch_task = asyncio.create_task(_do_switch())

        return {"status": "switching to the realtime pipeline"}


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Cascade to Realtime", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(SupportAgent, on_ready=on_ready)
