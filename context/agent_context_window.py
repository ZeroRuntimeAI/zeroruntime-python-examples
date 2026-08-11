# Bounding a long call's context: ContextWindow summarises or truncates the
# older part of the conversation before each LLM turn and keeps recent turns
# verbatim. System messages, handoffs and config updates always survive.

import logging
import os

import zrt
from zrt import Agent, ContextWindow, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "context-window")

pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-2"),
    llm=GoogleLLM(model="gemini-2.5-flash"),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
    context_window=ContextWindow(
        max_tokens=1500,
        keep_recent_turns=4,
        summary_llm=GoogleLLM(model="gemini-2.5-flash"),
    ),
)


@pipeline.on("metrics.llm")
def on_llm_metrics(data: dict) -> None:
    """Worth watching here: compaction costs a model call on the turn it fires."""
    logger.info("llm %s", data)


class LongCallAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a patient support agent. The caller may talk for a long "
                "time. Refer back to what they told you earlier."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! Tell me what is going on, in as much detail as you like.")

    async def on_exit(self) -> None:
        history = await self.session.get_context_history()
        logger.info("conversation ended with %d items in context", len(history))


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Context Window", playground=True))


if __name__ == "__main__":
    zrt.serve(LongCallAgent, on_ready=on_ready)
