# Answering from your own documents rather than the model's own knowledge:
# KnowledgeBase names the knowledge documents uploaded in the dashboard, and
# every user turn searches all of them and puts the best sections in front of
# the model. The search runs in the runtime, so nothing here calls an API.

import logging
import os

import zeroruntime
from zeroruntime import Agent, KnowledgeBase, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "knowledge-base")

# Replace these with your own document ids, from the dashboard's knowledge page.
KNOWLEDGE_IDS = [
    doc_id.strip()
    for doc_id in os.getenv("KNOWLEDGE_IDS", "<knowledge-id-1>,<knowledge-id-2>").split(",")
    if doc_id.strip()
]

pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-2"),
    llm=GoogleLLM(model="gemini-2.5-flash"),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
    knowledge_base=KnowledgeBase(
        knowledge_ids=KNOWLEDGE_IDS,
        top_k=5,
        # A question the documents cannot answer is filed in the dashboard's open
        # questions; answering it there is what the next re-index picks up.
        report_unanswered=True,
    ),
)


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a support agent. Answer from the context you are given "
                "and nothing else. When it does not cover the question, say so "
                "and offer to pass the question on."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Ask me anything about our documentation.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(name="Knowledge Base", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(SupportAgent, on_ready=on_ready)
