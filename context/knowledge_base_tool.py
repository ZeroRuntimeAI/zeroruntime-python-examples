import logging

import zeroruntime
from zeroruntime import Agent, KnowledgeBase, Pipeline, Room, function_tool
from zeroruntime.plugins import GeminiLiveConfig, GeminiRealtime

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


kb = KnowledgeBase(knowledge_ids=["9eslt"], top_k=5)


@function_tool
async def search_docs(query: str) -> str:
    """Search the product documentation and support policies."""
    return await kb.retrieve(query, top_k=5, context_prefix="Product documentation and support policies:\n\n")


pipeline = Pipeline(
    llm=GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        config=GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"]),
    ),
)


@pipeline.metrics.on("knowledge_base")
def on_knowledge_base_metrics(data: dict) -> None:
    logger.info("knowledge_base %s", data)


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a support agent. For any question about our products or "
                "policies, call search_docs and answer only from what it returns. "
                "When it does not cover the question, say so."
            ),
            pipeline=pipeline,
            tools=[search_docs],
        )

    @function_tool
    async def get_weather(self, latitude: str, longitude: str) -> dict:
        """Fetch the current weather for a location.

        Args:
            latitude: Latitude of the location. Estimate it; do not ask.
            longitude: Longitude of the location. Estimate it; do not ask.
        """
        return {"latitude": latitude, "longitude": longitude, "temperature_c": 28}



    async def on_enter(self) -> None:
        await self.session.say("Hi! Ask me anything about our documentation.")


def on_ready() -> None:
    zeroruntime.invoke(room=Room(name="Knowledge Base Tool", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(SupportAgent, on_ready=on_ready)
