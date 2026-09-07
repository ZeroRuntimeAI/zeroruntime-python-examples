# Grounding a realtime agent on your own documents, using Vertex AI RAG Engine.
# The corpus is retrieved from inside the Live session -- the model decides when
# to look something up, so there is no tool to write and no retrieval step in
# the pipeline. Vertex only: RAG Engine has no Gemini Developer API equivalent,
# which is why vertexai=True is not optional here.
#
# The corpus itself is not created by zeroruntime. Run setup_corpus() once (see
# the block at the bottom), keep the resource name it prints, and point
# RAG_CORPUS at it.

import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.plugins import (
    GeminiLiveConfig,
    GeminiRealtime,
    VertexAIConfig,
    VertexRagConfig,
)

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "realtime-rag")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
RAG_CORPUS = os.getenv("RAG_CORPUS")


class MyVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Answer from the documents "
                "you have been given whenever they cover the question, and say "
                "so plainly when they do not."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                realtime=GeminiRealtime(
                    model="gemini-3.1-flash-live-preview",
                    vertexai=True,
                    vertexai_config=VertexAIConfig(
                        project_id=PROJECT_ID,
                        location=LOCATION,
                    ),
                    rag_config=VertexRagConfig(
                        rag_corpora=[RAG_CORPUS],
                        similarity_top_k=5,
                    ),
                    config=GeminiLiveConfig(
                        voice="Leda",
                        response_modalities=["AUDIO"],
                    ),
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, ask me anything about your documents.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(name="Realtime RAG", playground=True))


# --------------------------------------------------------------------------
# One-time corpus setup. NOT part of the agent and NOT a zeroruntime API: this
# talks to Vertex AI directly and needs `pip install google-cloud-aiplatform`.
# Run it once, put the printed name in RAG_CORPUS, then forget about it.
#
#     python context/realtime_rag.py setup ./handbook.pdf ./faq.md
# --------------------------------------------------------------------------
def setup_corpus(paths: list[str]) -> str:
    import vertexai
    from vertexai import rag

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    corpus = rag.create_corpus(
        display_name="zeroruntime-realtime-rag",
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    # A Google first-party embedding model is required.
                    publisher_model="publishers/google/models/text-embedding-005",
                )
            )
        ),
    )
    rag.import_files(corpus.name, paths, chunk_size=512, chunk_overlap=100)
    return corpus.name


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("RAG_CORPUS=" + setup_corpus(sys.argv[2:]))
        raise SystemExit(0)

    if not RAG_CORPUS:
        raise SystemExit(
            "RAG_CORPUS is not set. Create a corpus first:\n"
            "    python context/realtime_rag.py setup <file> [<file> ...]"
        )

    zeroruntime.serve(MyVoiceAgent, on_ready=on_ready)
