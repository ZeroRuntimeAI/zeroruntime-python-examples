# Long-term memory with mem0. Each user turn searches the store for what they
# just said and injects the hits as a system message; the exchange is written
# back afterwards. The store is reached from this process, with your own key.

import os
from typing import Optional

import httpx

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "agent-memory")
USER_ID = os.getenv("MEM0_USER_ID", "demo-user")


class Mem0Memory:
    """Thin mem0 client. Swap it for a vector index or your CRM -- it runs here."""

    STORE_KEYWORDS = (
        "remember", "my name", "i like", "i dislike", "favorite",
        "i prefer", "i love", "i hate", "i'm", "i am", "i work",
    )

    def __init__(self, api_key: str, user_id: str) -> None:
        self.user_id = user_id
        self._client = httpx.AsyncClient(
            base_url="https://api.mem0.ai",
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            timeout=10.0,
        )

    async def search(self, query: str, top_k: int = 5) -> list[str]:
        try:
            r = await self._client.post(
                "/v1/memories/search/",
                json={"query": query, "user_id": self.user_id, "top_k": top_k},
            )
            r.raise_for_status()
            body = r.json()
            results = body if isinstance(
                body, list) else body.get("results", [])
            return [
                e.get("memory", "")
                for e in results
                if isinstance(e, dict) and e.get("memory", "").strip()
            ]
        except Exception:
            logger.warning("memory search failed", exc_info=True)
            return []

    async def store(self, user_msg: str, assistant_msg: Optional[str] = None) -> None:
        messages = [{"role": "user", "content": user_msg}]
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
        try:
            r = await self._client.post(
                "/v1/memories/", json={"messages": messages, "user_id": self.user_id}
            )
            r.raise_for_status()
        except Exception:
            logger.warning("memory store failed", exc_info=True)


mem0_key = os.getenv("MEM0_API_KEY")
memory = Mem0Memory(api_key=mem0_key, user_id=USER_ID) if mem0_key else None
if memory is None:
    logger.warning("MEM0_API_KEY not set -- running without memory")

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)

session: Optional[zrt.Session] = None
pending_msg: Optional[str] = None


@pipeline.on("user_turn_start")
async def on_user(transcript: str) -> None:
    """Runs the moment the caller stops talking, before the LLM generates."""
    global pending_msg
    pending_msg = transcript
    if memory is None or session is None:
        return

    relevant = await memory.search(transcript)
    if not relevant:
        return

    facts = "\n".join(f"- {m}" for m in relevant)
    await session.add_message(
        "system",
        f"Relevant memories about this user:\n{facts}\n\n"
        "Use these to answer personally.",
    )
    logger.info("injected %d memories", len(relevant))


@pipeline.on("llm")
async def on_llm(data: dict) -> None:
    """The agent's reply, paired with what prompted it."""
    global pending_msg
    if memory is None or not pending_msg:
        pending_msg = None
        return
    await memory.store(pending_msg, data.get("text", ""))
    pending_msg = None


class PersonalAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly personal assistant. You remember things users "
                "tell you like their name, preferences, and interests. Use what you "
                "know to make conversations feel personal. Keep responses short and "
                "conversational."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        global session
        session = self.session
        await self.session.say("Hey! Welcome back. How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Bye! I'll remember everything for next time.")


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Personal Assistant", playground=True))


if __name__ == "__main__":
    zrt.serve(PersonalAssistant, on_ready=on_ready)
