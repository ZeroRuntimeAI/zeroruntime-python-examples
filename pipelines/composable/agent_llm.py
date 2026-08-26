# Text in, text out -- Pipeline(llm=...) infers LLM_ONLY. Input arrives on one
# pubsub topic and answers go out on another, so the agent does not read its
# own replies back as new input.

import logging
import os

import zeroruntime
from zeroruntime import (
    Agent,
    Pipeline,
    PubSubPublishConfig,
    PubSubSubscribeConfig,
    Room,
    current_session,
)
from zeroruntime.plugins import GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "llm-only-agent")
IN_TOPIC = "CHAT"
OUT_TOPIC = "AGENT_RESPONSE"

pipeline = Pipeline(llm=GoogleLLM())


@pipeline.on("llm")
async def on_llm(data: dict) -> None:
    """The agent's answer, as text. With no TTS this is the only output there is."""
    text = (data or {}).get("text", "")
    if not text.strip():
        return
    logger.info("agent: %s", text)
    await current_session().publish_to_pubsub(
        PubSubPublishConfig(topic=OUT_TOPIC, message=text)
    )


async def on_pubsub_message(frame: dict, backlog: bool) -> None:
    """One frame on IN_TOPIC, handed to whichever agent is running."""
    await current_session().agent.on_chat(frame, backlog)


room = Room(name="LLM Only", playground=True)
room.subscribe_to_pubsub(PubSubSubscribeConfig(
    topic=IN_TOPIC, cb=on_pubsub_message))


class LlmAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant. Answer in text, concisely."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_chat(self, frame: dict, backlog: bool) -> None:
        """One frame on IN_TOPIC. The second parameter is what keeps the agent
        from answering everything typed before it joined."""
        text = str(frame.get("message") or "")
        if backlog or not text.strip():
            return

        logger.info("user: %s", text)
        await self.session.process_text(text)

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zeroruntime.invoke(
        AGENT_ID,
        room=room,
    )
    logger.info("publish text on %r; answers arrive on %r",
                IN_TOPIC, OUT_TOPIC)


if __name__ == "__main__":
    zeroruntime.serve(LlmAgent, on_ready=on_ready)
