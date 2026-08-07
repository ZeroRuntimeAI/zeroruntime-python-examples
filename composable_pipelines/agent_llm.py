# Text in, text out -- Pipeline(llm=...) infers LLM_ONLY. Input arrives on one
# pubsub topic and answers go out on another, so the agent does not read its
# own replies back as new input.

import os

import zrt
from zrt import Agent, Pipeline, Room, RoomMessage
from zrt.plugins import GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "llm-only-agent")
IN_TOPIC = "CHAT"
OUT_TOPIC = "AGENT_RESPONSE"

pipeline = Pipeline(llm=GoogleLLM())

session: "zrt.Session | None" = None


@pipeline.on("llm")
async def on_llm(data: dict) -> None:
    """The agent's answer, as text. With no TTS this is the only output there is."""
    text = (data or {}).get("text", "")
    if not text.strip() or session is None:
        return
    logger.info("agent: %s", text)
    await session.publish(OUT_TOPIC, text)


class LlmAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant. Answer in text, concisely."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        global session
        session = self.session

    async def on_message(self, message: RoomMessage) -> None:
        if message.backlog or message.topic != IN_TOPIC:
            return
        if not message.text.strip():
            return

        logger.info("user: %s", message.text)
        await self.session.process_text(message.text)

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    result = zrt.invoke(
        AGENT_ID,
        room=Room(name="LLM Only", playground=True, subscribe=[IN_TOPIC]),
    )
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])
        logger.info("publish text on %r; answers arrive on %r",
                    IN_TOPIC, OUT_TOPIC)


if __name__ == "__main__":
    zrt.serve(LlmAgent, on_ready=on_ready)
