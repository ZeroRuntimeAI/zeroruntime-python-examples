

import logging
import os

import zeroruntime
from zeroruntime import (
    Agent,
    Participant,
    Pipeline,
    PubSubPublishConfig,
    PubSubSubscribeConfig,
    Room,
    function_tool,
)
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


TOPIC = "CHAT"

AGENT_ID = os.getenv("AGENT_ID", "chat-agent")


room = Room(name="Chat Agent", playground=True)

pipeline = Pipeline(stt=DeepgramSTT(), llm=GoogleLLM(), tts=CartesiaTTS())


class ChatAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant in a room's text chat. You can post "
                "messages to the room's chat when asked. Keep replies short."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_llm(self, data: dict) -> None:
        """Every answer the agent produces, echoed back into the chat."""
        text = str((data or {}).get("text") or "").strip()
        if text:
            await self.session.publish_to_pubsub(
                PubSubPublishConfig(topic=TOPIC, message=text)
            )

    @function_tool
    async def send_chat_message(self, message: str) -> dict:
        """Send a message to everyone in the room. Use when the caller asks you
        to post, announce, or share something with the room.

        Args:
            message: The text to post.
        """
        await self.session.publish_to_pubsub(
            PubSubPublishConfig(topic=TOPIC, message=message)
        )
        return {"status": "sent", "topic": TOPIC}

    async def on_enter(self) -> None:
        await self.session.subscribe_to_pubsub(
            PubSubSubscribeConfig(topic=TOPIC, cb=self.on_chat)
        )
        await self.session.say("Hi! Say something, or type in the room chat.")

    async def on_chat(self, message: dict, backlog: bool) -> None:
        """One frame on TOPIC, as the transport delivered it."""
        logger.info("Pubsub message received: %s", message)
        text = str((message or {}).get("message") or "")
        if text.strip():
            await self.session.process_text(text)

    async def on_participant_joined(self, participant: Participant) -> None:
        logger.info("joined: %s (%s)",
                    participant.name or "anonymous", participant.id)
        if participant.name:
            await self.session.say(f"Welcome, {participant.name}.")

    async def on_participant_left(self, participant: Participant) -> None:
        logger.info("left: %s", participant.name or participant.id)

    async def on_exit(self) -> None:
        logger.info("session finished")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID)


if __name__ == "__main__":
    zeroruntime.serve(ChatAgent, on_ready=on_ready, room=room)
