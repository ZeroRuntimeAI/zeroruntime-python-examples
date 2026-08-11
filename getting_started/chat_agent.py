# An agent you only type at: Pipeline(llm=...) infers LLM_ONLY, so there is no
# STT, TTS or VAD -- chat text goes in and a tool posts back. The chat topic is
# named at join, because the room does not exist until the agent has connected.

import asyncio
import logging
import os

import zrt
from zrt import Agent, Participant, Pipeline, Room, RoomMessage, function_tool
from zrt.plugins import GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


TOPIC = "CHAT"


@function_tool
async def send_chat_message(message: str) -> dict:
    """Send a message to everyone in the room. Use when the caller asks you to
    post, announce, or share something with the room.

    Args:
        message: The text to post.
    """
    await zrt.current_session().publish(TOPIC, message)
    return {"status": "sent", "topic": TOPIC}


class ChatAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant in a room's text chat. You can post "
                "messages to the room's chat when asked. Keep replies short."
            ),
            agent_id=os.getenv("AGENT_ID", "chat-agent"),
            pipeline=Pipeline(llm=GoogleLLM(model="gemini-2.5-flash")),
            tools=[send_chat_message],
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Say something, or type in the room chat.")

    async def on_message(self, message: RoomMessage) -> None:
        """One frame on a subscribed topic.

        ``backlog`` is checked first and it matters: subscribing replays whatever
        was already in the topic, so without this the agent answers every message
        sent before it joined, one after another, the moment the call connects.
        """
        if message.backlog:
            logger.info("[history] %s: %s", message.topic, message.text)
            return

        logger.info("[chat] %s: %s", message.topic, message.text)

        await self.session.process_text(message.text)

    async def on_participant_joined(self, participant: Participant) -> None:
        logger.info("joined: %s (%s)",
                    participant.name or "anonymous", participant.id)
        if participant.name:
            await self.session.say(f"Welcome, {participant.name}.")

    async def on_participant_left(self, participant: Participant) -> None:
        logger.info("left: %s", participant.name or participant.id)

    async def on_exit(self) -> None:
        logger.info("session finished")


async def chat_loop(session) -> None:
    """Type at the terminal and the agent answers as though you had spoken.

    The SDK's example runs this against ``pipeline.process_text``; the only
    difference here is that the pipeline is a process away.
    """
    loop = asyncio.get_running_loop()
    while True:
        text = await loop.run_in_executor(None, input, "you: ")
        if text.strip().lower() in {"quit", "exit"}:
            await session.end(reason="user quit")
            return
        if text.strip():
            await session.process_text(text)


def on_ready() -> None:
    zrt.invoke(
        os.getenv("AGENT_ID", "chat-agent"),
        room=Room(name="Chat Agent", playground=True, subscribe=[TOPIC]),
    )


if __name__ == "__main__":
    zrt.serve(ChatAgent, on_ready=on_ready)
