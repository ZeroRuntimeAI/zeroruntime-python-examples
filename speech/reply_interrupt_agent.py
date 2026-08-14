# Driving the agent from outside the conversation: a room message makes it speak
# or cuts it off mid-sentence. say, reply and process_text are three different
# things; interrupt(force=True) also cuts uninterruptible utterances.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, RoomMessage
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import AnthropicLLM, DeepgramSTT, ElevenLabsTTS, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "reply-interrupt-agent")
TOPIC = "CHAT"


class ControllableAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=AnthropicLLM(),
                tts=ElevenLabsTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_message(self, message: RoomMessage) -> None:
        if message.backlog or message.topic != TOPIC:
            return

        if message.text == "reply":
            logger.info("replying")
            handle = await self.session.reply(
                "Create a random number between 1 and 100. Tell the user a joke "
                "using that number."
            )
            logger.info("utterance %s started", handle.utterance_id)

        elif message.text == "interrupt":
            logger.info("interrupting")
            await self.session.interrupt()

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(
        AGENT_ID,
        room=Room(name="Reply / Interrupt",
                  playground=True, subscribe=[TOPIC]),
    )
    logger.info("publish 'reply' or 'interrupt' on the %r topic", TOPIC)


if __name__ == "__main__":
    zeroruntime.serve(ControllableAgent, on_ready=on_ready)
