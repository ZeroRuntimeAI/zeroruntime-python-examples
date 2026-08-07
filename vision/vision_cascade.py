# Showing the model what the camera sees: a room message triggers a reply that
# captures the newest N frames. Room(vision=True) subscribes the agent to the
# video track, and the pixels stay there -- only the count travels.

import os

import zrt
from zrt import Agent, Pipeline, Room, RoomMessage
from zrt.inference import TurnDetector
from zrt.plugins import DeepgramSTT, CartesiaTTS, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "vision-agent")
TOPIC = "CHAT"


class VisionAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "YOU CAN ONLY SPEAK IN ENGLISH. You are a helpful voice assistant "
                "that can answer questions and help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_message(self, message: RoomMessage) -> None:
        if message.backlog:
            return
        if message.topic != TOPIC or message.text != "capture_frames":
            return

        logger.info("capturing frames on %r", message.topic)
        await self.session.reply(
            "Please analyze this frame and describe what you see in details, "
            "within one line.",
            frames=2,
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    result = zrt.invoke(
        AGENT_ID,
        room=Room(
            name="Vision Agent",
            playground=True,
            vision=True,
            subscribe=[TOPIC],
        ),
    )
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])
        logger.info(
            "publish 'capture_frames' on the %r topic to trigger a look", TOPIC)


if __name__ == "__main__":
    zrt.serve(VisionAgent, on_ready=on_ready)
