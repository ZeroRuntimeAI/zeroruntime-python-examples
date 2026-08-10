# Showing a speech-to-speech model what the camera sees -- the realtime
# counterpart to vision_cascade.py, with identical mechanics. Room(vision=True)
# subscribes the agent to the track; only the frame count crosses the wire.

import os

import zrt
from zrt import Agent, Pipeline, Room, RoomMessage
from zrt.plugins import GeminiRealtime
from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "vision-realtime-agent")
TOPIC = "vision"


class VisionRealtimeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can see. Describe what "
                "you are shown briefly and naturally."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                realtime=GeminiRealtime(
                    model="gemini-3.1-flash-live-preview",
                    config={"voice": "Leda", "response_modalities": ["AUDIO"]},
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! Show me something and I'll tell you what I see.")

    async def on_message(self, message: RoomMessage) -> None:
        if message.backlog or message.topic != TOPIC:
            return
        if message.text != "capture_frames":
            return

        logger.info("capturing frames")
        await self.session.reply(
            "Please analyze this frame and describe what you see in detail, "
            "within one line.",
            frames=2,
        )

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zrt.invoke(
        AGENT_ID,
        room=Room(
            name="Vision Realtime",
            playground=True,
            vision=True,
            subscribe=[TOPIC],
        ),
    )
    logger.info(
        "publish 'capture_frames' on the %r topic to trigger a look", TOPIC)


if __name__ == "__main__":
    zrt.serve(VisionRealtimeAgent, on_ready=on_ready)
