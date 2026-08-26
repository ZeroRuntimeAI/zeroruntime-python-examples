# Showing a speech-to-speech model what the camera sees -- the realtime
# counterpart to vision_cascade.py, with identical mechanics. Room(vision=True)
# subscribes the agent to the track; only the frame count crosses the wire.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, PubSubSubscribeConfig, Room
from zeroruntime.plugins import GeminiLiveConfig, GeminiRealtime
from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "vision-realtime-agent")
TOPIC = "vision"


async def on_pubsub_message(frame: dict, backlog: bool, session) -> None:
    """One frame on TOPIC, handed to whichever agent is running."""
    await session.agent.on_chat(frame, backlog)


room = Room(name="Vision Realtime", playground=True, vision=True)
room.subscribe_to_pubsub(PubSubSubscribeConfig(topic=TOPIC, cb=on_pubsub_message))


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
                    config=GeminiLiveConfig(
                        voice="Leda", response_modalities=["AUDIO"]
                    ),
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! Show me something and I'll tell you what I see.")

    async def on_chat(self, frame: dict, backlog: bool) -> None:
        if backlog or str(frame.get("message") or "") != "capture_frames":
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
    zeroruntime.invoke(
        AGENT_ID,
        room=room,
    )
    logger.info(
        "publish 'capture_frames' on the %r topic to trigger a look", TOPIC)


if __name__ == "__main__":
    zeroruntime.serve(VisionRealtimeAgent, on_ready=on_ready)
