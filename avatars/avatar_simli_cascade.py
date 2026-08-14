# Giving a cascade agent a face. The avatar plugin renders a talking head from
# the TTS output and publishes it into the room as the agent's video. Nested
# vendor config objects cross as plain dicts and are rebuilt in the runtime.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, SimliAvatar

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "simli-avatar-agent")
FACE_ID = os.getenv("SIMLI_FACE_ID", "your-simli-face-id")


class AvatarAgent(Agent):
    def __init__(self) -> None:
        if FACE_ID == "your-simli-face-id":
            logger.warning("set SIMLI_FACE_ID -- the placeholder will not render")

        super().__init__(
            instructions=(
                "You are a friendly assistant with a face. Keep replies short and "
                "conversational -- long monologues look wrong on a talking head."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(model="nova-2"),
                llm=GoogleLLM(model="gemini-2.5-flash"),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
                avatar=SimliAvatar(
                    config={
                        "faceId": FACE_ID,
                        "handleSilence": True,
                        "maxSessionLength": 3600,
                        "maxIdleTime": 300,
                    },
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! You can see me as well as hear me now.")

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Simli Avatar", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(AvatarAgent, on_ready=on_ready)
