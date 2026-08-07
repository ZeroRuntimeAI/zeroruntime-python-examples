# Giving a cascade agent a face. The avatar plugin renders a talking head from
# the TTS output and publishes it into the room as the agent's video. Nested
# vendor config objects cross as plain dicts and are rebuilt in the runtime.

import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, SimliAvatar

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "avatar-agent")
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
    result = zrt.invoke(AGENT_ID, room=Room(
        name="Avatar Agent", playground=True))
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    zrt.serve(AvatarAgent, on_ready=on_ready)
