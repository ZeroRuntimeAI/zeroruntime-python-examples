# An Anam avatar on a cascade pipeline -- avatar_simli_cascade.py with a
# different vendor in the same slot. An avatar is orthogonal to the pipeline's
# mode, so the swap touches nothing but the avatar line. The Anam key is read in
# the runtime and never sent from here.

import logging
import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import AnamAvatar, CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "anam-avatar-agent")
AVATAR_ID = os.getenv("ANAM_AVATAR_ID", "your-anam-avatar-id")


@function_tool
async def get_weather(latitude: str, longitude: str) -> dict:
    """Called when the user asks about the weather.

    Estimate the latitude and longitude of the location yourself rather than
    asking for them.

    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
    """
    import aiohttp

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    )
    async with aiohttp.ClientSession() as http:
        async with http.get(url) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"weather lookup failed: HTTP {response.status}")
            data = await response.json()

    return {
        "temperature": data["current"]["temperature_2m"],
        "temperature_unit": "Celsius",
    }


class AvatarVoiceAgent(Agent):
    def __init__(self) -> None:
        if AVATAR_ID == "your-anam-avatar-id":
            logger.warning(
                "set ANAM_AVATAR_ID -- the placeholder will not render")

        super().__init__(
            instructions=(
                "You are a helpful virtual assistant with a visual avatar that can "
                "answer questions about weather and help with other tasks. Keep "
                "replies short and conversational -- long monologues look wrong on "
                "a talking head."
            ),
            agent_id=AGENT_ID,
            tools=[get_weather],
            pipeline=Pipeline(
                stt=DeepgramSTT(model="nova-2"),
                llm=GoogleLLM(model="gemini-2.5-flash"),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
                avatar=AnamAvatar(avatar_id=AVATAR_ID),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello! I'm your AI avatar assistant. How can I help you today?"
        )

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Anam Avatar Agent", playground=True))


if __name__ == "__main__":
    zrt.serve(AvatarVoiceAgent, on_ready=on_ready)
