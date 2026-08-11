# The smallest complete cascading agent: STT, LLM, TTS, VAD and turn detector.


import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "cascade-basic-agent"


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant that can answer questions "
                "and help with tasks."
            ),
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

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_weather(self, latitude: str, longitude: str) -> dict:
        """Fetch the current weather for a location.

        Args:
            latitude: Latitude of the location. Estimate it; do not ask.
            longitude: Longitude of the location. Estimate it; do not ask.
        """
        return {"latitude": latitude, "longitude": longitude, "temperature_c": 28}


def invoke_agent() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Cascade Agent", playground=True))


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=invoke_agent)
