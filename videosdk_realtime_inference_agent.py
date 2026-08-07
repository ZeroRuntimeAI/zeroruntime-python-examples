# A realtime model reached through the VideoSDK gateway, with no vendor key --
# the realtime counterpart to videosdk_cascade_inference_agent.py. The gateway
# class flattens the arguments the direct plugin nests; they are not the same class.

import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import GeminiRealtime
from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "videosdk-realtime-inference-agent")


class MyVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                realtime=GeminiRealtime(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                    voice="Puck",
                    language_code="en-US",
                    response_modalities=["AUDIO"],
                    temperature=0.7,
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    result = zrt.invoke(
        AGENT_ID, room=Room(
            name="VideoSDK Realtime Inference", playground=True)
    )
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    zrt.serve(MyVoiceAgent, on_ready=on_ready)
