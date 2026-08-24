# realtime_with_vad.py with two things in front of the model: VAD and denoise.
# A realtime model hears the caller directly, so anything that shapes that
# audio has to sit in the pipeline -- SileroVAD marks where speech starts and
# stops, which is what sharpens time to first byte (TTFB), and AICousticsDenoise
# cleans the inbound stream before either of them sees it.

import os
import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.inference import AICousticsDenoise
from zeroruntime.plugins import GeminiRealtime,SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "realtime-basi-with-vad")


class MyVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                llm=GeminiRealtime(
                    model="gemini-3.1-flash-live-preview",
                    config={
                        "voice": "Leda",
                        "response_modalities": ["AUDIO"],
                    },
                ),
                vad=SileroVAD(),
                denoise=AICousticsDenoise(model_id="quail-vf-2.2-l-16khz"),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Realtime Basic Vad", playground=True))

if __name__ == "__main__":
    zeroruntime.serve(MyVoiceAgent, on_ready=on_ready)
