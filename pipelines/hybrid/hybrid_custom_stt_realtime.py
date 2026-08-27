# Your own transcriber on the input, a realtime model behind it. Worth doing
# when the model transcribes your callers' language badly. The vad is required
# once transcription is external -- something has to close the turn.

import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.plugins import GeminiLiveConfig, GeminiRealtime, SarvamAISTT, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "hybrid-stt-agent")


class AdditionalSTTAndRealtime(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                realtime=GeminiRealtime(
                    model="gemini-3.1-flash-live-preview",
                    config=GeminiLiveConfig(
                        voice="Leda", response_modalities=["AUDIO"]
                    ),
                ),
                stt=SarvamAISTT(),
                vad=SileroVAD(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Hybrid STT", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(AdditionalSTTAndRealtime, on_ready=on_ready)
