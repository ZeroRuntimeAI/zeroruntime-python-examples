# A realtime model that thinks, with your own TTS on the output -- a brand
# voice, a cloned voice, or a language its built-in voices do not cover. Its
# ears are untouched, so this is still speech-to-speech on the way in.

import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.plugins import CartesiaTTS, GeminiRealtime

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "hybrid-tts-agent")


class HybridVoiceAgent(Agent):
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
                    config={"voice": "Leda", "response_modalities": ["AUDIO"]},
                ),
                tts=CartesiaTTS(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Hybrid TTS", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(HybridVoiceAgent, on_ready=on_ready)
