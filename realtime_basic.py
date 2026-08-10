# Speech to speech with one model doing all of it -- the counterpart to
# cascade_basic.py. Filling the realtime slot is what makes it a realtime
# pipeline; the mode is inferred from the components, never declared.

import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.plugins import GeminiRealtime

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "realtime-basic")


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
                    model="gemini-3.1-flash-live-preview",
                    config={
                        "voice": "Leda",
                        "response_modalities": ["AUDIO"],
                    },
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Gemini Realtime Agent", playground=True))


if __name__ == "__main__":
    zrt.serve(MyVoiceAgent, on_ready=on_ready)
