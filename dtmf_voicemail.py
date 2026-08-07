# Keypad input and answering-machine detection, both declared on the agent with
# their callbacks as methods.

import zrt
from zrt import Agent, Pipeline, Room
from zrt.core.agent import VoiceMail
from zrt.inference import TurnDetector
from zrt.plugins import DeepgramSTT, ElevenLabsTTS, OpenAILLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "dtmf-voicemail-agent"


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions="You are a helpful voice assistant that can answer questions.",
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=OpenAILLM(),
                tts=ElevenLabsTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
            dtmf_enabled=True,
            voice_mail=VoiceMail(llm=OpenAILLM()),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    async def on_dtmf(self, key: str, payload: dict) -> None:
        """One keypress. Fire and forget -- nothing in the pipeline waits."""
        print("DTMF message received:", key, payload)

    async def on_voicemail(self) -> None:
        """Awaited, so anything said here finishes before the call ends."""
        print("Voice Mail detected, Shutting down the agent")
        await self.hangup(reason="reached voicemail")


def invoke_agent() -> None:
    print(zrt.invoke(AGENT_ID, room=Room(
        name="Sandbox Agent", playground=True))["playground_url"])


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=invoke_agent)
