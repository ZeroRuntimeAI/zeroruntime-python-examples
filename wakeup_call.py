# Nudging a caller who has gone quiet: a wake_up timer on the agent, with the
# callback as a method so the handler travels with the agent that owns it.

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import AnthropicLLM, DeepgramSTT, GoogleTTS, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "wakeup-call-agent"


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant that can answer questions "
                "and help with tasks and help with horoscopes and weather."
            ),
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=AnthropicLLM(),
                tts=GoogleTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
            wake_up=15,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    async def on_wake_up(self) -> None:
        await self.session.say("Hello, are you there?")


def invoke_agent() -> None:
    print(zrt.invoke(AGENT_ID, room=Room(
        name="Sandbox Agent", playground=True))["playground_url"])


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=invoke_agent)
