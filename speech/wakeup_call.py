# Nudging a caller who has gone quiet: a wake_up timer on the agent, with the
# callback as a method so the handler travels with the agent that owns it.

import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import AnthropicLLM, DeepgramSTT, GoogleTTS, SileroVAD

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
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Wakeup Call", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(VoiceAgent, on_ready=invoke_agent)
