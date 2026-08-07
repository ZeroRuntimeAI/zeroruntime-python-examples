# Keeping the call up when a provider degrades: a pipeline slot takes a list,
# the head serves and the tail stands by. Every credential, standbys included,
# is checked when the session starts rather than at failover.


import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import (
    CartesiaTTS,
    GoogleLLM,
    DeepgramSTT,
    OpenAILLM,
    OpenAISTT,
    OpenAITTS,
    SileroVAD,
)

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "fallback-recovery-agent"


class ResilientAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant that can answer questions "
                "and help with tasks."
            ),
            pipeline=Pipeline(
                stt=[OpenAISTT(), DeepgramSTT()],
                llm=[OpenAILLM(model="gpt-4o-mini"), GoogleLLM(model="gemini-2.5-flash")],
                tts=[OpenAITTS(voice="alloy"), CartesiaTTS()],
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello Buddy, Welcome to Videosdk's Voice AI Agent Framework."
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def invoke_agent() -> None:
    print(zrt.invoke(AGENT_ID, room=Room(name="Sandbox Agent", playground=True))["playground_url"])


if __name__ == "__main__":
    zrt.serve(ResilientAgent,on_ready=invoke_agent)
