# A cascade through the VideoSDK inference gateway: the only difference from
# cascade_basic.py is the import line, and the pipeline needs one credential
# rather than one per vendor. VAD stays local -- there is no gateway twin.
import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import CartesiaTTS, DeepgramSTT, GoogleLLM, TurnDetector
from zrt.plugins import SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "videosdk-cascade-inference-agent"


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant that can answer questions "
                "and help with tasks."
            ),
            pipeline=Pipeline(
                stt=DeepgramSTT(model="nova-2"),
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


def invoke_agent() -> None:
    print(zrt.invoke(AGENT_ID, room=Room(name="Sandbox Agent", playground=True))["playground_url"])


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=invoke_agent)
