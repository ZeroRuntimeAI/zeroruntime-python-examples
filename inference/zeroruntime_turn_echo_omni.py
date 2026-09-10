# The echo-omni turn detector: the only difference from
# inference/zeroruntime_cascade.py is the `model` on TurnDetector.
#
# echo-large and echo-small read the transcript alone. echo-omni also listens to
# the audio, so it hears prosody -- it can tell a finished sentence from one that
# merely sounds finished, and a "mm-hmm" from a real turn.
import zeroruntime
from zeroruntime import Agent, Pipeline, Room
from zeroruntime.inference import CartesiaTTS, DeepgramSTT, GoogleLLM, TurnDetector
from zeroruntime.plugins import SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)

AGENT_ID = "zeroruntime-turn-echo-omni-agent"


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
                # model="echo-large" (the default) or "echo-small" for the
                # text-only detectors. `language` is echo-omni's alone -- passing
                # it to the other two is a TypeError, not a silently ignored
                # keyword. So are `sample_rate` and `channels`, which override
                # what the detector otherwise takes from the pipeline's audio.
                turn_detector=TurnDetector(model="echo-omni", language="en"),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def invoke_agent() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(name="ZeroRuntime Turn Echo Omni", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(VoiceAgent, on_ready=invoke_agent)
