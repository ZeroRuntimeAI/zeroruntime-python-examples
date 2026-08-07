# Tuning the handoff between caller and agent on a cascade pipeline: EOU config
# decides when the caller has finished speaking, interruption config decides
# when a barge-in stops the agent. Also shows an uninterruptible utterance.

import os

import zrt
from zrt import Agent, EOUConfig, InterruptConfig, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "cascade-advanced")


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
                eou=EOUConfig(
                    mode="ADAPTIVE",
                    min_max_speech_wait_timeout=[0.5, 0.8],
                ),
                interrupt=InterruptConfig(
                    mode="HYBRID",
                    interrupt_min_duration=0.2,
                    interrupt_min_words=2,
                    false_interrupt_pause_duration=2.0,
                    resume_on_false_interrupt=True,
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "This example script showcases advanced cascade features, including "
            "interruptible speech. This message cannot be interrupted.",
            interruptible=False,
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    result = zrt.invoke(AGENT_ID, room=Room(
        name="Cascade Advanced", playground=True))
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=on_ready)
