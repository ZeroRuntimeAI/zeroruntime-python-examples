# What the agent publishes into the room rather than back to this process the
# way the hooks do: per-turn metrics, each stage's metrics as it finishes, and
# interim transcripts. send_interim is a gate -- the provider still has to emit.

import logging
import os

import zeroruntime
from zeroruntime import Agent, AgentEventsOptions, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logging.basicConfig(level=logging.INFO)

AGENT_ID = os.getenv("AGENT_ID", "agent-events")


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions=(
                "You are a helpful voice assistant. Keep answers to a sentence "
                "or two so the transcript stays easy to follow."
            ),
            pipeline=Pipeline(
                stt=DeepgramSTT(forward_interim_transcripts=True),
                llm=GoogleLLM(),
                tts=CartesiaTTS(word_timestamps=True),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, ask me anything.")


def invoke_agent() -> None:
    zeroruntime.invoke(
        AGENT_ID,
        room=Room(
            name="Agent Events",
            playground=True,
            # Reaches only participants joined with enableAgentEvents.
            agent_events=AgentEventsOptions(
                # Left at its default (on). Set False for a room that only wants
                # the live stream and collects turn metrics elsewhere.
                turn_metrics=True,
                # Additive, not a replacement: the per-turn payload still
                # arrives. Off by default because it is a message per stage.
                component_metrics=True,
                send_interim=True,
            ),
        ),
    )


if __name__ == "__main__":
    zeroruntime.serve(VoiceAgent, on_ready=invoke_agent)
