# Observing a cascade pipeline's turn lifecycle. The hook bodies run in this
# process and their return values are discarded, so they cost the turn nothing.
# Only the observation hooks are shown, not the stream transforms.

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "voice-pipeline-hooks-agent"


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str) -> None:
        logging.info(f"[USER TURN START] {transcript}")

    @pipeline.on("user_turn_end")
    async def on_user_turn_end() -> None:
        logging.info("[USER TURN END]")

    @pipeline.on("agent_turn_start")
    async def on_agent_turn_start() -> None:
        logging.info("[AGENT TURN START]")

    @pipeline.on("agent_turn_end")
    async def on_agent_turn_end() -> None:
        logging.info("[AGENT TURN END]")

    @pipeline.on("turn_state")
    async def on_turn_state(data: dict) -> None:
        logging.info(f"[TURN STATE] {data}")

    return pipeline


class VoicePipelineHooks(Agent):
    def __init__(self) -> None:
        super().__init__(
            agent_id=AGENT_ID,
            instructions="You are a helpful voice assistant.",
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def invoke_agent() -> None:
    print(zrt.invoke(AGENT_ID, room=Room(
        name="Sandbox Agent", playground=True))["playground_url"])


if __name__ == "__main__":
    zrt.serve(VoicePipelineHooks, on_ready=invoke_agent)
