# Subscribing to what the agent process sees and this one otherwise cannot:
# component errors, recording state changes, and per-component latency metrics
# as they are measured. You receive only the events you register for.


import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "pipeline-events")

pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-2"),
    llm=GoogleLLM(model="gemini-2.5-flash"),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)


@pipeline.on("error")
def on_error(data: dict) -> None:
    """A component failed.

    ``source`` is one of STT, LLM, TTS, VAD, TURN-D, REALTIME or room. This is
    also where a *fallback* announces itself: the fallback wrappers emit an error
    as they switch, so a provider quietly failing over shows up here rather than
    only in the metrics after the call.

    Worth wiring on any real deployment. A 404 from a TTS voice looks exactly
    like a silent agent from the caller's side, and this is the difference
    between knowing that within a second and reading it in the logs tomorrow.
    """
    logger.error("[%s] %s", data.get("source", "?"), data.get("error", ""))


@pipeline.on("recording_started")
def on_recording_started(data: dict) -> None:
    logger.info("recording started (%s)", data.get(
        "kind") or data.get("type") or "")


@pipeline.on("recording_stopped")
def on_recording_stopped(data: dict) -> None:
    logger.info("recording stopped")


@pipeline.on("recording_failed")
def on_recording_failed(data: dict) -> None:
    """Only after the SDK has retried and given up.

    Not a blip: this fires once retries over a few seconds are exhausted, so it
    means the call genuinely is not being recorded. If that matters for
    compliance, this is the hook that has to page somebody.
    """
    logger.error("recording FAILED: %s", data.get("error", ""))


@pipeline.on("metrics.stt")
def on_stt_metrics(data: dict) -> None:
    logger.info("stt metric collector  %s", data)
    logger.info("stt %s", data)


@pipeline.on("metrics.llm")
def on_llm_metrics(data: dict) -> None:
    """Time-to-first-token, per turn, while the call is still running.

    ``session.get_metrics()`` returns the same numbers, but only once you ask --
    and if the session ends before you do, they are gone. Subscribe when you want
    to react to slowness rather than report on it.
    """
    logger.info("llm metric collector %s", data)
    logger.info("llm %s", data)


class WatchedAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant. Keep answers short.",
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I am being watched closely today.")

    async def on_exit(self) -> None:
        logger.info("session finished")


def on_ready() -> None:
    result = zrt.invoke(
        AGENT_ID,
        room=Room(name="Pipeline Events", playground=True, recording=True),
    )
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    zrt.serve(WatchedAgent, on_ready=on_ready, recording=True)
