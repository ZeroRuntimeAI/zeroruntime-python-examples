# Three separate things: OpenTelemetry traces/metrics/logs configured on the
# Room, platform recording, and the conversation history fetched in on_exit.
# Traces and metrics default on; logs default off, because they are the noisy one.

import logging
import os

import zrt
from zrt import Agent, Exporter, Observability, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "observability-agent")
OTLP_URL = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)


@pipeline.on("metrics.stt")
def on_stt_metrics(data: dict) -> None:
    logger.info("stt %s", data)


@pipeline.on("metrics.llm")
def on_llm_metrics(data: dict) -> None:
    logger.info("llm %s", data)


@pipeline.on("metrics.tts")
def on_tts_metrics(data: dict) -> None:
    logger.info("tts %s", data)


@pipeline.on("error")
def on_error(data: dict) -> None:
    """Component failures. Worth watching even when everything else is quiet --
    a TTS that has stopped answering looks, from the room, like an agent that
    has decided not to speak."""
    logger.error("component error: %s", data)


class MyVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        history = await self.session.get_context_history()

        logger.info(
            "=== SESSION END: CONTEXT HISTORY (%d items) ===", len(history))
        for message in history:
            role = str(message.get("role", "unknown")).upper()
            if role == "SYSTEM":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part if isinstance(part, str) else "[Image/Other]" for part in content
                )
            logger.info("%s: %s", role, content)
        logger.info("===============================================")

        await self.session.say("Goodbye!")


def on_ready() -> None:
    zrt.invoke(
        AGENT_ID,
        room=Room(
            name="Observability Hooks",
            playground=True,
            recording=True,
            observability=Observability(
                traces=Exporter(enabled=True, export_url=OTLP_URL or None),
                metrics=Exporter(enabled=True, export_url=OTLP_URL or None),
                logs=Exporter(enabled=bool(OTLP_URL),
                              export_url=OTLP_URL or None),
                log_level="INFO",
            ),
        ),
    )


if __name__ == "__main__":
    zrt.serve(MyVoiceAgent, on_ready=on_ready)
