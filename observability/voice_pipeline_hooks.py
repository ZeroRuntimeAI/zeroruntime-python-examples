# Observing and rewriting a cascade pipeline at every stage.
#
# Three kinds of hook appear here, and they are not interchangeable:
#
#   stt          an async generator. Runs once per utterance, and what it yields
#                is what the turn sees. Yield nothing and the utterance is
#                dropped, which is how a noisy transcript gets filtered.
#   llm          an async generator rewrites chunks on their way to TTS; a plain
#                coroutine observes the finished response. Registering both is
#                fine -- they are separate hooks.
#   turn/state   plain coroutines. Nothing waits on them, so they cost the turn
#                nothing.
#
# The TTS stage is `pronunciations`, not a hook. See build_pipeline.

import logging
import re

import zeroruntime
from zeroruntime import Agent, Pipeline,Room, run_stt
from zeroruntime.inference import TurnDetector,SarvamAITTS
from zeroruntime.plugins import DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = "voice-pipeline-hooks-agent"

FILLERS = re.compile(r"\b(?:uh|um|like)\b")

SYNONYMS = {
    "working hours": "office hours",
    "timing": "office hours",
}

# One alternation rather than a chain of str.replace calls: replacing "# "
# before "## " turns a second-level heading into a stray "#" instead of
# removing it.
MARKDOWN = re.compile(r"\*\*|__|#{1,6} |[-*] ")


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=SarvamAITTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    # @pipeline.on("stt")
    # async def stt_hook(audio_stream):
    #     """Normalise the transcript before the LLM sees it.

    #     ``run_stt`` yields this utterance's event; rewriting ``event.data.text``
    #     rewrites what the turn receives. The audio phase is a passthrough -- STT
    #     runs in the agent process and only its transcript crosses, so iterating
    #     ``audio_stream`` yields nothing and transforming audio is not possible
    #     from here.
    #     """
    #     async for event in run_stt(audio_stream):
    #         text = FILLERS.sub("", event.data.text.lower())
    #         for src, dst in SYNONYMS.items():
    #             text = re.sub(rf"\b{src}\b", dst, text)
    #         text = " ".join(text.split())

    #         if not text:
    #             logging.info("[STT] dropped (nothing left after filtering)")
    #             continue

    #         event.data.text = text
    #         logging.info(f"[STT] {text} (final={event.data.final})")
    #         yield event

    # @pipeline.on("llm")
    # async def llm_text_filter(text_stream):
    #     """Rewrite the response as it streams, before TTS speaks it.

    #     Strips markdown so the voice does not read asterisks and hashes aloud.
    #     A generator, so one chunk in may yield none (buffer), one (rewrite) or
    #     several (split); code after the loop still runs, which is where a
    #     buffering hook would flush its tail.
    #     """
    #     async for chunk in text_stream:
    #         yield MARKDOWN.sub("", chunk)

    # @pipeline.on("llm")
    # async def on_llm(data: dict) -> None:
    #     """Observe the finished response -- logging, analytics, memory.

    #     Its return value is discarded, so this cannot change what is spoken.
    #     Use the generator above for that.
    #     """
    #     text = data.get("text", "")
    #     logging.info(f"[LLM] generated {text[:100]}...")

    # @pipeline.on("user_turn_start")
    # async def on_user_turn_start(transcript: str) -> None:
    #     logging.info(f"[USER TURN START] {transcript}")

    # @pipeline.on("user_turn_end")
    # async def on_user_turn_end() -> None:
    #     logging.info("[USER TURN END]")

    # @pipeline.on("agent_turn_start")
    # async def on_agent_turn_start() -> None:
    #     logging.info("[AGENT TURN START]")

    # @pipeline.on("agent_turn_end")
    # async def on_agent_turn_end() -> None:
    #     logging.info("[AGENT TURN END]")

    # @pipeline.on("turn_state")
    # async def on_turn_state(data: dict) -> None:
    #     logging.info(f"[TURN STATE] {data}")

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
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Voice Pipeline Hooks", playground=True,room_id="8ci6-jzbc-e049"))


if __name__ == "__main__":
    zeroruntime.serve(VoicePipelineHooks, on_ready=invoke_agent)
