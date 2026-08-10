# An outbound lead-qualification agent: an stt hook that cleans transcripts
# before the LLM sees them, an llm hook watching for goodbye intent, wake_up
# nudges with a counter, and a duration cap with a spoken farewell.
import asyncio
import re

import zrt
from zrt import Agent, Pipeline, Room, function_tool, run_stt
from zrt.core.agent import BackgroundAudio
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = "lead-qualification-agent"

HANGUP_TIMER = 300

WAKE_UP_AFTER_S = 10
MAX_WAKE_UPS = 3

IGNORE_PATTERNS = [
    r"^\s*$",
    r"^(thank you|thanks)[.!]?$",
    r"^(bye|okay|ok)[.!]?$",
    r"subtitles? by",
    r"^\W+$",
]

NOISE_WORDS = {"recording": "", "recorded": ""}

GOODBYE_KEYWORDS = ("goodbye", "have a great day", "thanks for your time")


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2-conversationalai"),
        llm=GoogleLLM(model="gemini-3-flash-preview", thinking_budget=0),
        tts=CartesiaTTS(model="sonic-3.5"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    @pipeline.on("stt")
    async def stt_hook(audio_stream):
        """Filter and clean transcripts before the LLM ever sees them."""
        async def audio_phase():
            async for audio in audio_stream:
                yield audio

        async for event in run_stt(audio_phase()):
            original = (event.data.text or "").lower().strip()

            if any(re.search(p, original) for p in IGNORE_PATTERNS):
                logger.info(f"[STT] ignored: {original!r}")
                continue

            text = original
            for src, dst in NOISE_WORDS.items():
                text = re.sub(rf"\b{src}\b", dst, text)
            cleaned = " ".join(text.split())

            if not cleaned:
                logger.info(f"[STT] empty after cleanup: {original!r}")
                continue

            event.data.text = cleaned
            logger.info(f"[STT] {cleaned}")
            Lead.wake_ups = 0
            yield event

    @pipeline.on("llm")
    async def on_llm(data: dict) -> None:
        """Observation only -- nothing waits on this."""
        text = data.get("text", "") or ""
        logger.info(f"[LLM] {text[:100]}...")
        if any(kw in text.lower() for kw in GOODBYE_KEYWORDS):
            logger.info("[LLM] goodbye intent detected")

    return pipeline


class Lead(Agent):
    """Qualifies an outbound lead, then wraps the call up."""

    wake_ups = 0

    def __init__(self) -> None:
        Lead.wake_ups = 0
        super().__init__(
            name="Lead Qualification",
            agent_id=AGENT_ID,
            instructions=(
                "You are a friendly outbound representative qualifying a lead. "
                "Ask, one at a time: what they are building, their team size, "
                "and their timeline. Keep every reply to one or two sentences. "
                "When you have all three, thank them and call wrap_up."
            ),
            pipeline=build_pipeline(),
            background_audio=BackgroundAudio(
                file_path="./assets/audio/office-quiet-work-ambience.wav",
                volume=0.5,
                looping=True,
            ),
            wake_up=WAKE_UP_AFTER_S,
        )

    async def on_enter(self) -> None:
        opening = await self.session.say(
            "Hi! This is a quick call about the demo you requested "
            "-- is now a good time?",
            interruptible=False,
        )
        await opening
        asyncio.create_task(self._cap_call_duration())

    async def on_wake_up(self) -> None:
        """Runs after each silent stretch. The session is live, so it can speak."""
        Lead.wake_ups += 1
        logger.info(f"wake up attempt {Lead.wake_ups}")
        if Lead.wake_ups == 1:
            await self.session.say("Are you still there?")
        elif Lead.wake_ups < MAX_WAKE_UPS:
            await self.session.say("I can call back later if now is a bad time.")
        else:
            await self.hangup(
                reason="no answer after repeated prompts",
                farewell="I'll let you go for now. Have a great day!",
            )

    @function_tool
    async def wrap_up(self, what_they_build: str, team_size: str, timeline: str) -> dict:
        """Record the qualified lead and end the call.

        Args:
            what_they_build: What the lead said they are building.
            team_size: How large their team is.
            timeline: When they are looking to start.
        """
        lead = {
            "what_they_build": what_they_build,
            "team_size": team_size,
            "timeline": timeline,
        }
        logger.info(f"[LEAD] {lead}")
        await self.hangup(
            reason="lead qualified",
            farewell="That's everything I needed -- thanks for your time!",
        )
        return {"status": "qualified", **lead}

    async def _cap_call_duration(self) -> None:
        await asyncio.sleep(HANGUP_TIMER)
        logger.info("call duration cap reached")
        await self.hangup(reason="duration cap", farewell="Thanks for your time!")


def invoke_agent() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Lead Qualification", playground=True))


if __name__ == "__main__":
    zrt.serve(Lead, on_ready=invoke_agent)
