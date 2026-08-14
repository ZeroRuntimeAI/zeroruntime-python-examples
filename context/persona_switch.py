# Five personas swapped live from the room's chat. A pubsub message naming one
# rebuilds the running pipeline -- different STT, LLM, TTS and turn detector, or
# a hop to a realtime model and back -- keeping everything already said.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, RoomMessage
from zeroruntime.core.tuning import EOUConfig, InterruptConfig
from zeroruntime.inference import (
    AssemblyAISTT,
    CartesiaTTS,
    DeepgramSTT,
    DeepgramTTS,
    GeminiRealtime,
    GoogleLLM,
    GoogleSTT,
    GoogleTTS,
    SarvamAISTT,
    SarvamAITTS,
    TurnDetector,
)
from zeroruntime.plugins import SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


TOPIC = "CHAT"
AGENT_ID = os.getenv("AGENT_ID", "persona-switch")

_VOICE = (
    "You are a general-purpose voice AI assistant powered by ZeroRuntime. "
    "You can answer any question, help with tasks, provide information, have "
    "casual conversations, and assist with anything the user needs. You are "
    "friendly, knowledgeable, concise, and natural in conversation. Keep "
    "responses short and conversational -- you are a voice agent, not a "
    "chatbot. Avoid long lists or overly structured answers. Speak naturally."
)

_TUNING = dict(
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
    eou=EOUConfig(mode="ADAPTIVE", min_max_speech_wait_timeout=[0.1, 0.5]),
    interrupt=InterruptConfig(
        mode="HYBRID",
        interrupt_min_duration=0.2,
        interrupt_min_words=2,
        false_interrupt_pause_duration=2.0,
        resume_on_false_interrupt=True,
    ),
)


def _persona(name: str, **components) -> dict:
    return {
        "name": name,
        "instructions": f"Your name is {name}. {_VOICE}",
        "pipeline": Pipeline(**components),
    }


PERSONAS = {
    "deepgram": _persona(
        "Alex",
        stt=DeepgramSTT(model="nova-2"),
        llm=GoogleLLM(model="gemini-3-flash-preview"),
        tts=CartesiaTTS(model="sonic-3"),
        **_TUNING,
    ),
    "assembly": _persona(
        "Maya",
        stt=AssemblyAISTT(),
        llm=GoogleLLM(model="gemini-3-flash-preview"),
        tts=DeepgramTTS(model="aura-2"),
        **_TUNING,
    ),
    "google": _persona(
        "Sophia",
        stt=GoogleSTT(model="chirp_3"),
        llm=GoogleLLM(model="gemini-3-flash-preview"),
        tts=GoogleTTS(),
        **_TUNING,
    ),
    "sarvam": _persona(
        "Emma",
        stt=SarvamAISTT(model="saaras:v3", language="en-IN"),
        llm=GoogleLLM(model="gemini-3-flash-preview"),
        tts=SarvamAITTS(model="bulbul:v3", speaker="suhani", language="en-IN"),
        **_TUNING,
    ),
    "realtime": _persona(
        "Ryan",
        realtime=GeminiRealtime(
            model="gemini-3.1-flash-live-preview",        
        ),
    ),
}

FIRST = "deepgram"


class PersonaAgent(Agent):
    """One long-lived agent that wears different personas.

    A switch rebuilds the pipeline on the *same* session, so the conversation is
    preserved across every hop -- cascade to cascade, cascade to realtime, and
    back. That is the whole point: a handoff would start a new agent and lose it.
    """

    def __init__(self) -> None:
        start = PERSONAS[FIRST]
        super().__init__(
            instructions=start["instructions"],
            agent_id=AGENT_ID,
            pipeline=start["pipeline"],
        )
        self._current = FIRST

    async def on_enter(self) -> None:
        await self.session.say(
            f"Hey! {PERSONAS[self._current]['name']} here -- what can I help with?"
        )

    async def on_message(self, message: RoomMessage) -> None:
        """A persona name in the room chat switches the pipeline."""
        if message.backlog:
            return
        key = message.text.strip().lower()
        if key not in PERSONAS:
            logger.info("ignoring unknown persona: %r", key)
            return
        await self.switch_persona(key)

    async def switch_persona(self, key: str) -> None:
        if key == self._current:
            return
        outgoing = PERSONAS[self._current]["name"]
        incoming = PERSONAS[key]["name"]
        logger.info("switching persona: %s -> %s", outgoing, incoming)

        try:
            mode = await self.session.change_pipeline(
                PERSONAS[key]["pipeline"],
                instructions=PERSONAS[key]["instructions"],
            )
        except Exception:
            logger.exception(
                "persona switch to %s failed; staying on %s", key, outgoing)
            await self.session.say(
                f"Sorry, I could not switch to {incoming}. Still {outgoing} here."
            )
            return

        self._current = key
        logger.info("now running %s (mode=%s)", incoming, mode)
        await self.session.say(
            f"{incoming} here, taking it from {outgoing}. What's next?"
        )

    async def on_exit(self) -> None:
        logger.info("session finished on persona %s",
                    PERSONAS[self._current]["name"])


def on_ready() -> None:
    zeroruntime.invoke(
        AGENT_ID, room=Room(name="Persona Switch",
                            playground=True, subscribe=[TOPIC])
    )


if __name__ == "__main__":
    zeroruntime.serve(PersonaAgent, on_ready=on_ready)
