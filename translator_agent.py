# Answering in whatever language the caller switched to: each turn detects the
# language here and, when it changes, rebuilds the pipeline with a TTS that
# speaks it. change_pipeline takes a complete pipeline, not a patch.

import os

import zrt
from zrt import Agent, Pipeline, Room
from zrt.plugins import CartesiaTTS, OpenAILLM, SarvamAISTT, SarvamAITTS, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "translator-agent")

current_language = "en-IN"

session: "zrt.Session | None" = None

pipeline = Pipeline(
    stt=SarvamAISTT(),
    llm=OpenAILLM(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
)


async def detect_language(transcript: str) -> str:
    """Whatever language service you have. Runs in this process, not the agent's."""
    api_key = os.getenv("SARVAMAI_API_KEY")
    if not api_key:
        raise ValueError("SARVAMAI_API_KEY is not set")

    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    response = client.text.identify_language(input=transcript)
    return response.language_code


@pipeline.on("user_turn_start")
async def on_user_turn_start(transcript: str) -> None:
    """Runs the moment the caller stops talking, before the LLM generates."""
    global current_language
    if session is None:
        return

    # A turn can arrive with nothing in it: the VAD heard speech, the STT
    # produced no transcript, and the turn was force-finalized anyway. The
    # transcript field is a protobuf string, so "no transcript" reaches us as
    # "". Sarvam rejects empty input with a 400, so asking would just be a
    # guaranteed failure logged once per silent turn.
    if not transcript or not transcript.strip():
        return

    try:
        detected = await detect_language(transcript)
    except Exception as e:
        logger.warning(
            "language detection failed (%r); staying on %s", e, current_language
        )
        return

    if not detected or detected == current_language:
        return

    logger.info("language changed %s -> %s", current_language, detected)
    current_language = detected

    await session.change_pipeline(
        Pipeline(
            stt=SarvamAISTT(),
            llm=OpenAILLM(),
            tts=SarvamAITTS(language=current_language),
            vad=SileroVAD(),
        )
    )


class TranslatorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful translator assistant that can speak to the user "
                "in their language."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        global session
        session = self.session

    async def on_exit(self) -> None:
        logger.info("call finished in %s", current_language)


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Translator Agent", playground=True))


if __name__ == "__main__":
    zrt.serve(TranslatorAgent, on_ready=on_ready)
