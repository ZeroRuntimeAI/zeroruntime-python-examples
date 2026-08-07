# Fixing how the agent pronounces things: a list of PronunciationRule
# substitutions applied in the agent process between LLM and TTS, so nothing
# crosses the wire per chunk and the turn pays nothing for them.

import os

import zrt
from zrt import Agent, Pipeline, PronunciationRule, Room
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)




AGENT_ID = os.getenv("AGENT_ID", "enhanced-pronounciation")

RULES = [
    PronunciationRule("nginx", "engine x"),
    PronunciationRule("URL", "U R L"),
    PronunciationRule("API", "A P I"),
    PronunciationRule("VideoSDK", "Video SDK"),
    PronunciationRule("HTTP", "H T T P"),
    PronunciationRule("HTTPS", "H T T P S"),
    PronunciationRule("JSON", "J SON"),
    PronunciationRule("SQL", "sequel"),
    PronunciationRule("AWS", "A W S"),
    PronunciationRule("CI/CD", "C I C D"),
]

pipeline = Pipeline(
    stt=DeepgramSTT(model="nova-2"),
    llm=GoogleLLM(model="gemini-2.5-flash"),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    pronunciations=RULES,
)


class DocsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a developer support agent. Answer questions about APIs, "
                "HTTP, JSON and SQL. Keep answers short and conversational."
            ),
            agent_id=AGENT_ID,
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi! Ask me anything about the API.")

    async def on_exit(self) -> None:
        logger.info("session finished")


def on_ready() -> None:
    result = zrt.invoke(AGENT_ID, room=Room(name="Pronunciation", playground=True))
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    zrt.serve(DocsAgent, on_ready=on_ready)
