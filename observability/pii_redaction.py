# Redaction travels on the Room, next to traces and metrics. `rules` names what
# counts as sensitive here and what each kind reads as once it is gone; `target`
# says where it is removed from.

import logging

import zeroruntime
from zeroruntime import (
    Agent,
    Observability,
    PIIRedaction,
    PIIType,
    Pipeline,
    RedactionTarget,
    Room,
)
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)

pipeline = Pipeline(
    stt=DeepgramSTT(),
    llm=GoogleLLM(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
    turn_detector=TurnDetector(),
)


class PaymentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You take card payments over the phone. Ask for the caller's "
                "date of birth, then a contact number, then an email for the "
                "receipt, then the card number. Ask for one at a time, and "
                "never repeat a value back -- acknowledge it and move on to the "
                "next question. Keep every question short."
            ),
            pipeline=pipeline,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, I can take your payment. May I start with your date of birth?")

    async def on_exit(self) -> None:
        await self.session.say("Thank you, goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(
        room=Room(
            name="PII Redaction",
            playground=True,
            recording=True,
            observability=Observability(
                pii_redaction=PIIRedaction(
                    target=[RedactionTarget.RECORDING, RedactionTarget.TRANSCRIPT],
                    rules={
                        PIIType.CREDIT_CARD: "CARD",
                        PIIType.PHONE_NUMBER: "NUMBER",
                        PIIType.EMAIL: "MAIL",
                        "Date Of Birth": "DOB",
                    },
                ),
            ),
        ),
    )


if __name__ == "__main__":
    zeroruntime.serve(PaymentAgent, on_ready=on_ready)
