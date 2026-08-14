# One loan-advisor agent in English, Hindi, Gujarati or Marathi. A language is
# picked at startup and the caller can change it mid-call: three things move per
# language -- STT, TTS and instructions -- and all three have to agree, which is
# what change_component does in one call. Sarvam speaks all four; Deepgram and
# Cartesia have no Gujarati or Marathi to offer.

import logging
import os
import sys

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import GoogleLLM, SarvamAISTT, SarvamAITTS, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)



AGENT_ID = os.getenv("AGENT_ID", "multilang-loan-advisor")

_BASE_PROMPT = (
    "You are a business loan advisor. Help the caller understand loan products, "
    "check eligibility, and work out an EMI. Use the tools rather than "
    "estimating. Keep answers short -- they are spoken aloud. Amounts are in "
    "{currency}. Reply only in {label}. If the caller asks for another language, "
    "or answers you in one, call switch_language and carry on in that language."
)

# Sarvam takes one language code for both ends of the call, so a language is a
# code and the words that go with it.
LANGUAGES: dict[str, dict] = {
    "en": {
        "label": "English",
        "code": "en-IN",
        "currency": "₹ (rupees)",
        "greeting": (
            "Hi! I'm your business loan advisor. Want to hear about our loan "
            "products, check eligibility, or work out an EMI?"
        ),
    },
    "hi": {
        "label": "Hindi",
        "code": "hi-IN",
        "currency": "₹ (rupees)",
        "greeting": (
            "नमस्ते! मैं आपकी business loan advisor "
            "हूँ। आप loan products, eligibility, या EMI के "
            "बारे में पूछ सकते हैं।"
        ),
    },
    "gu": {
        "label": "Gujarati",
        "code": "gu-IN",
        "currency": "₹ (rupees)",
        "greeting": (
            "નમસ્તે! હું તમારી business loan "
            "advisor છું."
        ),
    },
    "mr": {
        "label": "Marathi",
        "code": "mr-IN",
        "currency": "₹ (rupees)",
        "greeting": (
            "नमस्कार! मी तुमची business loan "
            "advisor आहे."
        ),
    },
}

LANG = (sys.argv[1] if len(sys.argv) >
        1 else os.getenv("LANG_CODE", "hi")).lower()
if LANG not in LANGUAGES:
    raise SystemExit(
        f"unknown language {LANG!r}; pick one of {sorted(LANGUAGES)}")
CFG = LANGUAGES[LANG]


def _instructions(cfg: dict) -> str:
    return _BASE_PROMPT.format(currency=cfg["currency"], label=cfg["label"])


@function_tool
async def get_loan_products(loan_type: str) -> dict:
    """List the loan products of a given type.

    Args:
        loan_type: "term", "working_capital" or "equipment".
    """
    products = {
        "term": {"min": 500_000, "max": 50_000_000, "rate": 14.5, "tenure_months": 60},
        "working_capital": {"min": 200_000, "max": 20_000_000, "rate": 16.0, "tenure_months": 24},
        "equipment": {"min": 300_000, "max": 30_000_000, "rate": 13.0, "tenure_months": 84},
    }
    return products.get(loan_type) or {"error": f"no product called {loan_type}"}


@function_tool
async def calculate_emi(
    principal: float, annual_rate_percent: float, tenure_months: int
) -> dict:
    """Work out the monthly instalment for a loan.

    Args:
        principal: The amount borrowed.
        annual_rate_percent: The annual interest rate, as a percentage.
        tenure_months: How many months the loan runs for.
    """
    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)
    return {"emi": round(emi, 2), "total_paid": round(emi * tenure_months, 2)}


@function_tool
async def check_eligibility(
    cibil_score: int, business_age_years: float, monthly_turnover: float
) -> dict:
    """Check whether the caller qualifies.

    Args:
        cibil_score: Their credit score.
        business_age_years: How long the business has traded.
        monthly_turnover: Average monthly turnover.
    """
    reasons = []
    if cibil_score < 700:
        reasons.append("credit score below 700")
    if business_age_years < 2:
        reasons.append("business younger than two years")
    if monthly_turnover < 100_000:
        reasons.append("monthly turnover below the minimum")
    return {"eligible": not reasons, "reasons": reasons}


class MultilangLoanAgent(Agent):
    def __init__(self) -> None:
        self.lang = LANG
        super().__init__(
            instructions=_instructions(CFG),
            agent_id=AGENT_ID,
            tools=[get_loan_products, calculate_emi, check_eligibility],
            pipeline=Pipeline(
                stt=SarvamAISTT(model="saaras:v3", language=CFG["code"]),
                llm=GoogleLLM(model="gemini-2.5-flash"),
                tts=SarvamAITTS(model="bulbul:v3", language=CFG["code"]),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    @function_tool
    async def switch_language(self, language: str) -> dict:
        """Continue the call in another language.

        Args:
            language: "en", "hi", "gu" or "mr".
        """
        cfg = LANGUAGES.get(language)
        if cfg is None:
            return {
                "error": f"no language called {language}",
                "available": sorted(LANGUAGES),
            }
        if language == self.lang:
            return {"already_speaking": cfg["label"]}

        # The LLM, VAD and turn detector are the same in every language, so they
        # are not named here and the swap leaves them running.
        await self.session.change_component(
            stt=SarvamAISTT(model="saaras:v3", language=cfg["code"]),
            tts=SarvamAITTS(model="bulbul:v3", language=cfg["code"]),
            instructions=_instructions(cfg),
        )
        logger.info("language switched %s -> %s",
                    LANGUAGES[self.lang]["label"], cfg["label"])
        self.lang = language
        return {"switched_to": cfg["label"]}

    async def on_enter(self) -> None:
        await self.session.say(CFG["greeting"])

    async def on_exit(self) -> None:
        logger.info("call finished in %s", LANGUAGES[self.lang]["label"])


def on_ready() -> None:
    zeroruntime.invoke(
        AGENT_ID, room=Room(
            name=f"Loan Advisor ({CFG['label']})", playground=True)
    )


if __name__ == "__main__":
    logger.info("running in %s", CFG["label"])
    zeroruntime.serve(MultilangLoanAgent, on_ready=on_ready)
