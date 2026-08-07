# One loan-advisor agent in English, Hindi, Gujarati or Marathi, picked at
# startup. Three things change per language -- STT, TTS and instructions -- and
# all three have to agree. For mid-call switching see translator_agent.py.

import os
import sys

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "multilang-loan-advisor")

_BASE_PROMPT = (
    "You are a business loan advisor. Help the caller understand loan products, "
    "check eligibility, and work out an EMI. Use the tools rather than "
    "estimating. Keep answers short -- they are spoken aloud. Amounts are in "
    "{currency}. Reply only in {label}."
)

LANGUAGES: dict[str, dict] = {
    "en": {
        "label": "English",
        "stt_language": "en",
        "tts_language": "en",
        "currency": "$ (US dollars)",
        "greeting": (
            "Hi! I'm your business loan advisor. Want to hear about our loan "
            "products, check eligibility, or work out an EMI?"
        ),
    },
    "hi": {
        "label": "Hindi",
        "stt_language": "hi",
        "tts_language": "hi",
        "currency": "₹ (rupees)",
        "greeting": (
            "नमस्ते! मैं आपकी business loan advisor "
            "हूँ। आप loan products, eligibility, या EMI के "
            "बारे में पूछ सकते हैं।"
        ),
    },
    "gu": {
        "label": "Gujarati",
        "stt_language": "gu",
        "tts_language": "gu",
        "currency": "₹ (rupees)",
        "greeting": (
            "નમસ्તે! હું તમારી business loan "
            "advisor છું."
        ),
    },
    "mr": {
        "label": "Marathi",
        "stt_language": "mr",
        "tts_language": "mr",
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
        super().__init__(
            instructions=_BASE_PROMPT.format(
                currency=CFG["currency"], label=CFG["label"]
            ),
            agent_id=AGENT_ID,
            tools=[get_loan_products, calculate_emi, check_eligibility],
            pipeline=Pipeline(
                stt=DeepgramSTT(model="nova-2", language=CFG["stt_language"]),
                llm=GoogleLLM(model="gemini-2.5-flash"),
                tts=CartesiaTTS(model="sonic-3", language=CFG["tts_language"]),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(CFG["greeting"])

    async def on_exit(self) -> None:
        logger.info("call finished in %s", CFG["label"])


def on_ready() -> None:
    result = zrt.invoke(
        AGENT_ID, room=Room(
            name=f"Loan Advisor ({CFG['label']})", playground=True)
    )
    if "playground_url" in result:
        logger.info("playground: %s", result["playground_url"])


if __name__ == "__main__":
    logger.info("running in %s", CFG["label"])
    zrt.serve(MultilangLoanAgent, on_ready=on_ready)
