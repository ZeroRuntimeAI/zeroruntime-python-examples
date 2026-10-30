# A support line of three subagents and no steps: reception finds out what the
# caller needs and hands them to billing or tech support, who can hand them on.
#
# Without steps, a subagent is just its own instructions and tools: it talks
# until one of its tools moves the call with transfer_to. Every subagent can
# transfer to every other; the first one opens the call.
#
#   python support_desk.py

import asyncio

from dotenv import load_dotenv

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, SubAgent, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

load_dotenv(override=True)

INVOICES = {"INV-1001": {"amount": 1499, "status": "paid"}, "INV-1002": {"amount": 2999, "status": "due"}}
SERVICES = {"internet": "running", "email": "slow in your region; a fix is due within the hour", "tv": "running"}


class Reception(SubAgent):
    name = "reception"
    instructions = (
        "You're the front desk of Nimbus Telecom. Find out, in a sentence or two, whether the caller "
        "needs help with a bill or with a service that isn't working, then hand them over."
    )

    async def on_enter(self, reason: str):
        if reason == "call started":
            name = self.userdata.get("caller_name", "the caller")  # from invoke(metadata=...)
            return f"Greet {name} as Nimbus Telecom, and ask how you can help."
        return "Ask what else you can help with."

    @function_tool
    async def to_billing(self, summary: str) -> dict:
        """Hand the caller to billing: invoices, payments, refunds.

        Args:
            summary: What they need, in a few words.
        """
        return await self.transfer_to("billing", reason=summary)

    @function_tool
    async def to_tech_support(self, summary: str) -> dict:
        """Hand the caller to tech support: a service that's down, slow or misbehaving.

        Args:
            summary: What's wrong, in a few words.
        """
        return await self.transfer_to("tech_support", reason=summary)


class Billing(SubAgent):
    name = "billing"
    instructions = "You handle invoices and payments. Only quote what look_up_invoice returned."

    async def on_enter(self, reason: str):
        # reason is what reception passed on: the caller isn't asked to repeat it.
        return f"Say you're in billing and can help with: {reason}. Ask for their invoice number."

    @function_tool
    async def look_up_invoice(self, invoice_id: str) -> dict:
        """Find an invoice.

        Args:
            invoice_id: The invoice number, e.g. INV-1001.
        """
        invoice = INVOICES.get(invoice_id.strip().upper())
        if invoice is None:
            return {"error": f"No invoice {invoice_id}. Ask them to read the number again."}
        return {"invoice_id": invoice_id.upper(), **invoice}

    @function_tool
    async def back_to_reception(self) -> dict:
        """The caller needs something billing doesn't handle."""
        return await self.transfer_to("reception", reason="needs something else")

    @function_tool
    async def to_tech_support(self, summary: str) -> dict:
        """The caller's billing question is really a service problem.

        Args:
            summary: What's wrong, in a few words.
        """
        return await self.transfer_to("tech_support", reason=summary)


class TechSupport(SubAgent):
    name = "tech_support"
    instructions = "You fix service problems. Check the service's status before suggesting anything."

    async def on_enter(self, reason: str):
        return f"Say you're in tech support and can look into: {reason}. Ask which service it is if they haven't said."

    @function_tool
    async def check_status(self, service: str) -> dict:
        """Check a service's status in the caller's area.

        Args:
            service: internet, email or tv.
        """
        await self.session.say("Let me check that for you.", add_to_chat_context=False)
        status = SERVICES.get(service.strip().lower())
        if status is None:
            return {"error": f"We look after {', '.join(SERVICES)}. Ask which one it is."}
        return {"service": service.lower(), "status": status}

    @function_tool
    async def back_to_reception(self) -> dict:
        """The caller needs something tech support doesn't handle."""
        return await self.transfer_to("reception", reason="needs something else")


class Desk(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You answer the phone at Nimbus Telecom. This is a phone call: a sentence or two at a time. "
                         "Once the caller has nothing else, call end_call.",
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector()),
            subagents=[Reception(), Billing(), TechSupport()],  # reception opens the call
            farewell="Thanks for calling Nimbus Telecom. Have a good day.",  # whichever subagent ends the call
        )
        self.on("agent_handoff", print)

    # The agent's own tool methods are shared: every subagent can call them.

    @function_tool
    async def end_call(self) -> dict:
        """End the call when the caller says goodbye or has nothing else."""
        asyncio.create_task(self.hangup(reason="caller done"))
        return {"status": "ending_call"}


def on_ready() -> None:
    zeroruntime.invoke(room=Room(name="Nimbus Telecom", playground=True), metadata={"caller_name": "Meera"})


if __name__ == "__main__":
    zeroruntime.serve(Desk, on_ready=on_ready)
