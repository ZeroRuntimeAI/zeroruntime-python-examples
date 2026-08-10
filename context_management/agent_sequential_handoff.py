# Passing the caller from one agent to the next: a tool that returns an Agent is
# the handoff. inherit_context carries the conversation across, and add_handoff
# records who moved them and why -- before the return, so it is inherited too.

import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD
from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "intake")


def build_pipeline() -> Pipeline:
    """A fresh pipeline per agent.

    Not one shared instance: the pipeline carries the hooks registered on it,
    and two agents sharing one would each receive the other's events.
    """
    return Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )


class BillingAgent(Agent):
    """The specialist. Greets the caller already knowing why they were moved."""

    def __init__(self, inherit_context: bool = False, reason: str = "") -> None:
        self._reason = reason
        super().__init__(
            instructions=(
                "You are the billing specialist. Resolve charge disputes, "
                "payment questions, and refunds."
            ),
            agent_id="billing",
            pipeline=build_pipeline(),
            inherit_context=inherit_context,
        )

    async def on_enter(self) -> None:
        if self._reason:
            await self.session.say(
                f"I'm the billing specialist -- I see you're here about "
                f"{self._reason}. Let's get that sorted."
            )
        else:
            await self.session.say(
                "I'm the billing specialist. What can I help you with?"
            )

        await self.session.add_message(
            "assistant", "[billing agent engaged]", agent_id=self.id
        )

    async def on_exit(self) -> None:
        logger.info("billing finished")


class IntakeAgent(Agent):
    """First line. Decides who the caller actually needs."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the first line of support. Find out what the caller "
                "needs. If it is about a charge, a payment or a refund, call "
                "transfer_to_billing with a short reason."
            ),
            agent_id=AGENT_ID,
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi, you've reached support. How can I help?")

    async def on_exit(self) -> None:
        logger.info("intake finished")

    @function_tool
    async def transfer_to_billing(self, reason: str) -> Agent:
        """Transfer the caller to the billing specialist.

        Args:
            reason: Short reason for the transfer, e.g. "disputed charge".
        """
        logger.info("transferring to billing: %s", reason)

        await self.session.add_handoff(
            to_agent="billing", from_agent=self.id, reason=reason
        )

        return BillingAgent(inherit_context=True, reason=reason)


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Sequential Handoff", playground=True))


if __name__ == "__main__":
    zrt.serve(IntakeAgent, on_ready=on_ready)
