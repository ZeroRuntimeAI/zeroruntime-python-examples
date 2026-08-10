# One caller, three agents: the front agent can hand off in either of two
# directions and each specialist inherits the conversation. Each agent gets its
# own pipeline instance, since a pipeline carries the hooks registered on it.


import logging
import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "travel")


def build_pipeline() -> Pipeline:
    return Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )


class BookingAgent(Agent):
    def __init__(self, inherit_context: bool = False) -> None:
        super().__init__(
            instructions=(
                "You are the booking specialist. Help the caller choose and "
                "book flights and hotels."
            ),
            agent_id="booking",
            pipeline=build_pipeline(),
            inherit_context=inherit_context,
        )

    async def on_enter(self) -> None:
        await self.session.say("I can help with your booking. What did you have in mind?")

    async def on_exit(self) -> None:
        logger.info("booking finished")


class TravelSupportAgent(Agent):
    def __init__(self, inherit_context: bool = False) -> None:
        super().__init__(
            instructions=(
                "You are travel support. Handle cancellations, delays, changes "
                "and anything that has gone wrong with an existing trip."
            ),
            agent_id="travel-support",
            pipeline=build_pipeline(),
            inherit_context=inherit_context,
        )

    async def on_enter(self) -> None:
        await self.session.say("I'm travel support. Tell me what has gone wrong.")

    async def on_exit(self) -> None:
        logger.info("support finished")


class TravelAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a travel assistant. Work out what the caller needs. "
                "For booking a new trip, call transfer_to_booking. For a problem "
                "with an existing trip, call transfer_to_travel_support."
            ),
            agent_id=AGENT_ID,
            pipeline=build_pipeline(),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! Are you booking a trip, or is something wrong with one?")

    async def on_exit(self) -> None:
        logger.info("travel agent finished")

    @function_tool
    async def transfer_to_booking(self) -> Agent:
        """Transfer the caller to the booking specialist."""
        logger.info("-> booking")
        await self.session.add_handoff(to_agent="booking", from_agent=self.id)
        return BookingAgent(inherit_context=True)

    @function_tool
    async def transfer_to_travel_support(self) -> Agent:
        """Transfer the caller to travel support."""
        logger.info("-> travel support")
        await self.session.add_handoff(to_agent="travel-support", from_agent=self.id)
        return TravelSupportAgent(inherit_context=True)


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(name="Multi Agent Switch", playground=True))


if __name__ == "__main__":
    zrt.serve(TravelAgent, on_ready=on_ready)
