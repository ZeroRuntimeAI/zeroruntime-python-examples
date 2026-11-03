# A feedback call that follows a workflow, with no subagents: the agent's own
# steps, and its own tools to finish them.
#
#   rating -> reason (low ratings only) -> follow_up (optional) -> thanks
#
# The runtime keeps the order: each step is asked once its turn comes, and only
# a tool's self.complete(Answer(...)) finishes it.
#
#   python feedback_survey.py

import asyncio
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, Step, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# Answers: one dataclass per step

@dataclass
class Rating:
    score: int


@dataclass
class Reason:
    what_went_wrong: str


@dataclass
class FollowUp:
    email: str


@dataclass
class Thanks:
    said: bool


SURVEY = [
    Step(
        id="rating",
        description="their rating",
        instructions="Ask them to rate the delivery from 1 to 5, then call record_rating.",
        opening="Ask how they'd rate their delivery, from 1 to 5.",
        result=Rating,
        max_attempts=3,
    ),
    Step(
        id="reason",
        description="what went wrong",
        instructions="Ask what went wrong, listen, say sorry, then call record_reason.",
        opening="Say you're sorry to hear that, and ask what went wrong.",
        result=Reason,
        condition={"field": "rating.score", "in": [1, 2, 3]},  # only for a low rating
    ),
    Step(
        id="follow_up",
        description="a follow-up",
        instructions="Offer to have someone email them about it, then call record_follow_up with their email.",
        opening="Ask if they'd like someone to follow up by email.",
        result=FollowUp,
        optional=True,  # if they decline, the model calls the built-in skip_step
    ),
    Step(
        id="thanks",
        description="thank them",
        instructions="Thank them for their time, then call finish_survey.",
        opening="Thank them for their time.",
        result=Thanks,
    ),
]


class Feedback(Agent):
    def __init__(self) -> None:
        super().__init__(
            name="survey",
            instructions="You call QuickShip customers for feedback about a recent delivery: short and friendly, "
                         "under two minutes. This is a phone call: a sentence or two at a time. If they don't "
                         "have time, thank them and call end_call. Once the survey is done, call end_call.",
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector()),
            workflow=SURVEY,  # the agent's own steps
        )
        for event in ("step_start", "step_end"):
            self.on(event, print)

    async def on_enter(self):
        # The call starts: say() speaks these exact words; the first step's opening follows them.
        name = self.userdata.get("caller_name", "there")  # from invoke(metadata=...)
        await self.session.say(f"Hi {name}, this is QuickShip, calling for two minutes of feedback on your delivery.")

    async def on_exit(self):
        # Run once every step is done (it's the only agent, so the call stays here), and as the call
        # ends: this line is the goodbye.
        return "Thanks again, and have a lovely day."

    @function_tool
    async def record_rating(self, score: int) -> dict:
        """Record their rating.

        Args:
            score: 1 to 5.
        """
        if not 1 <= score <= 5:
            return {"error": "The rating is 1 to 5. Ask again."}
        return await self.complete(Rating(score))

    @function_tool
    async def record_reason(self, what_went_wrong: str) -> dict:
        """Record what went wrong.

        Args:
            what_went_wrong: In their words, briefly.
        """
        return await self.complete(Reason(what_went_wrong))

    @function_tool
    async def record_follow_up(self, email: str) -> dict:
        """Record the email for a follow-up.

        Args:
            email: Read it back to confirm before calling.
        """
        if "@" not in email:
            return {"error": "That isn't an email address. Ask them to spell it out."}
        return await self.complete(FollowUp(email.strip()))

    @function_tool
    async def finish_survey(self) -> dict:
        """They've been thanked; the survey is done."""
        rating, reason = self.workflow.rating, self.workflow.reason
        logger.info("survey: %s/5%s", rating.score, f", {reason.what_went_wrong}" if reason else "")
        return await self.complete(Thanks(True))

    @function_tool
    async def end_call(self) -> dict:
        """End the call."""
        asyncio.create_task(self.hangup(reason="caller done"))  # on_exit gives the goodbye
        return {"status": "ending_call"}


def on_ready() -> None:
    zeroruntime.invoke(room=Room(name="QuickShip feedback", playground=True), metadata={"caller_name": "Arjun"})


if __name__ == "__main__":
    zeroruntime.serve(Feedback, on_ready=on_ready)
