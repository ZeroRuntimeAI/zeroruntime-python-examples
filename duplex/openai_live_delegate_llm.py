# Client delegation answered by a model you choose: GPT-Live runs the
# conversation, and every delegation goes to Gemini, which runs the agent's own
# function tools. The ZeroRuntime builds the Gemini model next to GPT-Live --
# credentials, route and billing like any llm slot -- so there is no delegation
# handler to write, and the tool bodies below run in this process as usual.

import logging
import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, function_tool
from zeroruntime.plugins import  OpenAIDelegateLLMConfig, OpenAILive
from zeroruntime.inference import GoogleLLM

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "openai-live-delegate-llm")

ORDERS = {
    "B1102": "shipped and arriving Thursday",
    "A2133": "still being packed",
}


@function_tool
async def check_order_status(order_id: str) -> dict:
    """Check the delivery status of an order.

    Args:
        order_id: The order reference, such as B1102.
    """
    print(f"[TOOLCALL] check_order_status tool called with order_id: {order_id}")
    status = ORDERS.get(order_id.upper().replace(" ", ""))
    logger.info("order %s -> %s", order_id, status)
    return {"order_id": order_id, "status": status or "no such order"}


@function_tool
async def schedule_delivery(order_id: str, day: str) -> dict:
    """Book a delivery day for an order that has shipped.

    Args:
        order_id: The order reference, such as A1042.
        day: The requested delivery day, such as Tuesday.
    """
    print(f"[TOOLCALL] schedule_delivery tool called with order_id: {order_id}, day: {day}")
    if order_id.upper().replace(" ", "") not in ORDERS:
        return {"booked": False, "reason": f"no order {order_id}"}
    logger.info("booked %s for %s", order_id, day)
    return {"booked": True, "order_id": order_id, "day": day}


class DelegateLLMAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant for an online furniture store. Keep "
                "spoken replies brief. Delegation policy: delegate order status "
                "questions and delivery bookings to the backend. Do not guess a "
                "result or confirm a booking before the backend reports it."
            ),
            agent_id=AGENT_ID,
            # run by Gemini while it answers a delegation, like cascade tools
            tools=[check_order_status, schedule_delivery],
            pipeline=Pipeline(
                llm=OpenAILive(
                    model="gpt-live-1",
                    voice="marin",
                    config=OpenAIDelegateLLMConfig(
                        llm=GoogleLLM(model="gemini-3-flash-preview", temperature=0.2),
                        instructions=(
                            "You are the order desk for an online furniture store. "
                            "Use your tools for order facts. Only book a delivery for "
                            "an order that exists, on the day the caller confirmed."
                        ),
                    ),
                ),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you with your order today?")

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="OpenAI Live + Gemini Delegate", playground=True))

if __name__ == "__main__":
    zeroruntime.serve(DelegateLLMAgent, on_ready=on_ready)
