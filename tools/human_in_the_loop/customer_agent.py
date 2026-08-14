# An agent that asks a human when it must not guess. ask_human is an ordinary
# MCP tool whose server waits on a person, so session_timeout has to be minutes
# rather than the vendor's 5s default meant for a local subprocess.

import logging
import os
import pathlib
import sys

import zeroruntime
from zeroruntime import Agent, MCPServerStdio, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import AnthropicLLM, DeepgramSTT, GoogleTTS, SileroVAD


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "customer-agent")
SERVER = pathlib.Path(__file__).parent / "discord_mcp_server.py"

HUMAN_TIMEOUT = float(os.getenv("HUMAN_REPLY_TIMEOUT", "300"))


class CustomerAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a customer-facing agent. You have tools to help with "
                "enquiries. When the caller asks about a discount percentage, "
                "always use the tool to get the answer from your human "
                "supervisor -- never estimate one. Tell the caller you are "
                "checking before you call it, because the answer takes a moment."
            ),
            agent_id=AGENT_ID,
            mcp_servers=[
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(SERVER)],
                    session_timeout=HUMAN_TIMEOUT,
                ),
            ],
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=AnthropicLLM(),
                tts=GoogleTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        names = [t._tool_info.name for t in self.tools]
        logger.info("%d tool(s): %s", len(names), ", ".join(names) or "none")
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Customer Agent", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(CustomerAgent, on_ready=on_ready)
