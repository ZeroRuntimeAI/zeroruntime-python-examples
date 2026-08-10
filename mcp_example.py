# Giving the agent tools from an MCP server. The connection is made here, in
# your process -- a stdio server needs your files and credentials -- and the
# discovered tools travel like any other. Needs the [mcp] extra.

import logging
import os
import sys
from pathlib import Path

import zrt
from zrt import Agent, MCPServerStdio, Pipeline, Room
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "mcp-agent")

MCP_SERVER = Path(
    os.getenv("MCP_SERVER", Path(__file__).parent /
              "mcp_servers" / "current_time.py")
)


class MCPAgent(Agent):
    def __init__(self) -> None:
        if not MCP_SERVER.exists():
            raise FileNotFoundError(
                f"no MCP server at {MCP_SERVER}. Set MCP_SERVER to one you have."
            )

        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks. You have tools available -- use them rather than "
                "guessing."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(model="nova-2"),
                llm=GoogleLLM(model="gemini-2.5-flash"),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
            mcp_servers=[
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(MCP_SERVER)],
                    session_timeout=30,
                ),
            ],
        )

    async def on_enter(self) -> None:
        names = [t._tool_info.name for t in self.tools]
        logger.info("%d tool(s) available: %s", len(names), ", ".join(names))

        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(name="MCP Agent", playground=True))


if __name__ == "__main__":
    zrt.serve(MCPAgent, on_ready=on_ready)
