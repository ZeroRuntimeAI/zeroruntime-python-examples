# An agent whose tools are an n8n workflow, reached over HTTP through its MCP
# trigger node -- same tools and same wire as a stdio server, different
# transport. The pipeline is tuned for a phone caller. Needs the [mcp] extra.

import logging
import os

import zeroruntime
from zeroruntime import Agent, EOUConfig, MCPServerHTTP, Pipeline, Room
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD


logger = logging.getLogger(__name__)


AGENT_ID = os.getenv("AGENT_ID", "appointment-agent")
N8N_URL = os.getenv(
    "N8N_MCP_URL", "https://your-n8n-instance/mcp/your-trigger-id")

INSTRUCTIONS = (
    "You are a restaurant's appointment assistant, speaking on the phone. Help "
    "the caller check, book, move or cancel a reservation. Use the tools rather "
    "than guessing -- you have no knowledge of the diary beyond what they "
    "return. Confirm the date and time back to the caller before you commit to "
    "anything. Keep replies short; they are spoken aloud."
)


class AppointmentAgent(Agent):
    def __init__(self) -> None:
        if "your-n8n-instance" in N8N_URL:
            logger.warning(
                "set N8N_MCP_URL -- the placeholder resolves to nothing")

        super().__init__(
            instructions=INSTRUCTIONS,
            agent_id=AGENT_ID,
            mcp_servers=[
                MCPServerHTTP(
                    endpoint_url=N8N_URL,
                    request_headers=(
                        {"Authorization":
                            f"Bearer {os.environ['N8N_API_KEY']}"}
                        if os.getenv("N8N_API_KEY")
                        else None
                    ),
                    session_timeout=30.0,
                ),
            ],
            pipeline=Pipeline(
                stt=DeepgramSTT(model="nova-2"),
                llm=GoogleLLM(model="gemini-2.5-flash"),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
                eou_config=EOUConfig(mode="ADAPTIVE",
                              min_max_speech_wait_timeout=[0.6, 1.4]),
            ),
        )

    async def on_enter(self) -> None:
        names = [t._tool_info.name for t in self.tools]
        logger.info("%d tool(s) from the workflow: %s",
                    len(names), ", ".join(names))
        await self.session.say(
            "Thanks for calling. Are you booking a table, or changing a reservation?"
        )

    async def on_exit(self) -> None:
        logger.info("call finished")


async def on_ready() -> None:
    zeroruntime.invoke(AGENT_ID, room=Room(
        name="Restaurant Agent", playground=True))


if __name__ == "__main__":
    zeroruntime.serve(AppointmentAgent, on_ready=on_ready)
