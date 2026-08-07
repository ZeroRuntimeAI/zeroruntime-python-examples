# The MCP server behind customer_agent.py: one stdio tool that posts the agent's
# question into a Discord thread and blocks until somebody replies. Standalone
# -- nothing here imports zrt, and discord is imported inside main.

import asyncio
import os
import sys



REPLY_TIMEOUT = float(os.getenv("HUMAN_REPLY_TIMEOUT", "240"))


class DiscordHuman:
    """One Discord thread, and whoever is reading it."""

    def __init__(self, user_id: int, channel_id: int) -> None:
        import discord
        from discord.ext import commands

        self.user_id = user_id
        self.channel_id = channel_id
        self._answers: "asyncio.Queue[str]" = asyncio.Queue()
        self.bot = commands.Bot(
            command_prefix="!", intents=discord.Intents.all())

        @self.bot.event
        async def on_ready() -> None:
            logger.info("connected as %s", self.bot.user)

        @self.bot.event
        async def on_message(message) -> None:
            if message.author.id == self.bot.user.id:
                return
            if message.channel.id == self.channel_id or getattr(
                message.channel, "parent_id", None
            ) == self.channel_id:
                await self._answers.put(message.content)

    async def start(self, token: str) -> None:
        await self.bot.start(token)

    async def ask(self, question: str) -> str:
        """Post the question and wait for the first human reply."""
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            return "I could not reach a supervisor."

        await channel.send(f"<@{self.user_id}> {question}")
        try:
            return await asyncio.wait_for(self._answers.get(), timeout=REPLY_TIMEOUT)
        except asyncio.TimeoutError:
            return "No supervisor replied in time."


async def main() -> None:
    from mcp.server import MCPServer

    token = os.getenv("DISCORD_TOKEN")
    user_id = os.getenv("DISCORD_USER_ID")
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    if not (token and user_id and channel_id):
        raise SystemExit(
            "set DISCORD_TOKEN, DISCORD_USER_ID and DISCORD_CHANNEL_ID"
        )

    human = DiscordHuman(int(user_id), int(channel_id))
    asyncio.create_task(human.start(token))

    server = MCPServer("DiscordHumanServer")

    @server.tool(
        description=(
            "Ask a human supervisor a question the agent must not answer "
            "itself, such as a discount percentage."
        )
    )
    async def ask_human(question: str) -> str:
        logger.info("asking a human: %s", question)
        answer = await human.ask(question)
        logger.info("human said: %s", answer)
        return answer

    await server.run_stdio_async()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
