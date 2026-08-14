# A small stdio MCP server, so mcp_example.py has something to connect to.
# Standalone: nothing here imports zeroruntime. Written against both the mcp package's
# 1.x and 2.x spellings of the server class.

import datetime
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from mcp.server import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("CurrentTimeServer")


@server.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: An IANA timezone name, e.g. "Asia/Kolkata" or "US/Pacific".
            Defaults to UTC.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"I do not know the timezone {timezone!r}. Try an IANA name like Asia/Kolkata."

    now = datetime.datetime.now(zone)
    return f"It is {now:%H:%M on %A, %d %B %Y} in {timezone}."


@server.tool()
def days_until(date: str) -> str:
    """How many days from today until a given date.

    Args:
        date: The target date as YYYY-MM-DD.
    """
    try:
        target = datetime.date.fromisoformat(date)
    except ValueError:
        return f"{date!r} is not a date I can read. Use YYYY-MM-DD."

    delta = (target - datetime.date.today()).days
    if delta == 0:
        return "That is today."
    if delta < 0:
        return f"That was {abs(delta)} day(s) ago."
    return f"That is {delta} day(s) away."


if __name__ == "__main__":
    sys.exit(server.run("stdio"))
