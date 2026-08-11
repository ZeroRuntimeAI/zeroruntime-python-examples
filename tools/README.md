# Tools

Giving the agent something to call. Nothing here needs an extra install —
function tools are built in, and the MCP client ships with `zeroruntime`.

## mcp_example.py

Connects to `mcp_servers/current_time.py`, a stdio server this repo ships. The
agent spawns it as a subprocess, so the path is resolved relative to
`mcp_example.py` rather than your working directory. Point `MCP_SERVER` at
another script to use a different one.

Nothing to configure — run it.

## Subfolders with their own setup

| Folder | Needs |
| --- | --- |
| `human_in_the_loop/` | a Discord bot — see its README |
| `n8n_workflow/` | a running n8n instance — see its README |
