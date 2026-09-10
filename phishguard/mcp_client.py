"""Client for the FastMCP tool server.

Tools run in a real MCP server subprocess over stdio, not as in-process function
calls. One session is opened per batch and every call is individually timed out,
so a hanging tool degrades to "unavailable" instead of stalling the graph.
"""

import asyncio
import json
import os
import sys
import threading
from typing import Any

from . import config

_SERVER_UNAVAILABLE = "mcp_server_unavailable"


def _unavailable(tool: str, args: dict[str, Any], detail: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": "unavailable",
        "detail": detail,
        "args": args,
        "failure": _SERVER_UNAVAILABLE,
    }


def _parse(result: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps plain return values under "result".
        payload = structured.get("result", structured)
        if isinstance(payload, dict):
            return payload

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return _unavailable(tool, args, "Tool returned no parsable result.")


async def _run_batch(calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(config.MCP_SERVER)],
        cwd=str(config.ROOT),
        env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR", "FASTMCP_SHOW_CLI_BANNER": "false"},
    )

    results: list[dict[str, Any]] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=config.TOOL_TIMEOUT * 2)
            for name, args in calls:
                try:
                    raw = await asyncio.wait_for(
                        session.call_tool(name, args), timeout=config.TOOL_TIMEOUT
                    )
                    parsed = _parse(raw, name, args)
                    parsed.setdefault("tool", name)
                    parsed["args"] = args
                    results.append(parsed)
                except asyncio.TimeoutError:
                    results.append(
                        _unavailable(
                            name, args,
                            f"Tool exceeded the {config.TOOL_TIMEOUT:.0f}s timeout.",
                        )
                    )
                except Exception as exc:
                    results.append(
                        _unavailable(name, args, f"Tool call failed: {type(exc).__name__}: {exc}")
                    )
    return results


def call_tools(calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Run a batch of tool calls. Never raises; failures come back as unavailable."""
    if not calls:
        return []

    box: dict[str, Any] = {}

    def worker() -> None:
        # A dedicated thread with its own event loop keeps this safe to call from
        # Streamlit's script thread, which may already have a loop attached.
        try:
            box["result"] = asyncio.run(_run_batch(calls))
        except Exception as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=config.TOOL_TIMEOUT * len(calls) + 30)

    if "result" in box:
        return box["result"]

    detail = box.get("error", "MCP server did not respond within the batch deadline.")
    return [_unavailable(name, args, detail) for name, args in calls]


def server_available() -> tuple[bool, str]:
    """Cheap health probe used by the UI to show tool-server status."""
    if not config.MCP_SERVER.exists():
        return False, f"{config.MCP_SERVER.name} not found."
    results = call_tools([("check_homoglyph", {"domain": "paypal-secure-login.com"})])
    if results and results[0].get("status") == "ok":
        return True, "MCP tool server responding."
    return False, results[0].get("detail", "No response.") if results else "No response."
