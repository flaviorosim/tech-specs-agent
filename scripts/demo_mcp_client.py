"""Demonstrate real MCP interoperability by launching the server over stdio.

Run from the project root: python scripts/demo_mcp_client.py --question "ADC resolution"
Defaults to a search-only call; --ask also invokes the local Qwen3 RAG agent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


def show_result(result) -> None:
    if result.is_error:
        raise RuntimeError("MCP tool error: " + " ".join(getattr(x, "text", "") for x in result.content))
    value = result.structured_content
    if value is None:
        value = [getattr(x, "text", "") for x in result.content]
    print(json.dumps(value, ensure_ascii=False, indent=2))


async def demo(document: str, question: str, method: str, ask: bool) -> None:
    params = StdioServerParameters(command=sys.executable,
        args=[str(ROOT / "scripts" / "mcp_server.py")],
        env={"TECH_SPECS_ROOT": str(ROOT), "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")})
    async with Client(params) as client:
        tools = await client.list_tools()
        print("Exposed MCP tools:", ", ".join(t.name for t in tools.tools))
        print("\nDatasheets:")
        show_result(await client.call_tool("list_datasheets", {}))
        print("\nSearch result:")
        search = await client.call_tool("search_datasheet", {
            "question": question, "document": document, "method": method, "top_k": 3})
        show_result(search)
        if ask:
            print("\nLocal RAG answer:")
            show_result(await client.call_tool("ask_datasheet", {
                "question": question, "document": document, "method": method}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", default="attiny85", help="Exact document ID from list_datasheets")
    parser.add_argument("--question", default="How does BOD affect sleep power consumption?")
    parser.add_argument("--method", choices=["bm25", "vector", "hybrid"], default="bm25")
    parser.add_argument("--ask", action="store_true", help="Also run Ollama/verification")
    args = parser.parse_args()
    asyncio.run(demo(args.document, args.question, args.method, args.ask))
