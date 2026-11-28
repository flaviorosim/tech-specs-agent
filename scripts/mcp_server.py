"""Local read-only MCP server for Tech-Specs Agent (Python MCP SDK v2).

Run from the repository root: python scripts/mcp_server.py
A host starts this subprocess and communicates using MCP over stdio.
Do not print or log to stdout: it is reserved for MCP JSON-RPC messages.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path

from mcp.server import MCPServer
from tech_specs.mcp_service import TechSpecsMCPService

PROJECT_ROOT = Path(os.environ.get("TECH_SPECS_ROOT") or Path(__file__).resolve().parents[1]).resolve()
SERVICE = TechSpecsMCPService(PROJECT_ROOT)


def build_server(service: TechSpecsMCPService) -> MCPServer:
    server = MCPServer("Tech-Specs Agent")

    @server.tool()
    def list_datasheets() -> dict:
        """List the available microcontroller datasheets and their exact document IDs."""
        return service.list_datasheets()

    @server.tool()
    def search_datasheet(question: str, document: str, method: str = "bm25", top_k: int = 5) -> dict:
        """Search one indexed datasheet; return ranked technical excerpts, page and chunk IDs. No LLM involved."""
        return service.search_datasheet(question, document, method=method, top_k=top_k)

    @server.tool()
    def get_chunk(document: str, chunk_id: str) -> dict:
        """Read a retrieved chunk's full stored text and PDF page using its exact ID."""
        return service.get_chunk(document, chunk_id)

    @server.tool()
    def ask_datasheet(question: str, document: str, method: str = "bm25") -> dict:
        """Ask the locally hosted LangGraph/Ollama RAG agent; respect verification and answer-withholding states."""
        return service.ask_datasheet(question, document, method=method)

    return server


mcp = build_server(SERVICE)

if __name__ == "__main__":
    atexit.register(SERVICE.close)
    mcp.run()  # stdio, no port or external API
