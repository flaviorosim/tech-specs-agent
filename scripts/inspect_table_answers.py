"""Inspect source-derived table answers from a previous guarded LangGraph JSON.

This does not invoke Ollama, rerun retrieval, or modify the original report.
It is an extraction diagnostic, NOT a replacement for factual evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.table_answers import answer_from_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write table extraction JSON (must not exist)")
    args = parser.parse_args()
    if not args.report.is_file():
        parser.error(f"Report not found: {args.report}")
    if args.output and args.output.exists():
        parser.error(f"Output exists: {args.output}")
    try:
        record = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read JSON: {exc}")
    original = record.get("original_result") or record
    question = record.get("question") or original.get("question")
    hits = original.get("retrieval", {}).get("hits", [])
    if not question or not isinstance(hits, list):
        parser.error("Expected a guarded agent JSON with question and original_result.retrieval.hits")
    extraction = answer_from_tables(question, hits)
    if extraction is None:
        print("No unambiguous recognized table found. Keep the existing RAG process.")
        return
    payload = {
        "question": question,
        "kind": extraction.kind,
        "source_id": extraction.source_id,
        "cells": list(extraction.cells),
        "draft": extraction.draft,
        "scope": "literal Markdown-table cells extracted from retrieved chunks; not independently verified",
    }
    print(f"Table type: {extraction.kind} | Source: {extraction.source_id}")
    print("\n" + extraction.draft)
    print("\nNote: this is source-cell extraction, not an independent PDF or semantic audit.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            parser.exit(2, f"Could not write JSON: {exc}\n")
        print(f"\nExtraction saved to: {args.output}")


if __name__ == "__main__":
    main()
