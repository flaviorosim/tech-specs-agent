"""Compare original vs evidence-focused retrieval on ONE imported datasheet.

From repository root:
  python scripts/compare_evidence.py --document user-... --method hybrid \
      --question 'Which CPU architecture does this MCU use?'

Does NOT call Ollama, use benchmark answers, modify indexes, or rewrite PDFs.
Close Streamlit before Qdrant Local (vector/hybrid), to avoid an index lock.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tech_specs.custom_docs import get_custom_document
from tech_specs.web_runtime import AgentResources


def _display(label: str, hits: list[dict]):
    print(f"\n{label}")
    print("-" * 68)
    for hit in hits:
        c = hit["chunk"]
        content = str(c.get("content", "")).replace("\n", " ")
        content = " ".join(content.split())
        print(f"#{hit.get('rank')}  PDF page {c.get('page')}  {c.get('chunk_id')}")
        print("  " + content[:250] + ("..." if len(content) > 250 else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, help="Imported document ID shown in the UI/MCP catalog")
    parser.add_argument("--question", required=True)
    parser.add_argument("--method", choices=["bm25", "vector", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 20:
        parser.error("--top-k must be between 1 and 20")
    resources = AgentResources(ROOT)
    try:
        get_custom_document(ROOT, args.document)
        raw = resources._base_retriever(args.method, args.document)
        focused = resources._retriever(args.method, args.document)
        before = raw.search(args.question, top_k=args.top_k, document=args.document)
        after = focused.search(args.question, top_k=args.top_k, document=args.document)
        print("Focus:", focused.last_diagnostics.get("focus") or "none (original search retained)")
        print("Queries:", focused.last_diagnostics.get("queries"))
        print("Candidate count:", focused.last_diagnostics.get("candidate_count", "original"))
        print("Added PDF-neighbor candidates:", focused.last_diagnostics.get("neighbor_candidates", 0))
        _display("ORIGINAL", before)
        _display("EVIDENCE-FOCUSED", after)
    finally:
        resources.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
