"""Run the first automatic LangGraph retrieval workflow (no LLM/API key)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.agent_graph import build_agent_graph
from tech_specs.retrieval import BM25Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Technical question in natural language")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--document", choices=("stm32l432kc", "attiny85"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates-per-facet", type=int, default=10)
    parser.add_argument("--corpus", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=PROCESSED / "qdrant_local")
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "vector_index_manifest.json")
    parser.add_argument("--expected", action="append", default=[], metavar="CHUNK_ID")
    parser.add_argument("--output", type=Path, help="Write full state as JSON (will not overwrite)")
    parser.add_argument("--full", action="store_true", help="Print full evidence text")
    args = parser.parse_args()
    if args.top_k < 1 or args.candidates_per_facet < 1:
        parser.error("--top-k and --candidates-per-facet must be >= 1")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    chunks = load_chunks(args.corpus)
    client = None
    try:
        bm25 = BM25Retriever(chunks) if args.method != "vector" else None
        if args.method == "bm25":
            searcher = bm25
        else:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from tech_specs.vector_retrieval import (
                MODEL_NAME, VectorRetriever, load_manifest, verify_index,
            )
            manifest = load_manifest(args.manifest)
            if manifest.get("partial_index"):
                raise ValueError("Cannot search a partial vector index")
            client = QdrantClient(path=str(args.db_path))
            verify_index(client, manifest, corpus_path=args.corpus)
            semantic = VectorRetriever(client, TextEmbedding(model_name=MODEL_NAME))
            if args.method == "vector":
                searcher = semantic
            else:
                from tech_specs.hybrid_retrieval import HybridRetriever
                searcher = HybridRetriever(bm25, semantic)

        graph = build_agent_graph(
            searcher,
            top_k=args.top_k,
            candidates_per_facet=args.candidates_per_facet,
        )
        state = graph.invoke({"question": args.question, "document_hint": args.document})
        print("\n" + state["report"])
        print("\nAutomatic plan:")
        for facet in state["plan"]["facets"]:
            print(f"  {facet['id']}: {facet['query']}")

        if args.expected:
            from tech_specs.facet_retrieval import anchor_coverage
            coverage = anchor_coverage(state["retrieval"], args.expected)
            state["expected_anchor_coverage"] = coverage
            print(
                "\nReference anchor coverage (not answer completeness): "
                f"{len(coverage['found'])}/{len(args.expected)}"
            )
            if coverage["missing"]:
                print("Missing: " + ", ".join(coverage["missing"]))

        if args.full:
            print("\nEvidence content:")
            for hit in state["retrieval"]["hits"]:
                print("\n" + "=" * 60)
                print(f"{hit['chunk']['chunk_id']} — {hit['chunk']['content']}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nReport saved to: {args.output}")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
