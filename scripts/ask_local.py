"""Ask a local Ollama LLM questions grounded in the indexed datasheets.

Only the Ollama loopback endpoint is contacted. BM25 works without a vector DB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.answer_graph import build_answer_graph
from tech_specs.local_answer import LocalOllama
from tech_specs.retrieval import BM25Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about STM32L432KC or ATtiny25/45/85")
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--document", choices=("stm32l432kc", "attiny85"))
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates-per-facet", type=int, default=10)
    parser.add_argument("--corpus", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=PROCESSED / "qdrant_local")
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "vector_index_manifest.json")
    parser.add_argument("--output", type=Path, help="Optional output state JSON; never overwritten")
    args = parser.parse_args()
    if args.top_k < 1 or args.candidates_per_facet < 1:
        parser.error("--top-k and --candidates-per-facet must be positive")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    chunks = load_chunks(args.corpus)
    bm25 = BM25Retriever(chunks) if args.method != "vector" else None
    client = None
    try:
        if args.method == "bm25":
            retriever = bm25
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
            vectors = VectorRetriever(client, TextEmbedding(model_name=MODEL_NAME))
            if args.method == "vector":
                retriever = vectors
            else:
                from tech_specs.hybrid_retrieval import HybridRetriever
                retriever = HybridRetriever(bm25, vectors)

        llm = LocalOllama(model=args.model, num_ctx=args.num_ctx, num_predict=args.num_predict)
        graph = build_answer_graph(
            retriever, llm,
            top_k=args.top_k, candidates_per_facet=args.candidates_per_facet,
        )
        result = graph.invoke({"question": args.question, "document_hint": args.document})
        print(f"\nMethod: {args.method} | model: {args.model} | status: {result['status']}")
        print("Facets: " + ", ".join(f["id"] for f in result["plan"]["facets"]))
        print("\n" + result["report"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nReport saved to: {args.output}")
    except RuntimeError as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
