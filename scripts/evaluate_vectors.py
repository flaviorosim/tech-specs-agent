"""Evaluate semantic Qdrant search using the existing BM25 question labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# Reuse the exact benchmark validation used in evaluate_bm25.py.
from evaluate_bm25 import load_questions
from tech_specs.evaluation import score_ranking, summarize_scores
from tech_specs.retrieval import load_chunks
from tech_specs.vector_retrieval import (
    DEFAULT_DB_PATH, DEFAULT_MANIFEST, MODEL_NAME,
    VectorRetriever, load_manifest, verify_index,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
DEFAULT_QUESTIONS = ROOT_DIR / "evaluation" / "questions.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "vector_eval.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be >= 1")

    chunks = load_chunks(args.corpus)
    questions = load_questions(args.questions, {c["chunk_id"] for c in chunks})
    manifest = load_manifest(args.manifest)
    if manifest.get("partial_index"):
        raise ValueError("Cannot benchmark a partial index. Build the full collection first")

    client = QdrantClient(path=str(args.db_path))
    try:
        verify_index(client, manifest, corpus_path=args.corpus)
        retriever = VectorRetriever(client, TextEmbedding(model_name=MODEL_NAME))
        rows = []
        for question in questions:
            hits = retriever.search(
                question["query"],
                top_k=args.top_k,
                document=question.get("document"),
            )
            ids = [hit["chunk"]["chunk_id"] for hit in hits]
            metrics = score_ranking(ids, question["relevant_chunk_ids"], args.top_k)
            rows.append({
                "id": question["id"],
                "query": question["query"],
                "document": question.get("document"),
                "relevant_chunk_ids": question["relevant_chunk_ids"],
                "retrieved_chunk_ids": ids,
                **metrics,
            })
            print(f"\n{question['id']} — {question['query']}")
            print(f"  Top-{args.top_k}: {ids}")
            print(f"  First relevant rank: {metrics['first_relevant_rank']}")
            print(f"  Hit@{args.top_k}: {metrics['hit_at_k']} | Recall@{args.top_k}: {metrics['recall_at_k']:.2f}")

        summary = summarize_scores(rows)
        report = {"retriever": "vector", "model": MODEL_NAME, "k": args.top_k,
                  "summary": summary, "questions": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n=== Vector search over {summary['queries']} questions ===")
        print(f"Hit@{args.top_k}: {summary['hit_rate_at_k']:.3f}")
        print(f"Mean Recall@{args.top_k}: {summary['mean_recall_at_k']:.3f}")
        print(f"MRR@{args.top_k}: {summary['mrr_at_k']:.3f}")
        print(f"Report saved to: {args.output}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
