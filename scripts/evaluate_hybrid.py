
"""Evaluate BM25 + vector RRF with existing questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from evaluate_bm25 import load_questions

from tech_specs.evaluation import (
    score_ranking,
    summarize_scores,
)

from tech_specs.hybrid_retrieval import HybridRetriever
from tech_specs.retrieval import BM25Retriever, load_chunks

from tech_specs.vector_retrieval import (
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST,
    MODEL_NAME,
    VectorRetriever,
    load_manifest,
    verify_index,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS = (
    ROOT_DIR / "data" / "processed" / "chunks.jsonl"
)

DEFAULT_QUESTIONS = (
    ROOT_DIR / "evaluation" / "questions.json"
)

DEFAULT_OUTPUT = (
    ROOT_DIR / "data" / "processed" / "hybrid_eval.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if min(
        args.top_k, args.candidate_k, args.rrf_k
    ) < 1:
        parser.error(
            "top-k, candidate-k and rrf-k must be >= 1"
        )

    chunks = load_chunks(args.corpus)

    questions = load_questions(
        args.questions,
        {
            chunk["chunk_id"]
            for chunk in chunks
        },
    )

    manifest = load_manifest(args.manifest)

    if manifest.get("partial_index"):
        raise ValueError(
            "Cannot evaluate a partial vector index"
        )

    client = QdrantClient(
        path=str(args.db_path)
    )

    try:
        verify_index(
            client,
            manifest,
            corpus_path=args.corpus,
        )

        searcher = HybridRetriever(
            BM25Retriever(chunks),
            VectorRetriever(
                client,
                TextEmbedding(model_name=MODEL_NAME),
            ),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )

        rows = []

        for question in questions:
            hits = searcher.search(
                question["query"],
                top_k=args.top_k,
                document=question.get("document"),
            )

            ids = [
                hit["chunk"]["chunk_id"]
                for hit in hits
            ]

            metrics = score_ranking(
                ids,
                question["relevant_chunk_ids"],
                args.top_k,
            )

            rows.append({
                "id": question["id"],
                "query": question["query"],
                "document": question.get("document"),
                "relevant_chunk_ids": (
                    question["relevant_chunk_ids"]
                ),
                "retrieved_chunk_ids": ids,
                **metrics,
            })

            print(
                f"\n{question['id']} "
                f"— {question['query']}"
            )

            print(
                f"  Top-{args.top_k}: {ids}"
            )

            print(
                "  First relevant rank: "
                f"{metrics['first_relevant_rank']}"
            )

            print(
                f"  Hit@{args.top_k}: "
                f"{metrics['hit_at_k']} "
                f"| Recall@{args.top_k}: "
                f"{metrics['recall_at_k']:.2f}"
            )

        summary = summarize_scores(rows)

        report = {
            "retriever": "hybrid_rrf",
            "model": MODEL_NAME,
            "k": args.top_k,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "summary": summary,
            "questions": rows,
        }

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"\n=== Hybrid RRF over "
            f"{summary['queries']} questions ==="
        )

        print(
            f"Hit@{args.top_k}: "
            f"{summary['hit_rate_at_k']:.3f}"
        )

        print(
            f"Mean Recall@{args.top_k}: "
            f"{summary['mean_recall_at_k']:.3f}"
        )

        print(
            f"MRR@{args.top_k}: "
            f"{summary['mrr_at_k']:.3f}"
        )

        print(f"Report saved to: {args.output}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
