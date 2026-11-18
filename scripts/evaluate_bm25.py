"""Evaluate BM25 on a small manually labeled technical-question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.evaluation import score_ranking, summarize_scores
from tech_specs.retrieval import BM25Retriever, load_chunks

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT_DIR / "data" / "processed" / "chunks.jsonl"
DEFAULT_QUESTIONS = ROOT_DIR / "evaluation" / "questions.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "processed" / "bm25_eval.json"


def load_questions(path: Path, known_ids: set[str]) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        questions = json.load(source)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Question file must be a nonempty JSON array")

    seen = set()
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each question must be a JSON object")
        query_id = item.get("id")
        query = item.get("query")
        gold = item.get("relevant_chunk_ids")
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError(f"Invalid or duplicate question id: {query_id!r}")
        seen.add(query_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Question {query_id}: missing query")
        if not isinstance(gold, list) or not gold or not all(
            isinstance(chunk_id, str) and chunk_id for chunk_id in gold
        ):
            raise ValueError(f"Question {query_id}: missing relevant_chunk_ids")
        missing = sorted(set(gold) - known_ids)
        if missing:
            raise ValueError(
                f"Question {query_id}: labeled chunks not found in corpus: {missing}. "
                "Review labels if the chunking pipeline changed."
            )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    chunks = load_chunks(args.corpus)
    retriever = BM25Retriever(chunks)
    questions = load_questions(args.questions, {c["chunk_id"] for c in chunks})

    rows = []
    for question in questions:
        hits = retriever.search(
            question["query"],
            top_k=args.top_k,
            document=question.get("document"),
        )
        ranked_ids = [hit["chunk"]["chunk_id"] for hit in hits]
        metrics = score_ranking(ranked_ids, question["relevant_chunk_ids"], args.top_k)
        row = {
            "id": question["id"],
            "query": question["query"],
            "document": question.get("document"),
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "retrieved_chunk_ids": ranked_ids,
            **metrics,
        }
        rows.append(row)

        print(f"\n{question['id']} — {question['query']}")
        print(f"  Top-{args.top_k}: {ranked_ids}")
        print(f"  First relevant rank: {metrics['first_relevant_rank']}")
        print(f"  Hit@{args.top_k}: {metrics['hit_at_k']}  Recall@{args.top_k}: {metrics['recall_at_k']:.2f}")

    summary = summarize_scores(rows)
    report = {"retriever": "bm25", "k": args.top_k, "summary": summary, "questions": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== BM25 over {summary['queries']} labeled questions ===")
    print(f"Hit@{args.top_k}: {summary['hit_rate_at_k']:.3f}")
    print(f"Mean Recall@{args.top_k}: {summary['mean_recall_at_k']:.3f}")
    print(f"MRR@{args.top_k}: {summary['mrr_at_k']:.3f}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
