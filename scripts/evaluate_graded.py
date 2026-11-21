
"""Evaluate BM25, vector and hybrid using graded judgments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tech_specs.graded_evaluation import (
    load_judgments,
    score_graded,
    summarize_graded,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def read_report(path: Path) -> dict:

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing report: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def run_evaluation(args):
    judgments, questions = load_judgments(
        args.judgments
    )

    paths = {
        "bm25": args.bm25,
        "vector": args.vector,
        "hybrid": args.hybrid,
    }

    results = {}
    export_rows = []

    for method, path in paths.items():

        report = read_report(path)

        reported_questions = {
            q["id"]: q
            for q in report["questions"]
        }

        if set(reported_questions) != set(questions):
            raise ValueError(
                f"Question IDs differ for {method}"
            )

        method_rows = []

        for qid in sorted(questions):

            question = reported_questions[qid]
            expected = questions[qid]

            if (
                question["query"] != expected["query"]
                or (question.get("document") or None)
                != expected["document"]
            ):
                raise ValueError(
                    f"Query/document differs: {method}/{qid}"
                )

            ids = question["retrieved_chunk_ids"]

            metrics = score_graded(
                ids,
                judgments[qid],
                args.top_k,
            )

            row = {
                "question_id": qid,
                "query": question["query"],
                "document": question.get("document"),
                "retrieved_chunk_ids": ids[:args.top_k],
                **metrics,
            }

            method_rows.append(row)

            export_rows.append({
                "method": method,
                "question_id": qid,
                "grades_top_k": " ".join(
                    map(str, metrics["grades_at_k"])
                ),
                "pooled_ndcg_at_k": round(
                    metrics["pooled_ndcg_at_k"], 6
                ),
                "hit_grade2_at_k": metrics[
                    "hit_grade2_at_k"
                ],
                "mrr_grade2_at_k": round(
                    metrics["mrr_grade2_at_k"], 6
                ),
                "pooled_recall_grade2_at_k": round(
                    metrics["pooled_recall_grade2_at_k"], 6
                ),
                "pdf_checks_in_top_k": metrics[
                    "pdf_checks_in_top_k"
                ],
            })

        results[method] = {
            "summary": summarize_graded(method_rows),
            "questions": method_rows,
        }

    pending = [
        (qid, chunk_id)
        for qid, labels in judgments.items()
        for chunk_id, item in labels.items()
        if item["verify_pdf"]
    ]

    output = {
        "metric_scope": (
            "pooled_judgments_not_exhaustive_relevance"
        ),
        "top_k": args.top_k,
        "judged_pairs": sum(
            len(value) for value in judgments.values()
        ),
        "pdf_checks_pending": pending,
        "methods": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    args.summary_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.summary_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(export_rows[0]),
        )

        writer.writeheader()
        writer.writerows(export_rows)

    print(f"Judged pairs: {output['judged_pairs']}")
    print(f"PDF checks pending: {len(pending)}")
    print(f"Top-k: {args.top_k}\n")

    for method, detail in results.items():

        summary = detail["summary"]

        print(
            f"{method:8s} "
            f"pooled nDCG@{args.top_k}: "
            f"{summary['pooled_ndcg_at_k']:.3f} | "
            f"Hit2@{args.top_k}: "
            f"{summary['hit_grade2_at_k']:.3f} | "
            f"MRR2@{args.top_k}: "
            f"{summary['mrr_grade2_at_k']:.3f}"
        )

    print(f"\nJSON: {args.output}")
    print(f"CSV: {args.summary_csv}")

    return output


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--judgments",
        type=Path,
        default=(
            ROOT / "evaluation"
            / "judgments_v2_reviewed.csv"
        ),
    )

    parser.add_argument(
        "--bm25",
        type=Path,
        default=PROCESSED / "bm25_scoped_eval.json",
    )

    parser.add_argument(
        "--vector",
        type=Path,
        default=PROCESSED / "vector_scoped_eval.json",
    )

    parser.add_argument(
        "--hybrid",
        type=Path,
        default=PROCESSED / "hybrid_scoped_eval.json",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED / "graded_eval_v2.json",
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=(
            PROCESSED / "graded_eval_v2_summary.csv"
        ),
    )

    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
