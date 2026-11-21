
"""Graded-relevance evaluation for a judged retrieval pool."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def load_judgments(path: Path) -> tuple[dict, dict]:
    """Read relevance judgments from the reviewed CSV."""

    judgments = {}
    questions = {}

    with Path(path).open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "question_id",
            "query",
            "expected_document",
            "chunk_id",
            "relevance",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        for row_number, row in enumerate(reader, start=2):

            qid = (row["question_id"] or "").strip()
            chunk_id = (row["chunk_id"] or "").strip()
            query = (row["query"] or "").strip()

            document = (
                (row["expected_document"] or "").strip()
                or None
            )

            relevance = (row["relevance"] or "").strip()

            if not qid or not chunk_id or not query:
                raise ValueError(
                    f"Missing ID/query on row {row_number}"
                )

            if relevance not in {"0", "1", "2"}:
                raise ValueError(
                    f"Invalid relevance on row {row_number}"
                )

            metadata = {
                "query": query,
                "document": document,
            }

            if qid in questions and questions[qid] != metadata:
                raise ValueError(
                    f"Inconsistent question metadata: {qid}"
                )

            questions[qid] = metadata

            grades = judgments.setdefault(qid, {})

            if chunk_id in grades:
                raise ValueError(
                    f"Duplicate judgment: {qid}/{chunk_id}"
                )

            pending = (
                (row.get("verificar_pdf") or "")
                .strip()
                .casefold()
                in {"sim", "yes", "true", "1"}
            )

            grades[chunk_id] = {
                "grade": int(relevance),
                "verify_pdf": pending,
            }

    if not judgments:
        raise ValueError("No judgments found")

    return judgments, questions


def discounted_gain(grades: list[int]) -> float:
    """Compute DCG with gain = 2^relevance - 1."""

    return sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def score_graded(
    ranked_ids: list[str],
    judgments: dict,
    k: int = 5,
) -> dict:

    if k < 1:
        raise ValueError("k must be >= 1")

    top_ids = ranked_ids[:k]

    if len(top_ids) != len(set(top_ids)):
        raise ValueError("Duplicate chunk IDs in ranking")

    # Never treat missing judgments as irrelevant.
    unknown = [
        chunk_id
        for chunk_id in top_ids
        if chunk_id not in judgments
    ]

    if unknown:
        raise ValueError(
            "Unjudged top-k chunks: "
            + ", ".join(unknown)
            + ". Extend the judgment pool first."
        )

    grades = [
        judgments[chunk_id]["grade"]
        for chunk_id in top_ids
    ]

    ideal = sorted(
        [
            entry["grade"]
            for entry in judgments.values()
        ],
        reverse=True,
    )[:k]

    ideal_gain = discounted_gain(ideal)

    ndcg = (
        discounted_gain(grades) / ideal_gain
        if ideal_gain
        else 0.0
    )

    first_direct = next(
        (
            rank
            for rank, grade in enumerate(
                grades,
                start=1,
            )
            if grade == 2
        ),
        None,
    )

    direct_hits = sum(
        grade == 2 for grade in grades
    )

    direct_in_pool = sum(
        entry["grade"] == 2
        for entry in judgments.values()
    )

    return {
        "pooled_ndcg_at_k": ndcg,
        "hit_grade2_at_k": int(
            first_direct is not None
        ),
        "mrr_grade2_at_k": (
            1 / first_direct
            if first_direct
            else 0.0
        ),
        "precision_grade2_at_k": direct_hits / k,
        "pooled_recall_grade2_at_k": (
            direct_hits / direct_in_pool
            if direct_in_pool
            else 0.0
        ),
        "first_grade2_rank": first_direct,
        "grades_at_k": grades,
        "judged_hits_at_k": len(top_ids),
        "pdf_checks_in_top_k": sum(
            judgments[chunk_id]["verify_pdf"]
            for chunk_id in top_ids
        ),
    }


def summarize_graded(rows: list[dict]) -> dict:
    """Compute macro averages across questions."""

    if not rows:
        raise ValueError("No scored questions")

    keys = [
        "pooled_ndcg_at_k",
        "hit_grade2_at_k",
        "mrr_grade2_at_k",
        "precision_grade2_at_k",
        "pooled_recall_grade2_at_k",
    ]

    return {
        "questions": len(rows),
        **{
            key: sum(
                row[key] for row in rows
            ) / len(rows)
            for key in keys
        },
    }
