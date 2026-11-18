"""Small, reproducible retrieval metrics for a labeled question set."""

from __future__ import annotations


def score_ranking(
    ranked_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> dict:
    """Compute Hit@k, Recall@k and reciprocal rank for one query."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_chunk_ids:
        raise ValueError("Each query needs at least one relevant chunk")

    relevant = set(relevant_chunk_ids)
    top_ids = ranked_chunk_ids[:k]
    retrieved_relevant = relevant.intersection(top_ids)

    first_relevant_rank = next(
        (rank for rank, chunk_id in enumerate(top_ids, start=1)
         if chunk_id in relevant),
        None,
    )

    return {
        "hit_at_k": int(first_relevant_rank is not None),
        "recall_at_k": len(retrieved_relevant) / len(relevant),
        "reciprocal_rank_at_k": (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
    }


def summarize_scores(scores: list[dict]) -> dict:
    """Return macro averages over queries (equal weight per question)."""
    if not scores:
        raise ValueError("Cannot summarize an empty evaluation")

    n = len(scores)
    return {
        "queries": n,
        "hit_rate_at_k": sum(s["hit_at_k"] for s in scores) / n,
        "mean_recall_at_k": sum(s["recall_at_k"] for s in scores) / n,
        "mrr_at_k": sum(s["reciprocal_rank_at_k"] for s in scores) / n,
    }
