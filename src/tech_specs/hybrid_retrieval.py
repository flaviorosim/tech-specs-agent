
"""Combine lexical and semantic search rankings using RRF."""

from __future__ import annotations

from typing import Any


def rrf_fuse(
    rankings: dict[str, list[dict]],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse ranked results without combining raw scores."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")

    grouped: dict[str, dict[str, Any]] = {}

    for method, hits in rankings.items():
        seen_in_method = set()

        for position, hit in enumerate(hits, start=1):
            chunk = hit["chunk"]
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen_in_method:
                continue

            seen_in_method.add(chunk_id)

            entry = grouped.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "score": 0.0,
                    "sources": {},
                },
            )

            entry["score"] += 1.0 / (rrf_k + position)

            entry["sources"][method] = {
                "rank": position,
                "original_score": hit["score"],
            }

    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            -item["score"],
            -len(item["sources"]),
            item["chunk"]["chunk_id"],
        ),
    )

    return [
        {
            "rank": position,
            "score": round(item["score"], 8),
            "chunk": item["chunk"],
            "sources": item["sources"],
        }
        for position, item in enumerate(
            ordered[:top_k],
            start=1,
        )
    ]


class HybridRetriever:
    """Combine BM25 and semantic search using RRF."""

    def __init__(
        self,
        bm25_retriever: Any,
        vector_retriever: Any,
        *,
        candidate_k: int = 30,
        rrf_k: int = 60,
    ) -> None:

        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")

        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")

        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        document: str | None = None,
    ) -> list[dict]:

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not query.strip():
            return []

        depth = max(self.candidate_k, top_k)

        lexical_hits = self.bm25.search(
            query,
            top_k=depth,
            document=document,
        )

        semantic_hits = self.vector.search(
            query,
            top_k=depth,
            document=document,
        )

        return rrf_fuse(
            {
                "bm25": lexical_hits,
                "vector": semantic_hits,
            },
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
