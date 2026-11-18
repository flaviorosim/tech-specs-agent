
"""Unit tests for Reciprocal Rank Fusion."""

import pytest

from tech_specs.hybrid_retrieval import (
    HybridRetriever,
    rrf_fuse,
)


def hit(chunk_id, rank, score=1.0):
    return {
        "rank": rank,
        "score": score,
        "chunk": {
            "chunk_id": chunk_id,
            "content": chunk_id,
        },
    }


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(
        self,
        query,
        top_k=5,
        document=None,
    ):
        self.calls.append(
            (query, top_k, document)
        )
        return self.hits[:top_k]


def test_rrf_rewards_agreement_between_methods():
    fused = rrf_fuse({
        "bm25": [
            hit("a", 1, 100),
            hit("shared", 2, 12),
        ],
        "vector": [
            hit("shared", 1, 0.9),
            hit("b", 2, 0.8),
        ],
    })

    assert fused[0]["chunk"]["chunk_id"] == "shared"
    assert fused[0]["sources"]["bm25"]["rank"] == 2
    assert fused[0]["sources"]["vector"]["rank"] == 1
    assert len(fused) == 3


def test_rrf_does_not_count_duplicates_twice():
    fused = rrf_fuse({
        "bm25": [
            hit("a", 1),
            hit("a", 2),
        ],
    })

    assert len(fused) == 1
    assert fused[0]["score"] == pytest.approx(1 / 61)


def test_hybrid_passes_document_filter_and_candidate_depth():
    bm25 = StubRetriever([
        hit("a", 1),
    ])

    vector = StubRetriever([
        hit("a", 1),
    ])

    searcher = HybridRetriever(
        bm25,
        vector,
        candidate_k=20,
    )

    result = searcher.search(
        "TIM2",
        document="stm32l432kc",
        top_k=3,
    )

    assert result[0]["chunk"]["chunk_id"] == "a"

    assert bm25.calls == [
        ("TIM2", 20, "stm32l432kc"),
    ]

    assert vector.calls == [
        ("TIM2", 20, "stm32l432kc"),
    ]


def test_empty_queries_and_invalid_parameters():
    bm25 = StubRetriever([])
    vector = StubRetriever([])

    searcher = HybridRetriever(bm25, vector)

    assert searcher.search("  ") == []
    assert searcher.search("no-match") == []

    with pytest.raises(ValueError):
        searcher.search("TIM2", top_k=0)

    with pytest.raises(ValueError):
        HybridRetriever(
            bm25,
            vector,
            candidate_k=0,
        )

    with pytest.raises(ValueError):
        rrf_fuse({}, rrf_k=0)
