"""Facet retrieval tests need no Qdrant or embedding model."""

import pytest

from tech_specs.facet_retrieval import (
    anchor_coverage,
    search_facets,
    validate_plan,
)


def make_hit(cid, score=1.0):
    return {
        "chunk": {
            "chunk_id": cid,
            "document": "attiny85",
            "page": 36,
            "content": cid,
        },
        "score": score,
    }


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5, document=None):
        self.calls.append((query, top_k, document))
        return self.results.get(query, [])[:top_k]


PLAN = {
    "question": "What can be disabled?",
    "document": "attiny85",
    "facets": [
        {"id": "adc", "query": "adc-query"},
        {"id": "bod", "query": "bod-query"},
    ],
}


def test_round_robin_reserves_evidence_for_both_facets():
    stub = StubRetriever({
        "adc-query": [make_hit("a1"), make_hit("a2"), make_hit("a3")],
        "bod-query": [make_hit("b1"), make_hit("b2"), make_hit("b3")],
    })
    result = search_facets(stub, PLAN, top_k=4, candidates_per_facet=3)
    assert [h["chunk"]["chunk_id"] for h in result["hits"]] == [
        "a1", "b1", "a2", "b2",
    ]
    assert stub.calls == [
        ("adc-query", 3, "attiny85"),
        ("bod-query", 3, "attiny85"),
    ]
    assert [f["selected_matches"] for f in result["facets"]] == [2, 2]


def test_deduplicate_chunks_and_preserve_all_facet_matches():
    stub = StubRetriever({
        "adc-query": [make_hit("shared"), make_hit("adc-only")],
        "bod-query": [make_hit("shared"), make_hit("bod-only")],
    })
    result = search_facets(stub, PLAN, top_k=3, candidates_per_facet=2)
    assert [h["chunk"]["chunk_id"] for h in result["hits"]] == [
        "shared", "adc-only", "bod-only",
    ]
    assert set(result["hits"][0]["facet_matches"]) == {"adc", "bod"}


def test_missing_candidates_does_not_fake_coverage():
    stub = StubRetriever({"adc-query": [make_hit("adc-only")]})
    result = search_facets(stub, PLAN)
    assert result["facets"][1]["candidates"] == 0
    coverage = anchor_coverage(result, ["adc-only", "bod-anchor"])
    assert coverage["fraction"] == 0.5
    assert coverage["missing"] == ["bod-anchor"]


def test_plan_validation():
    with pytest.raises(ValueError, match="Duplicate facet"):
        validate_plan({
            "question": "x", "facets": [
                {"id": "a", "query": "x"},
                {"id": "a", "query": "y"},
            ],
        })
    with pytest.raises(ValueError, match="nonempty 'query'"):
        validate_plan({
            "question": "x", "facets": [{"id": "a", "query": " "}],
        })
    with pytest.raises(ValueError, match="nonempty 'facets'"):
        validate_plan({"question": "x", "facets": []})


def test_validation_of_depth_and_reference_ids():
    with pytest.raises(ValueError, match="must be >= 1"):
        search_facets(StubRetriever({}), PLAN, top_k=0)
    with pytest.raises(ValueError, match="Duplicate reference"):
        anchor_coverage({"hits": []}, ["x", "x"])
