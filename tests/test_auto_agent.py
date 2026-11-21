"""Automatic planner and first LangGraph retrieval workflow tests."""

import pytest

from tech_specs.auto_planner import detect_document, make_plan
from tech_specs.facet_retrieval import anchor_coverage


QUESTION = (
    "How do the ADC and Brown-out Detector contribute to "
    "ATtiny25/45/85 sleep-mode power consumption, "
    "and what should be disabled if they are not needed?"
)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5, document=None):
        self.calls.append((query, top_k, document))
        if "Brown-out Detector" in query:
            cid = "attiny85_p036_c006"
        elif "ADC" in query:
            cid = "attiny85_p036_c004"
        else:
            return []
        return [{
            "rank": 1,
            "score": 1.0,
            "chunk": {"chunk_id": cid, "document": "attiny85", "page": 36,
                      "content": f"Evidence for {cid}"},
        }]


def test_automatic_plan_detects_both_facets():
    plan = make_plan(QUESTION)
    assert plan["document"] == "attiny85"
    assert [f["id"] for f in plan["facets"]] == ["adc", "bod"]
    assert "Brown-out Detector" not in plan["facets"][0]["query"]
    assert "ADC" not in plan["facets"][1]["query"]


def test_known_single_topic_and_unknown_topic_use_original_query():
    single = "What is the TIM2 counter resolution on STM32L432KC?"
    assert make_plan(single)["facets"] == [{"id": "timer", "query": single}]
    general = "How do I calculate my checksum?"
    assert make_plan(general)["facets"] == [{"id": "general", "query": general}]


def test_device_ambiguity_and_override():
    assert detect_document("What is the ADC resolution?", "attiny85") == "attiny85"
    with pytest.raises(ValueError, match="conflicts"):
        detect_document("STM32L432KC ADC", "attiny85")
    with pytest.raises(ValueError, match="both device families"):
        detect_document("Compare STM32L432KC and ATtiny85 ADC")
    with pytest.raises(ValueError, match="Unknown document"):
        make_plan("ADC?", "not-a-device")


def test_facets_limited_and_query_validation():
    with pytest.raises(ValueError, match="refusing to silently discard"):
        make_plan("ADC BOD watchdog ATtiny85 sleep", max_facets=2)
    with pytest.raises(ValueError):
        make_plan("  ")
    with pytest.raises(ValueError):
        make_plan("ADC BOD", max_facets=0)


def test_langgraph_runs_automatic_plan_retrieval_and_report():
    pytest.importorskip("langgraph")
    from tech_specs.agent_graph import build_agent_graph

    fake = FakeRetriever()
    graph = build_agent_graph(fake, top_k=6, candidates_per_facet=10)
    state = graph.invoke({"question": QUESTION})
    assert [h["chunk"]["chunk_id"] for h in state["retrieval"]["hits"]] == [
        "attiny85_p036_c004", "attiny85_p036_c006",
    ]
    assert state["assessment"]["facets_with_candidates"] == 2
    assert "NOT a verified technical answer" in state["report"]
    assert len(fake.calls) == 2
    assert all(document == "attiny85" for _, _, document in fake.calls)
    assert anchor_coverage(state["retrieval"], [
        "attiny85_p036_c004", "attiny85_p036_c006",
    ])["fraction"] == 1.0


def test_langgraph_missing_evidence_not_claimed_as_complete():
    pytest.importorskip("langgraph")
    from tech_specs.agent_graph import build_agent_graph

    class OnlyADC(FakeRetriever):
        def search(self, query, top_k=5, document=None):
            if "Brown-out Detector" in query:
                return []
            return super().search(query, top_k=top_k, document=document)

    state = build_agent_graph(OnlyADC()).invoke({"question": QUESTION})
    assert state["assessment"]["facets_without_candidates"] == ["bod"]
    assert "Candidate" not in state["assessment"]
    assert "1/2" in state["report"]


def test_langgraph_without_any_candidates():
    pytest.importorskip("langgraph")
    from tech_specs.agent_graph import build_agent_graph

    class Empty:
        def search(self, query, top_k=5, document=None):
            return []

    state = build_agent_graph(Empty()).invoke({"question": "UART error on STM32L432KC?"})
    assert state["assessment"]["has_any_candidates"] is False
    assert "do not answer" in state["report"]
