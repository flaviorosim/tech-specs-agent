"""Offline checks: prompt assembly, citation checking, Ollama request mocking, graph."""

import json
from unittest.mock import patch

import pytest

from tech_specs.local_answer import (
    LocalOllama, ModelAbstained, citation_sources, make_grounding_prompt,
    make_answer_schema, render_structured_answer, validate_citations,
)


QUESTION = (
    "How do the ADC and Brown-out Detector contribute to "
    "ATtiny25/45/85 sleep-mode power consumption?"
)


def make_hits():
    return [
        {
            "rank": 1,
            "chunk": {
                "chunk_id": "attiny85_p036_c004", "document": "attiny85",
                "page": 36, "content": "ADC should be disabled before entering sleep.",
            },
            "facet_matches": {"adc": {"rank": 1}},
        },
        {
            "rank": 2,
            "chunk": {
                "chunk_id": "attiny85_p036_c006", "document": "attiny85",
                "page": 36, "content": "BOD continues consuming power in sleep modes.",
            },
            "facet_matches": {"bod": {"rank": 1}},
        },
    ]


def test_prompt_uses_actual_chunk_metadata():
    prompt = make_grounding_prompt(QUESTION, make_hits())
    assert 'chunk_id="attiny85_p036_c006"' in prompt
    assert 'pdf_page="36"' in prompt
    assert "BOD continues consuming" in prompt
    with pytest.raises(ValueError):
        make_grounding_prompt(QUESTION, [])


def test_citations_accept_both_facets():
    answer = (
        "ADC consumes power [[attiny85_p036_c004]]. "
        "BOD also consumes power [[attiny85_p036_c006]]."
    )
    result = validate_citations(answer, make_hits(), facet_ids=["adc", "bod"])
    assert result["accepted"] is True
    assert len(result["cited_chunk_ids"]) == 2
    assert citation_sources(make_hits(), result["cited_chunk_ids"])[1]["pdf_page"] == 36


def test_citations_reject_fabricated_and_missing_facets():
    unknown = validate_citations("Wrong [[attiny85_p999_c001]]", make_hits())
    assert not unknown["accepted"]
    missing = validate_citations(
        "Only ADC [[attiny85_p036_c004]]", make_hits(), facet_ids=["adc", "bod"]
    )
    assert missing["missing_cited_facets"] == ["bod"]
    assert not validate_citations("No cites", make_hits())["accepted"]


def test_ollama_payload_is_local_and_nonstreaming():
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            content = json.dumps({"claims": [{"text": "Disable ADC before sleep", "source_ids": ["attiny85_p036_c004"]}], "missing_information": ""})
            return json.dumps({"message": {"content": content}, "done_reason": "stop"}).encode()

    def fake_open(req, timeout):
        assert req.full_url == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(req.data)
        assert payload["model"] == "qwen3:4b"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 4096
        assert payload["think"] is False
        assert payload["format"]["properties"]["claims"]["items"]["properties"]["source_ids"]["items"]["enum"] == ["attiny85_p036_c004", "attiny85_p036_c006"]
        return FakeResponse()

    with patch("tech_specs.local_answer.urlopen", side_effect=fake_open):
        answer = LocalOllama().generate(QUESTION, make_hits())
    assert "[[attiny85_p036_c004]]" in answer


class FakeRetriever:
    def search(self, query, top_k=10, document=None):
        hits = make_hits()
        if "Brown-out Detector" in query:
            hits = hits[1:]
        elif "ADC" in query:
            hits = hits[:1]
        return [{"rank": 1, "score": 1.0, "chunk": h["chunk"]} for h in hits]


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
    def generate(self, question, hits):
        self.calls += 1
        return self.response


def test_graph_valid_answer_and_citations():
    pytest.importorskip("langgraph")
    from tech_specs.answer_graph import build_answer_graph
    llm = FakeLLM(
        "ADC should be disabled [[attiny85_p036_c004]]. "
        "BOD draws power [[attiny85_p036_c006]]."
    )
    result = build_answer_graph(FakeRetriever(), llm).invoke({"question": QUESTION})
    assert result["status"] == "citations_checked"
    assert result["citation_check"]["accepted"]
    assert llm.calls == 1
    assert "PDF page 36" in result["report"]


def test_graph_withholds_answer_with_invalid_citation():
    pytest.importorskip("langgraph")
    from tech_specs.answer_graph import build_answer_graph
    llm = FakeLLM("I made it up [[fake_source]]")
    result = build_answer_graph(FakeRetriever(), llm).invoke({"question": QUESTION})
    assert result["status"] == "citation_check_failed"
    assert "I made it up" not in result["report"]


def test_graph_does_not_call_model_when_no_evidence():
    pytest.importorskip("langgraph")
    from tech_specs.answer_graph import build_answer_graph
    class Empty:
        def search(self, query, top_k=10, document=None):
            return []
    llm = FakeLLM("Should not be called")
    result = build_answer_graph(Empty(), llm).invoke({"question": QUESTION})
    assert result["status"] == "no_evidence"
    assert llm.calls == 0


def test_structured_answer_renders_claim_level_citations():
    raw = {
        "claims": [
            {"text": "ADC consumes energy in sleep", "source_ids": ["attiny85_p036_c004"]},
            {"text": "BOD consumes energy in sleep", "source_ids": ["attiny85_p036_c006"]},
        ],
        "missing_information": "",
    }
    answer = render_structured_answer(raw, make_hits())
    assert "ADC consumes energy in sleep [[attiny85_p036_c004]]" in answer
    assert validate_citations(answer, make_hits(), facet_ids=["adc", "bod"])["accepted"]


def test_structured_rejects_fabricated_ids_and_uncited_claims():
    with pytest.raises(ValueError, match="not among retrieved"):
        render_structured_answer(
            {"claims": [{"text": "False", "source_ids": ["made_up"]}], "missing_information": ""},
            make_hits(),
        )
    with pytest.raises(ValueError, match="needs 1 to 3"):
        render_structured_answer(
            {"claims": [{"text": "Missing", "source_ids": []}], "missing_information": ""},
            make_hits(),
        )


def test_structured_can_abstain_without_fake_citations():
    with pytest.raises(ModelAbstained, match="Not present"):
        render_structured_answer(
            {"claims": [], "missing_information": "Not present in the evidence"}, make_hits()
        )


def test_ollama_retries_when_first_result_invalid():
    responses = [
        {"claims": [{"text": "X", "source_ids": ["invented"]}], "missing_information": ""},
        {"claims": [{"text": "Good", "source_ids": ["attiny85_p036_c006"]}], "missing_information": ""},
    ]
    calls = []
    class FakeResponse:
        def __init__(self, content):
            self.content = content
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"message": {"content": json.dumps(self.content)}}).encode()
    def fake_open(req, timeout):
        calls.append(json.loads(req.data))
        return FakeResponse(responses.pop(0))
    with patch("tech_specs.local_answer.urlopen", side_effect=fake_open):
        answer = LocalOllama().generate(QUESTION, make_hits())
    assert len(calls) == 2
    assert "[[attiny85_p036_c006]]" in answer


def test_graph_model_abstains_when_no_supported_claims():
    pytest.importorskip("langgraph")
    from tech_specs.answer_graph import build_answer_graph
    class AbstainingLLM:
        def generate(self, question, hits):
            raise ModelAbstained("Not enough evidence for the detail")
    result = build_answer_graph(FakeRetriever(), AbstainingLLM()).invoke({"question": QUESTION})
    assert result["status"] == "insufficient_evidence"
    assert "Not enough evidence" in result["report"]
