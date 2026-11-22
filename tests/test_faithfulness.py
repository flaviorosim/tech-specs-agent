"""Offline tests: no Ollama or LangGraph required."""

import json
from unittest.mock import patch

import pytest

from tech_specs.faithfulness import (
    Claim, OllamaEvidenceJudge, audit_report, evaluate_judge_result,
    get_claims, quote_matches,
)


def fixture_report():
    a = {
        "chunk_id": "stm32l432kc_p038_c004", "document": "stm32l432kc",
        "page": 38, "content": "Timer TIM2 has 32-bit counter resolution in Table 9.",
    }
    b = {
        "chunk_id": "stm32l432kc_p040_c002", "document": "stm32l432kc",
        "page": 40, "content": "TIM2 is a 32-bit general-purpose timer.",
    }
    return {
        "status": "citations_checked",
        "citation_check": {"accepted": True},
        "question": "What is TIM2 counter resolution on STM32L432KC?",
        "draft": "TIM2 has 32-bit counter resolution. [[stm32l432kc_p038_c004]]",
        "retrieval": {"hits": [{"chunk": a}, {"chunk": b}]},
    }


def test_parse_claim_and_source_ids():
    claims = get_claims(fixture_report())
    assert claims == [Claim("TIM2 has 32-bit counter resolution.", ("stm32l432kc_p038_c004",))]


def test_parse_two_claims_and_missing_information():
    r = fixture_report()
    r["draft"] = (
        "TIM2 is 32-bit [[stm32l432kc_p040_c002]]\n\n"
        "TIM2 has 32-bit resolution [[stm32l432kc_p038_c004]]\n\n"
        "Information not established by the supplied evidence: Nothing else"
    )
    assert len(get_claims(r)) == 2


def test_uncited_claim_is_not_discarded_silently():
    r = fixture_report()
    r["draft"] += "\n\nTIM2 has 8-bit resolution."
    with pytest.raises(ValueError, match="uncited"):
        get_claims(r)


def test_reject_unchecked_source_or_report():
    r = fixture_report()
    r["status"] = "citation_check_failed"
    with pytest.raises(ValueError, match="citation"):
        get_claims(r)
    r = fixture_report()
    r["draft"] = "TIM2 has 32-bit counter resolution. [[fake]]"
    with pytest.raises(ValueError, match="absent"):
        get_claims(r)


def test_quote_matching_with_linebreak_hyphenation():
    assert quote_matches("Brown-out Detector", "Brown-out Detec-\ntor is present")
    assert not quote_matches("TIM2 has 64-bit", "TIM2 has 32-bit resolution")
    assert not quote_matches("TIM2", "TIM2")  # Too short to be meaningful


def test_verified_verbatim_quote_with_supported_verdict():
    c = get_claims(fixture_report())[0]
    chunk = fixture_report()["retrieval"]["hits"][0]["chunk"]
    result = evaluate_judge_result(
        {"verdict": "supported", "reason": "32 bits explicitly stated", "quotes": [
            {"source_id": c.source_ids[0], "quote": "TIM2 has 32-bit counter resolution"}
        ]}, c, {c.source_ids[0]: chunk}
    )
    assert result["verdict"] == "supported"
    assert result["quote_checks"][0]["verbatim_found"]


def test_no_quote_or_invented_quote_cannot_pass():
    c = get_claims(fixture_report())[0]
    chunk = fixture_report()["retrieval"]["hits"][0]["chunk"]
    sources = {c.source_ids[0]: chunk}
    for quotes in ([], [{"source_id": c.source_ids[0], "quote": "TIM2 has 64-bit counter resolution"}]):
        result = evaluate_judge_result(
            {"verdict": "supported", "reason": "guess", "quotes": quotes}, c, sources
        )
        assert result["verdict"] == "uncertain"
        assert result["issue"]


def test_judge_unsupported_is_preserved():
    c = Claim("TIM2 has 64-bit resolution", ("stm32l432kc_p038_c004",))
    result = evaluate_judge_result(
        {"verdict": "unsupported", "reason": "Source says 32-bit", "quotes": []},
        c, {c.source_ids[0]: {"content": "TIM2 has 32-bit counter resolution"}},
    )
    assert result["verdict"] == "unsupported"


def test_mocked_ollama_request_and_end_to_end_audit():
    report = fixture_report()
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            raw = {"verdict": "supported", "reason": "explicitly stated", "quotes": [
                {"source_id": "stm32l432kc_p038_c004", "quote": "TIM2 has 32-bit counter resolution"}
            ]}
            return json.dumps({"message": {"content": json.dumps(raw)}, "done_reason": "stop"}).encode()
    def fake_open(req, timeout):
        assert req.full_url == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(req.data)
        assert payload["stream"] is False and payload["think"] is False
        assert payload["model"] == "qwen3:4b"
        assert payload["format"]["properties"]["quotes"]["items"]["properties"]["source_id"]["enum"] == ["stm32l432kc_p038_c004"]
        return FakeResponse()

    with patch("tech_specs.faithfulness.urlopen", side_effect=fake_open):
        audit = audit_report(report, OllamaEvidenceJudge())
    assert audit["audit_status"] == "all_supported_by_local_judge"
    assert audit["counts"]["supported"] == 1


def test_uncertain_audit_flags_review():
    class FakeJudge:
        def judge(self, question, claim, source_by_id):
            return {"verdict": "uncertain", "judge_verdict": "uncertain",
                    "reason": "not enough context", "quote_checks": [], "issue": ""}
    audit = audit_report(fixture_report(), FakeJudge())
    assert audit["audit_status"] == "review_required"
