"""Offline tests; Ollama is mocked, LangGraph tests run when installed."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tech_specs.review_models import OllamaCoverageJudge, OllamaRepair, validate_coverage_result
from tech_specs.local_answer import ModelAbstained

CID = "stm32l432kc_p038_c004"
QUESTION = "What is TIM2 counter resolution on STM32L432KC?"
HITS = [{"rank": 1, "chunk": {"chunk_id": CID, "document": "stm32l432kc",
                           "page": 38, "content": "TIM2 has a 32-bit counter."},
         "facet_matches": {"timer": {"rank": 1}}}]


def fake_base_result(draft="TIM2 is 64-bit. [[stm32l432kc_p038_c004]]"):
    return {
        "question": QUESTION, "status": "citations_checked", "draft": draft,
        "plan": {"facets": [{"id": "timer"}], "document": "stm32l432kc"},
        "retrieval": {"hits": HITS},
        "citation_check": {"accepted": True, "cited_chunk_ids": [CID]},
    }


class BaseGraph:
    def __init__(self, result):
        self.result, self.calls = result, 0
    def invoke(self, value):
        self.calls += 1
        assert value["question"] == QUESTION
        return self.result


class Judge:
    def judge(self, question, claim, sources):
        correct = "32-bit" in claim.text and "64-bit" not in claim.text
        return {
            "verdict": "supported" if correct else "unsupported",
            "judge_verdict": "supported" if correct else "unsupported",
            "reason": "Source has 32 bits", "quote_checks": [], "issue": "",
        }


class Coverage:
    def __init__(self, verdict="complete"):
        self.verdict = verdict
    def check(self, question, answer):
        return {"verdict": self.verdict,
                "missing_aspects": [] if self.verdict == "complete" else ["counter resolution"],
                "reason": "test"}


class Repair:
    def __init__(self, draft="TIM2 is 32-bit. [[stm32l432kc_p038_c004]]"):
        self.draft, self.calls = draft, 0
    def repair(self, question, hits, previous, feedback):
        self.calls += 1
        assert "64-bit" in previous or "coverage" in feedback.lower()
        assert ("FRESH_REANSWER" in feedback or "coverage" in feedback.lower()
                or "32 bits" in feedback)
        return self.draft


def build(base, repair=None, coverage=None, max_repairs=1):
    pytest.importorskip("langgraph")
    from tech_specs.verified_answer_graph import build_verified_answer_graph
    return build_verified_answer_graph(
        base, Judge(), coverage or Coverage(), repair or Repair(),
        max_repairs=max_repairs,
    )


def test_repair_factual_contradiction_and_keep_original():
    repair = Repair()
    graph = build(BaseGraph(fake_base_result()), repair=repair)
    result = graph.invoke({"question": QUESTION})
    assert result["status"] == "local_checks_passed"
    assert result["repairs"] == 1 and repair.calls == 1
    assert "64-bit" in result["original_result"]["draft"]
    assert "32-bit" in result["report"]
    assert "64-bit" not in result["report"]
    assert len(result["audits"]) == len(result["coverage_checks"]) == 2


def test_success_first_time_requires_no_repair():
    repair = Repair()
    graph = build(BaseGraph(fake_base_result("TIM2 is 32-bit. [[stm32l432kc_p038_c004]]")), repair=repair)
    result = graph.invoke({"question": QUESTION})
    assert result["status"] == "local_checks_passed"
    assert repair.calls == 0 and result["repairs"] == 0


def test_still_wrong_after_one_repair_is_withheld():
    repair = Repair(draft="TIM2 is 64-bit. [[stm32l432kc_p038_c004]]")
    graph = build(BaseGraph(fake_base_result()), repair=repair)
    result = graph.invoke({"question": QUESTION})
    assert result["status"] == "review_required"
    assert "64-bit" not in result["report"]
    assert result["repairs"] == 1
    assert len(result["audits"]) == 2


def test_unknown_citation_on_repair_is_withheld():
    repair = Repair(draft="TIM2 is 32-bit. [[invented_source]]")
    result = build(BaseGraph(fake_base_result()), repair=repair).invoke({"question": QUESTION})
    assert result["status"] == "citation_check_failed"
    assert "32-bit" not in result["report"]
    assert len(result["audits"]) == 1


def test_no_evidence_never_invokes_judge_or_repair():
    base = BaseGraph({"question": QUESTION, "status": "no_evidence",
                      "report": "No evidence", "draft": ""})
    result = build(base).invoke({"question": QUESTION})
    assert result["status"] == "no_evidence"
    assert "Answer withheld" in result["report"]
    assert result["repairs"] == 0


def test_coverage_incomplete_triggers_review_without_gold():
    repair = Repair()
    initial = fake_base_result("TIM2 is 32-bit. [[stm32l432kc_p038_c004]]")
    graph = build(BaseGraph(initial), coverage=Coverage("incomplete"), repair=repair)
    result = graph.invoke({"question": QUESTION})
    assert result["repairs"] == 1
    assert result["status"] == "coverage_review_recommended"
    assert "32-bit" in result["report"]
    assert "COVERAGE REVIEW RECOMMENDED" in result["report"]


def test_no_repair_mode_blocks_unsupported_claim():
    repair = Repair()
    graph = build(BaseGraph(fake_base_result()), repair=repair, max_repairs=0)
    result = graph.invoke({"question": QUESTION})
    assert result["repairs"] == 0 and repair.calls == 0
    assert result["status"] == "review_required"


def test_coverage_validation_fails_closed():
    assert validate_coverage_result({
        "verdict": "complete", "missing_aspects": ["Stop 2"], "reason": "contradiction",
    })["verdict"] == "uncertain"
    with pytest.raises(ValueError):
        validate_coverage_result({"verdict": "wrong", "missing_aspects": [], "reason": ""})


def test_coverage_ollama_payload_mocked():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            body = {"verdict": "incomplete", "missing_aspects": ["Stop 2"], "reason": "Missing mode"}
            return json.dumps({"message": {"content": json.dumps(body)}}).encode()
    def fake_open(req, timeout):
        assert req.full_url == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(req.data)
        assert payload["stream"] is False and payload["think"] is False
        assert payload["format"]["properties"]["verdict"]["enum"] == ["complete", "incomplete", "uncertain"]
        return Response()
    with patch("tech_specs.review_models.urlopen", side_effect=fake_open):
        result = OllamaCoverageJudge().check(QUESTION, "TIM2 is 32-bit")
    assert result["verdict"] == "incomplete"


def test_repair_ollama_source_ids_constrained():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            answer = {"claims": [{"text": "TIM2 has 32-bit resolution", "source_ids": [CID]}],
                      "missing_information": ""}
            return json.dumps({"message": {"content": json.dumps(answer)}}).encode()
    def fake_open(req, timeout):
        payload = json.loads(req.data)
        assert payload["format"]["properties"]["claims"]["items"]["properties"]["source_ids"]["items"]["enum"] == [CID]
        assert "PREVIOUS ANSWER" in payload["messages"][1]["content"]
        return Response()
    with patch("tech_specs.review_models.urlopen", side_effect=fake_open):
        text = OllamaRepair().repair(QUESTION, HITS, "TIM2 64-bit", "Source has 32 bits")
    assert text == f"TIM2 has 32-bit resolution [[{CID}]]"


def test_repair_rejects_invented_source():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            answer = {"claims": [{"text": "Wrong", "source_ids": ["fake"]}], "missing_information": ""}
            return json.dumps({"message": {"content": json.dumps(answer)}}).encode()
    with patch("tech_specs.review_models.urlopen", return_value=Response()):
        with pytest.raises(ValueError, match="not among retrieved"):
            OllamaRepair().repair(QUESTION, HITS, "Wrong", "Fix")


# Regression tests for the failures observed in the first live run.

def test_coverage_verdict_cannot_accept_explicit_missing_information():
    text = (
        "TIMx can trigger ADC/DAC in Low-power Run. [[stm32l432kc_p028_c005]]\n\n"
        "Information not established by the supplied evidence: "
        "whether the connection is supported in Stop 2."
    )
    checked = validate_coverage_result({
        "verdict": "complete", "missing_aspects": [], "reason": "all done"
    }, answer=text)
    assert checked["verdict"] == "incomplete"
    assert checked["missing_aspects"]


def test_unsupported_claim_repair_feedback_excludes_wrong_claim_text():
    pytest.importorskip("langgraph")
    from tech_specs.verified_answer_graph import make_repair_feedback
    feedback = make_repair_feedback({"claims": [
        {"claim": "Programming WDTON disables the watchdog by always enabling it",
         "verdict": "unsupported", "reason": "wrong"},
    ]}, {"verdict": "complete", "missing_aspects": [], "reason": ""})
    assert "FRESH_REANSWER" in feedback
    assert "disables the watchdog by always enabling" not in feedback


def test_fresh_repair_omits_bad_previous_answer_from_ollama_prompt():
    from tech_specs.review_models import OllamaRepair, FRESH_REPAIR_SIGNAL

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            answer = {"claims": [{"text": "TIM2 has 32-bit resolution", "source_ids": [CID]}],
                      "missing_information": ""}
            return json.dumps({"message": {"content": json.dumps(answer)}}).encode()

    def fake_open(req, timeout):
        payload = json.loads(req.data)
        prompt = payload["messages"][1]["content"]
        assert "PREVIOUS ANSWER" not in prompt
        assert "TIM2 has 64-bit" not in prompt
        assert "Re-read the evidence" in prompt
        return Response()

    with patch("tech_specs.review_models.urlopen", side_effect=fake_open):
        answer = OllamaRepair().repair(QUESTION, HITS, "TIM2 has 64-bit", FRESH_REPAIR_SIGNAL)
    assert "32-bit" in answer


def test_coverage_ollama_receives_table_evidence_and_strict_prompt():
    from tech_specs.review_models import OllamaCoverageJudge

    table_hit = [{"chunk": {
        "chunk_id": "stm32l432kc_p028_c005", "page": 28, "document": "stm32l432kc",
        "content": "|TIMx|ADCx DAC1|Conversion triggers|Y|Y|Y|Y|-|-|",
    }}]

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            data = {"verdict": "incomplete", "missing_aspects": ["Stop 2"],
                    "reason": "No explicit answer for Stop 2"}
            return json.dumps({"message": {"content": json.dumps(data)}}).encode()

    def fake_open(req, timeout):
        payload = json.loads(req.data)
        prompt = payload["messages"][1]["content"]
        assert "|TIMx|ADCx DAC1|Conversion triggers|" in prompt
        assert "counter resolution" in payload["messages"][0]["content"]
        return Response()

    with patch("tech_specs.review_models.urlopen", side_effect=fake_open):
        result = OllamaCoverageJudge().check(
            "Is this trigger available in Stop 2?", "The trigger is supported in Low-power Run.",
            hits=table_hit,
        )
    assert result["verdict"] == "incomplete"


def test_graph_rejects_unresolved_stop2_even_if_coverage_judge_says_complete():
    pytest.importorskip("langgraph")
    from tech_specs.verified_answer_graph import build_verified_answer_graph

    bad = fake_base_result(
        "TIM2 is 32-bit. [[stm32l432kc_p038_c004]]\n\n"
        "Information not established by the supplied evidence: Stop 2"
    )
    class NoopRepair:
        def repair(self, question, hits, previous, feedback):
            return previous
    result = build_verified_answer_graph(
        BaseGraph(bad), Judge(), Coverage("complete"), NoopRepair(), max_repairs=1,
    ).invoke({"question": QUESTION})
    assert result["status"] == "review_required"
    assert result["repairs"] == 1


def test_graph_coverage_can_receive_retrieved_evidence():
    pytest.importorskip("langgraph")
    from tech_specs.verified_answer_graph import build_verified_answer_graph
    class EvidenceCoverage:
        def check(self, question, answer, hits=None):
            assert hits is not None
            assert hits[0]["chunk"]["chunk_id"] == CID
            return {"verdict": "complete", "missing_aspects": [], "reason": "verified"}
    result = build_verified_answer_graph(
        BaseGraph(fake_base_result("TIM2 is 32-bit. [[stm32l432kc_p038_c004]]")),
        Judge(), EvidenceCoverage(), Repair(), max_repairs=1,
    ).invoke({"question": QUESTION})
    assert result["status"] == "local_checks_passed"


# v3 regression tests: noisy coverage, truncation, and table contradictions.

INTERCONNECT = """Table 5. STM32L432xx peripherals interconnect matrix
|Interconnect source|Interconnect<br>destination|Interconnect action|Run|Sleep|Low-power run|Low-power sleep|Stop 0 / Stop 1|Stop 2|
|---|---|---|---|---|---|---|---|---|
|TIMx|TIMx|Timers synchronization or chaining|Y|Y|Y|Y|-|-|
|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|
|GPIO|ADCx<br>DAC1|Conversion external trigger|Y|Y|Y|Y|-|-|
"""
STOP_CID = "stm32l432kc_p028_c005"
STOP_QUESTION = "Can STM32 timers trigger ADC/DAC in Low-power run and Stop 2?"
STOP_HITS = [{"chunk": {"chunk_id": STOP_CID, "document": "stm32l432kc", "page": 28,
                        "content": INTERCONNECT}, "facet_matches": {"timer": {"rank": 1}}}]


def stop_result(draft):
    return {"question": STOP_QUESTION, "status": "citations_checked", "draft": draft,
            "plan": {"facets": [{"id": "timer"}], "document": "stm32l432kc"},
            "retrieval": {"hits": STOP_HITS},
            "citation_check": {"accepted": True, "cited_chunk_ids": [STOP_CID]}}


def test_table_guard_detects_positive_stop2_contradiction():
    from tech_specs.table_guard import inspect_table_claims
    result = inspect_table_claims(stop_result(
        f"TIMx can trigger ADCx and DAC1 conversions in Stop 2 mode [[{STOP_CID}]]"
    ))
    assert result["checked_cells"] == 1
    assert result["passed"] is False
    assert len(result["issues"]) == 1
    assert result["issues"][0]["table_value"] == "-"
    assert result["issues"][0]["source_component"] == "TIMx"
    assert result["issues"][0]["destination_component"] == "ADCx DAC1"


def test_table_guard_accepts_negative_stop2_and_positive_lowpower():
    from tech_specs.table_guard import inspect_table_claims
    result = inspect_table_claims(stop_result(
        f"TIMx cannot trigger ADCx and DAC1 conversions in Stop 2 mode [[{STOP_CID}]]\n\n"
        f"TIMx can trigger ADCx and DAC1 conversions in Low-power run mode [[{STOP_CID}]]"
    ))
    assert result["checked_cells"] == 2
    assert result["passed"] is True


def test_table_guard_does_not_match_unrelated_row():
    from tech_specs.table_guard import inspect_table_claims
    result = inspect_table_claims(stop_result(
        f"GPIO can trigger ADCx and DAC1 conversions in Stop 2 mode [[{STOP_CID}]]"
    ))
    assert result["checked_cells"] == 1
    assert result["issues"][0]["source_component"] == "GPIO"


def test_noisy_uncertain_coverage_shows_sourced_answer_no_repair():
    repair = Repair()
    result = build(
        BaseGraph(fake_base_result("TIM2 is 32-bit. [[stm32l432kc_p038_c004]]")),
        coverage=Coverage("uncertain"), repair=repair,
    ).invoke({"question": QUESTION})
    assert result["status"] == "coverage_review_recommended"
    assert "TIM2 is 32-bit" in result["report"]
    assert result["repairs"] == repair.calls == 0
    assert result["coverage_checks"][0]["verdict"] == "uncertain"


def test_unavailable_coverage_judge_keeps_supported_claims_with_warning():
    class BrokenCoverage:
        def check(self, question, answer, hits=None):
            raise RuntimeError("Ollama output truncated")
    result = build(
        BaseGraph(fake_base_result("TIM2 is 32-bit. [[stm32l432kc_p038_c004]]")),
        coverage=BrokenCoverage(),
    ).invoke({"question": QUESTION})
    assert result["status"] == "coverage_review_recommended"
    assert "truncated" in result["coverage_checks"][0]["reason"]


def test_table_contradiction_blocks_even_if_llm_judge_says_supported():
    from tech_specs.verified_answer_graph import build_verified_answer_graph
    class EagerJudge:
        def judge(self, question, claim, sources):
            return {"verdict": "supported", "judge_verdict": "supported",
                    "reason": "supported", "quote_checks": [], "issue": ""}
    class StopGraph:
        def invoke(self, value):
            return stop_result(f"TIMx can trigger ADCx and DAC1 conversions in Stop 2 mode [[{STOP_CID}]]")
    graph = build_verified_answer_graph(StopGraph(), EagerJudge(), Coverage("complete"),
                                       Repair(), max_repairs=0)
    result = graph.invoke({"question": STOP_QUESTION})
    assert result["status"] == "review_required"
    assert "Stop 2 mode [[" not in result["report"]
    assert len(result["table_checks"][0]["issues"]) == 1


def test_table_contradiction_triggers_and_rechecks_repair():
    from tech_specs.verified_answer_graph import build_verified_answer_graph
    class EagerJudge:
        def judge(self, question, claim, sources):
            return {"verdict": "supported", "judge_verdict": "supported",
                    "reason": "supported", "quote_checks": [], "issue": ""}
    class StopGraph:
        def invoke(self, value):
            return stop_result(f"TIMx can trigger ADCx and DAC1 conversions in Stop 2 mode [[{STOP_CID}]]")
    class StopRepair:
        def repair(self, question, hits, previous, feedback):
            assert "FRESH_REANSWER" in feedback
            assert "stop 2=-" in feedback
            return f"TIMx cannot trigger ADCx and DAC1 conversions in Stop 2 mode [[{STOP_CID}]]"
    result = build_verified_answer_graph(
        StopGraph(), EagerJudge(), Coverage("complete"), StopRepair()
    ).invoke({"question": STOP_QUESTION})
    assert result["status"] == "local_checks_passed"
    assert result["repairs"] == 1
    assert result["table_checks"][0]["passed"] is False
    assert result["table_checks"][1]["passed"] is True


def test_repair_runtime_failure_is_recorded_not_raised():
    class BrokenRepair:
        def repair(self, question, hits, previous, feedback):
            raise RuntimeError("Ollama JSON truncated twice")
    result = build(BaseGraph(fake_base_result()), repair=BrokenRepair()).invoke({"question": QUESTION})
    assert result["status"] == "repair_failed"
    assert "truncated" in result["report"]
    assert "64-bit" in result["original_result"]["draft"]
    assert result["working_result"]["draft"] == ""


def test_ollama_retries_truncated_json_once():
    from tech_specs.review_models import _call_ollama_json, COVERAGE_SCHEMA
    budgets = []
    class Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(self.payload).encode()
    def fake_open(req, timeout):
        request = json.loads(req.data)
        budgets.append(request["options"]["num_predict"])
        if len(budgets) == 1:
            return Response({"done_reason": "length", "message": {"content": ""}})
        return Response({"done_reason": "stop", "message": {"content": json.dumps({
            "verdict": "complete", "missing_aspects": [], "reason": "enough"
        })}})
    with patch("tech_specs.review_models.urlopen", side_effect=fake_open):
        result = _call_ollama_json("qwen3:4b", "sys", "prompt", COVERAGE_SCHEMA,
                                   num_ctx=4096, num_predict=300, timeout=20)
    assert result["verdict"] == "complete"
    assert budgets == [300, 900]


def test_table_guard_mixed_mode_claim_in_one_sentence():
    from tech_specs.table_guard import inspect_table_claims
    result = inspect_table_claims(stop_result(
        f"TIMx can trigger ADCx and DAC1 conversions in Low-power run and Low-power sleep, "
        f"but not in Stop 2 mode [[{STOP_CID}]]"
    ))
    assert result["checked_cells"] == 3
    assert result["passed"] is True


def test_table_guard_catches_wrong_row_citation_when_true_row_was_retrieved():
    from tech_specs.table_guard import inspect_table_claims
    report = stop_result("TIMx can trigger ADCx and DAC1 conversions in Stop 2 mode [[stm32l432kc_p029_c002]]")
    report["retrieval"]["hits"].append({"chunk": {
        "chunk_id": "stm32l432kc_p029_c002", "document": "stm32l432kc", "page": 29,
        "content": "|GPIO|ADCx<br>DAC1|Conversion external trigger|Y|Y|Y|Y|-|-|"
    }})
    report["citation_check"]["cited_chunk_ids"] = ["stm32l432kc_p029_c002"]
    result = inspect_table_claims(report)
    assert not result["passed"]
    assert result["issues"][0]["source_id"] == STOP_CID
