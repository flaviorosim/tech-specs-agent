"""Regression tests: source-cell path must not be rejected by the local LLM."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tech_specs.local_answer import validate_citations
from tech_specs.source_cell_check import check_source_cell_answer
from tech_specs.table_answers import answer_from_tables

WDT_SOURCE = '''|WDTON|Safety<br>Level|WDT Initial<br>State|How to Disable the<br>WDT|How to Change Time-<br>out|
|---|---|---|---|---|
|Unprogrammed|1|Disabled|Timed sequence|No limitations|
|Programmed|2|Enabled|Always enabled|Timed sequence|
'''
STOP_SOURCE = '''|Interconnect source|Interconnect<br>destination|Interconnect action|Run|Sleep|Low-power run|Low-power sleep|Stop 0 / Stop 1|Stop 2|
|---|---|---|---|---|---|---|---|---|
|TIMx|TIMx|Timers synchronization|Y|Y|Y|Y|-|-|
|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|
|GPIO|ADCx<br>DAC1|Conversion external trigger|Y|Y|Y|Y|-|-|
'''
WDT_Q = "How does programming the WDTON fuse change the watchdog initial state and disable behavior on ATtiny25/45/85?"
STOP_Q = "Can STM32L432xx timers trigger ADC/DAC conversions in Low-power Run and Low-power Sleep, and is that connection supported in Stop 2?"


def make_report(question=WDT_Q, source=WDT_SOURCE, cid="attiny85_p043_c002", facets=("watchdog",)):
    hit = {"rank": 1, "facet_matches": {fid: {"rank": 1} for fid in facets},
           "chunk": {"chunk_id": cid, "document": "attiny85" if cid.startswith("attiny") else "stm32l432kc",
                     "page": 43 if cid.startswith("attiny") else 28, "content": source}}
    hits = [hit]
    extraction = answer_from_tables(question, hits)
    assert extraction is not None
    citations = validate_citations(extraction.draft, hits, facet_ids=list(facets))
    assert citations["accepted"] is True
    return {"question": question, "status": "citations_checked", "draft": extraction.draft,
            "plan": {"facets": [{"id": fid} for fid in facets]},
            "retrieval": {"hits": hits}, "citation_check": citations}


def test_wdton_exact_table_extraction_accepted():
    report = make_report()
    check = check_source_cell_answer(report)
    assert check["applicable"] and check["accepted"]
    assert check["kind"] == "configuration_table" and len(check["cells"]) == 2
    assert check["cells"][1]["how_to_disable"] == "Always enabled"
    assert len(check["source_sha256"]) == 64


def test_stop2_matrix_cells_accepted():
    report = make_report(STOP_Q, STOP_SOURCE, "stm32l432kc_p028_c005", ("timer",))
    check = check_source_cell_answer(report)
    assert check["accepted"] and check["kind"] == "interconnect_matrix"
    assert [c["cell"] for c in check["cells"]] == ["Y", "Y", "-"]
    assert "not supported in Stop 2" in report["draft"]


def test_modified_source_is_used_not_benchmark_key():
    source = STOP_SOURCE.replace("|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|",
                                 "|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|Y|")
    report = make_report(STOP_Q, source, "stm32l432kc_p028_c005", ("timer",))
    assert check_source_cell_answer(report)["accepted"]
    assert "is supported in Stop 2" in report["draft"]


def test_mutated_draft_cannot_bypass_judge():
    report = make_report()
    report["draft"] = report["draft"].replace("Always enabled", "Never enabled")
    check = check_source_cell_answer(report)
    assert check["applicable"] and not check["accepted"]
    assert check["reason"] == "draft_differs_from_exact_source_cells"


def test_missing_plan_or_facet_membership_cannot_pass():
    report = make_report()
    report["plan"]["facets"] = [{"id": "madeup"}]
    check = check_source_cell_answer(report)
    assert not check["accepted"]
    assert check["reason"] == "citations_or_facet_coverage_invalid"


def test_tampered_citation_rejected():
    report = make_report()
    report["citation_check"]["cited_chunk_ids"] = ["invented"]
    assert check_source_cell_answer(report)["reason"] == "stored_citations_do_not_match"


def test_original_citation_gate_required():
    report = make_report()
    report["citation_check"]["accepted"] = False
    assert not check_source_cell_answer(report)["accepted"]


def test_source_change_does_not_revalidate_old_draft():
    report = make_report()
    report["retrieval"]["hits"][0]["chunk"]["content"] = report["retrieval"]["hits"][0]["chunk"]["content"].replace(
        "Always enabled", "Cannot be enabled"
    )
    assert not check_source_cell_answer(report)["accepted"]


def test_ambiguous_conflicting_tables_do_not_pass():
    report = make_report(STOP_Q, STOP_SOURCE, "stm32l432kc_p028_c005", ("timer",))
    other = deepcopy(report["retrieval"]["hits"][0])
    other["chunk"]["chunk_id"] = "stm32l432kc_p029_c007"
    other["chunk"]["content"] = other["chunk"]["content"].replace(
        "|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|",
        "|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|Y|"
    )
    report["retrieval"]["hits"].append(other)
    assert not check_source_cell_answer(report)["accepted"]


def test_normal_llm_answer_has_no_shortcut():
    report = make_report()
    report["question"] = "What is the TIM2 resolution?"
    report["draft"] = "TIM2 is 32-bit. [[attiny85_p043_c002]]"
    assert not check_source_cell_answer(report)["accepted"]


def test_rejected_generation_never_gets_shortcut():
    report = make_report()
    report["status"] = "citation_check_failed"
    assert not check_source_cell_answer(report)["accepted"]


class ExplodingJudge:
    def judge(self, *args, **kwargs):
        raise AssertionError("source-derived table answer should not call an LLM")
    def check(self, *args, **kwargs):
        raise AssertionError("source-derived table answer should not call an LLM")
    def repair(self, *args, **kwargs):
        raise AssertionError("source-derived table answer should not call an LLM")


class Base:
    def __init__(self, report):
        self.report = report
    def invoke(self, payload):
        assert payload["question"] == self.report["question"]
        return deepcopy(self.report)


@pytest.mark.parametrize("question,source,cid,facets", [
    (WDT_Q, WDT_SOURCE, "attiny85_p043_c002", ("watchdog",)),
    (STOP_Q, STOP_SOURCE, "stm32l432kc_p028_c005", ("timer",)),
])
def test_graph_uses_source_cells_without_llm(question, source, cid, facets):
    pytest.importorskip("langgraph")
    from tech_specs.verified_answer_graph import build_verified_answer_graph
    graph = build_verified_answer_graph(
        Base(make_report(question, source, cid, facets)),
        judge=ExplodingJudge(), coverage_judge=ExplodingJudge(),
        repairer=ExplodingJudge(),
    )
    result = graph.invoke({"question": question})
    assert result["status"] == "source_table_checked"
    assert result["repairs"] == 0
    assert result["audits"] == result["coverage_checks"] == []
    assert result["source_cell_check"]["accepted"]
    assert "SOURCE TABLE CELLS CHECKED" in result["report"]


def test_graph_rejects_fabricated_table_text_even_with_valid_citation():
    pytest.importorskip("langgraph")
    from tech_specs.verified_answer_graph import build_verified_answer_graph
    report = make_report()
    report["draft"] = report["draft"].replace("Always enabled", "Never enabled")
    graph = build_verified_answer_graph(Base(report), judge=ExplodingJudge(),
                                        coverage_judge=ExplodingJudge(), repairer=ExplodingJudge(),
                                        max_repairs=0)
    with pytest.raises(AssertionError, match="source-derived table answer should not call an LLM"):
        graph.invoke({"question": WDT_Q})
