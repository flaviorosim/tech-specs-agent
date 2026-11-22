"""Source-table answer tests. No Ollama, LangGraph, or live corpus needed."""
from __future__ import annotations

from tech_specs.table_answers import TableFirstGenerator, answer_from_tables

WDT = '''Table 8-1.
WDT Configuration as a Function of the Fuse Settings of WDTON
|WDTON|Safety<br>Level|WDT Initial<br>State|How to Disable the<br>WDT|How to Change Time-<br>out|
|---|---|---|---|---|
|Unprogrammed|1|Disabled|Timed sequence|No limitations|
|Programmed|2|Enabled|Always enabled|Timed sequence|
'''
INTERCONNECT = '''Table 5. STM32L432xx peripherals interconnect matrix
|Interconnect source|Interconnect<br>destination|Interconnect action|Run|Sleep|Low-power run|Low-power sleep|Stop 0 / Stop 1|Stop 2|
|---|---|---|---|---|---|---|---|---|
|TIMx|TIMx|Timers synchronization or chaining|Y|Y|Y|Y|-|-|
|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|
|TIMx|DMA|Memory to memory transfer trigger|Y|Y|Y|Y|-|-|
|GPIO|ADCx<br>DAC1|Conversion external trigger|Y|Y|Y|Y|-|-|
'''
WDT_Q = "How does programming the WDTON fuse change the watchdog initial state and disable behavior on ATtiny25/45/85?"
STOP_Q = "Can STM32L432xx timers trigger ADC/DAC conversions in Low-power Run and Low-power Sleep, and is that connection supported in Stop 2?"
WDT_ID = "attiny85_p043_c002"
STOP_ID = "stm32l432kc_p028_c005"


def hit(cid, content):
    return {"chunk": {"chunk_id": cid, "content": content,
                      "document": "attiny85" if cid.startswith("attiny") else "stm32l432kc",
                      "page": 43 if cid.startswith("attiny") else 28}}


def test_wdt_answer_uses_both_rows_and_provenance():
    answer = answer_from_tables(WDT_Q, [hit(WDT_ID, WDT)])
    assert answer is not None
    assert answer.source_id == WDT_ID
    assert answer.kind == "configuration_table"
    assert "WDTON = Unprogrammed" in answer.draft
    assert "WDTON = Programmed" in answer.draft
    assert "Disabled" in answer.draft
    assert "Enabled" in answer.draft
    assert "Always enabled" in answer.draft
    assert answer.draft.count(f"[[{WDT_ID}]]") == 2
    assert answer.cells[1]["how_to_disable"] == "Always enabled"


def test_wdt_answer_reads_modified_cells_not_hardcoded():
    variant = WDT.replace("|Programmed|2|Enabled|Always enabled|Timed sequence|",
                          "|Programmed|2|Disabled|No limitations|Timed sequence|")
    answer = answer_from_tables(WDT_Q, [hit(WDT_ID, variant)])
    assert answer is not None
    assert "Always enabled" not in answer.draft
    assert answer.cells[1]["initial_state"] == "Disabled"
    assert answer.cells[1]["how_to_disable"] == "No limitations"


def test_wdt_answer_requires_relevant_question_and_both_rows():
    assert answer_from_tables("What is the supply voltage?", [hit(WDT_ID, WDT)]) is None
    assert answer_from_tables(WDT_Q, [hit(WDT_ID, WDT.replace("|Unprogrammed|1|Disabled|Timed sequence|No limitations|", ""))]) is None


def test_interconnect_modes_source_cells_are_all_correct():
    answer = answer_from_tables(STOP_Q, [hit(STOP_ID, INTERCONNECT)])
    assert answer is not None
    assert answer.source_id == STOP_ID and answer.kind == "interconnect_matrix"
    assert [c["mode"] for c in answer.cells] == ["Low-power run", "Low-power sleep", "Stop 2"]
    assert [c["cell"] for c in answer.cells] == ["Y", "Y", "-"]
    assert "not supported in Stop 2" in answer.draft
    assert "TIMx -> ADCx DAC1" in answer.draft
    assert "GPIO" not in answer.draft
    assert answer.draft.count(f"[[{STOP_ID}]]") == 3


def test_interconnect_reversed_stop_cell_changes_the_output():
    modified = INTERCONNECT.replace("|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|",
                                    "|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|Y|")
    answer = answer_from_tables(STOP_Q, [hit(STOP_ID, modified)])
    assert answer is not None
    assert answer.cells[2]["cell"] == "Y"
    assert "is supported in Stop 2" in answer.draft
    assert "not supported in Stop 2" not in answer.draft


def test_interconnect_rejects_missing_requested_mode_cell():
    modified = INTERCONNECT.replace("|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|",
                                    "|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|?|")
    assert answer_from_tables(STOP_Q, [hit(STOP_ID, modified)]) is None


def test_interconnect_rejects_unrelated_row():
    only_gpio = INTERCONNECT.replace("|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|\n", "")
    assert answer_from_tables(STOP_Q, [hit(STOP_ID, only_gpio)]) is None


def test_interconnect_no_ambiguity_if_duplicate_same_source():
    answer = answer_from_tables(STOP_Q, [hit(STOP_ID, INTERCONNECT),
                                          hit("stm32l432kc_p030_c001", INTERCONNECT)])
    assert answer is not None
    assert answer.source_id == STOP_ID


def test_interconnect_conflicting_chunks_trigger_fallback():
    changed = INTERCONNECT.replace("|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|-|",
                                   "|TIMx|ADCx<br>DAC1|Conversion triggers|Y|Y|Y|Y|-|Y|")
    assert answer_from_tables(STOP_Q, [hit(STOP_ID, INTERCONNECT),
                                      hit("stm32l432kc_p030_c001", changed)]) is None


def test_invalid_table_does_not_pass():
    broken = WDT.replace("|---|---|---|---|---|", "not a real table")
    assert answer_from_tables(WDT_Q, [hit(WDT_ID, broken)]) is None


def test_unrelated_question_falls_back_to_llm():
    class FakeLLM:
        def __init__(self): self.calls = 0
        def generate(self, question, hits):
            self.calls += 1
            return "LLM answer"
    llm = FakeLLM()
    generator = TableFirstGenerator(llm)
    assert generator.generate("What is the TIM2 counter resolution?", [hit(STOP_ID, INTERCONNECT)]) == "LLM answer"
    assert generator.last_strategy == "local_llm" and llm.calls == 1
    assert generator.last_extraction is None


def test_table_first_skips_llm_when_evidence_complete():
    class ForbiddenLLM:
        def generate(self, question, hits):
            raise AssertionError("should not call LLM for exact table")
    generator = TableFirstGenerator(ForbiddenLLM())
    result = generator.generate(WDT_Q, [hit(WDT_ID, WDT)])
    assert "Always enabled" in result
    assert generator.last_strategy == "source_table_extraction"
    assert generator.last_extraction is not None


def test_explicit_disable_switch_allows_baseline():
    class FakeLLM:
        def generate(self, question, hits): return "baseline"
    generator = TableFirstGenerator(FakeLLM(), enabled=False)
    assert generator.generate(STOP_Q, [hit(STOP_ID, INTERCONNECT)]) == "baseline"
    assert generator.last_extraction is None


def test_source_derived_mode_claim_passes_existing_citation_and_table_checks():
    from tech_specs.local_answer import validate_citations
    from tech_specs.table_guard import inspect_table_claims
    row = hit(STOP_ID, INTERCONNECT)
    row["facet_matches"] = {"timer": {"rank": 1}}
    answer = answer_from_tables(STOP_Q, [row])
    check = validate_citations(answer.draft, [row], facet_ids=["timer"])
    assert check["accepted"] is True
    report = {"draft": answer.draft, "status": "citations_checked",
              "citation_check": check, "retrieval": {"hits": [row]}}
    table_check = inspect_table_claims(report)
    assert table_check["passed"] is True
    assert table_check["checked_cells"] == 3


def test_source_derived_wdt_claim_passes_existing_citation_checks():
    from tech_specs.local_answer import validate_citations
    row = hit(WDT_ID, WDT)
    row["facet_matches"] = {"watchdog": {"rank": 1}}
    answer = answer_from_tables(WDT_Q, [row])
    check = validate_citations(answer.draft, [row], facet_ids=["watchdog"])
    assert check["accepted"] is True
