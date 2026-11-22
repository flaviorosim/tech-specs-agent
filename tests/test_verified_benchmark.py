"""Offline tests for v5 split safety, audit tracking, and resumable reports."""

import csv
import json

import pytest

from tech_specs.answer_benchmark import ensure_run_config, load_dev_set
from tech_specs.verified_benchmark import make_verified_row, run_verified_batch, summarize


def question():
    return {"id": "q_v3_a04", "query": "How does programming WDTON affect the WDT?",
            "document": "attiny85", "category": "table_comparison",
            "anchors": ["attiny85_p043_c002"],
            "reference_answer": "When programmed, enabled and always enabled.",
            "pdf_review_recommended": False}


def fake_state(status="source_table_checked"):
    cid = "attiny85_p043_c002"
    orig = {"plan": {"document": "attiny85"}}
    work = {"draft": "WDTON programmed: always enabled. [[attiny85_p043_c002]]",
            "retrieval": {"hits": [{"chunk": {"chunk_id": cid}}]},
            "citation_check": {"accepted": True, "cited_chunk_ids": [cid]}}
    return {"question": question()["query"], "status": status, "repairs": 0,
            "original_result": orig, "working_result": work,
            "source_cell_check": {"accepted": True, "kind": "configuration_table",
                                  "cells": [{"row": "Programmed"}, {"row": "Unprogrammed"}]},
            "initial_answer_strategy": "source_table_extraction",
            "final_answer_strategy": "source_table_extraction",
            "audits": [], "coverage_checks": [], "table_checks": []}


def test_source_cell_answer_has_separate_provenance_not_llm_audit():
    row = make_verified_row(question(), fake_state())
    assert row["visible_answer"] is True
    assert row["source_cell_checked"] is True
    assert row["source_cells_checked"] == 2
    assert row["latest_claim_audit"] == "not_run"
    assert row["gold_anchors_retrieved"] == 1 and row["gold_anchors_cited"] == 1
    assert row["manual_correct"] == "" and row["manual_complete"] == ""


def test_coverage_warning_is_visible_but_not_counted_checked():
    state = fake_state("coverage_review_recommended")
    state["audits"] = [{"audit_status": "all_supported_by_local_judge",
                        "counts": {"supported": 1, "unsupported": 0, "uncertain": 0}}]
    state["coverage_checks"] = [{"verdict": "uncertain"}]
    row = make_verified_row(question(), state)
    assert row["visible_answer"] is True
    assert row["source_cell_checked"] is False
    assert summarize([row])["coverage_review_recommended"] == 1
    assert summarize([row])["local_checks_passed"] == 0


def test_blocked_answer_not_displayed_or_scored_correct():
    row = make_verified_row(question(), fake_state("review_required"))
    assert not row["visible_answer"]
    assert row["answer_displayed"] == ""
    assert row["generated_answer_for_review"]
    assert summarize([row])["withheld"] == 1
    assert summarize([row])["accuracy_evaluated"] is False


def test_source_table_must_have_accepted_check():
    state = fake_state()
    state["source_cell_check"]["accepted"] = False
    row = make_verified_row(question(), state)
    assert not row["visible_answer"]
    assert not row["source_cell_checked"]


def test_resume_and_no_gold_in_generator(tmp_path):
    sent = []
    def invoke(query, device):
        sent.append((query, device))
        return fake_state()
    rows = run_verified_batch([question()], invoke, tmp_path)
    assert len(sent) == 1
    assert sent == [(question()["query"], "attiny85")]
    assert rows[0]["status"] == "source_table_checked"
    assert (tmp_path / "q_v3_a04.json").exists()
    assert (tmp_path / "summary.json").exists()
    again = run_verified_batch([question()], invoke, tmp_path)
    assert len(sent) == 1
    assert again[0]["source_cell_checked"] is True
    with (tmp_path / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
        csvrow = next(csv.DictReader(stream))
    assert csvrow["reference_answer"] == question()["reference_answer"]
    assert csvrow["manual_correct"] == ""


def test_stale_state_fails_closed_without_overwriting(tmp_path):
    path = tmp_path / "q_v3_a04.json"
    report = fake_state()
    report["question"] = "some other question"
    path.write_text(json.dumps(report))
    rows = run_verified_batch([question()], lambda *_: (_ for _ in ()).throw(AssertionError("no rerun")), tmp_path)
    assert rows[0]["status"] == "execution_error"
    assert "does not match" in rows[0]["error"]
    assert json.loads(path.read_text()) == report


def test_failed_call_logs_error_and_can_be_retried(tmp_path):
    def fail(*_):
        raise RuntimeError("Ollama server offline")
    rows = run_verified_batch([question()], fail, tmp_path)
    assert rows[0]["status"] == "execution_error"
    assert "offline" in rows[0]["error"]
    assert not (tmp_path / "q_v3_a04.json").exists()
    rows = run_verified_batch([question()], lambda *_: fake_state(), tmp_path)
    assert rows[0]["status"] == "source_table_checked"


def test_refuse_inconsistent_source_cell_acceptance(tmp_path):
    state = fake_state()
    state["source_cell_check"]["accepted"] = False
    rows = run_verified_batch([question()], lambda *_: state, tmp_path)
    assert rows[0]["status"] == "execution_error"
    assert not (tmp_path / "q_v3_a04.json").exists()


def test_configuration_reuse_guard(tmp_path):
    ensure_run_config(tmp_path, {"version": "v5", "method": "bm25"})
    ensure_run_config(tmp_path, {"version": "v5", "method": "bm25"})
    with pytest.raises(ValueError):
        ensure_run_config(tmp_path, {"version": "v5", "method": "hybrid"})


def test_holdout_rejected_even_if_user_changes_questions_path(tmp_path):
    dev = [{"id": question()["id"], "query": question()["query"],
            "document": "attiny85", "relevant_chunk_ids": question()["anchors"]}]
    qs = tmp_path / "questions.json"
    key = tmp_path / "answer_key.json"
    qs.write_text(json.dumps(dev))
    key.write_text(json.dumps({"questions": [{**dev[0], "split": "test", "reference_answer": "ABC"}]}))
    with pytest.raises(ValueError, match="non-development"):
        load_dev_set(qs, key)


def test_reject_invalid_limit(tmp_path):
    with pytest.raises(ValueError):
        run_verified_batch([question()], lambda *_: fake_state(), tmp_path, limit=0)
