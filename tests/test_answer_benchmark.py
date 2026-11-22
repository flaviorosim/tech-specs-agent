"""Offline tests for resumable answer evaluation; no Ollama needed."""

import csv
import json

import pytest

from tech_specs.answer_benchmark import (
    ensure_run_config, load_dev_set, make_row, run_batch, write_review_csv,
)


def q():
    return {
        "id": "q_v3_a01", "query": "What is the EEPROM mode?", "document": "attiny85",
        "category": "table_lookup", "anchors": ["attiny85_p021_c007"],
        "reference_answer": "Erase Only; 1.8 ms.", "pdf_review_recommended": True,
    }


def report():
    return {
        "question": q()["query"], "plan": {"document": "attiny85"},
        "status": "citations_checked", "draft": "Erase Only; 1.8 ms. [[attiny85_p021_c007]]",
        "citation_check": {"accepted": True, "cited_chunk_ids": ["attiny85_p021_c007"]},
        "retrieval": {"hits": [{"chunk": {"chunk_id": "attiny85_p021_c007", "content": "Erase Only; 1.8 ms."}}]},
    }


def audit():
    return {"question": q()["query"], "audit_status": "all_supported_by_local_judge",
            "counts": {"supported": 1, "unsupported": 0, "uncertain": 0}}


def test_load_dev_refuses_holdout(tmp_path):
    p1 = tmp_path / "questions.json"
    p2 = tmp_path / "key.json"
    question = {"id": "q_v3_a01", "query": q()["query"], "document": "attiny85",
                "relevant_chunk_ids": q()["anchors"], "category": "table_lookup"}
    p1.write_text(json.dumps([question]))
    item = dict(question, split="test", reference_answer=q()["reference_answer"])
    p2.write_text(json.dumps({"questions": [item]}))
    with pytest.raises(ValueError, match="non-development"):
        load_dev_set(p1, p2)
    item["split"] = "dev"
    p2.write_text(json.dumps({"questions": [item]}))
    assert load_dev_set(p1, p2)[0]["reference_answer"] == "Erase Only; 1.8 ms."


def test_row_labels_anchors_not_factual_correctness():
    row = make_row(q(), report(), audit())
    assert row["gold_anchors_retrieved"] == 1
    assert row["gold_anchors_cited"] == 1
    assert row["audit_status"] == "all_supported_by_local_judge"
    assert row["manual_correct"] == ""
    assert row["manual_complete"] == ""


def test_run_batch_resume_and_persist(tmp_path):
    calls = []
    def fake_invoke(question, document):
        calls.append((question, document))
        return report()
    class DummyJudge:
        pass
    checks = []
    def fake_audit(rep, judge):
        checks.append(1)
        return audit()
    first = run_batch([q()], fake_invoke, tmp_path, judge=DummyJudge(), audit_fn=fake_audit)
    assert len(first) == 1 and first[0]["status"] == "citations_checked"
    assert (tmp_path / "q_v3_a01.json").is_file()
    assert (tmp_path / "q_v3_a01_faithfulness.json").is_file()
    assert len(calls) == len(checks) == 1
    second = run_batch([q()], fake_invoke, tmp_path, judge=DummyJudge(), audit_fn=fake_audit)
    assert len(calls) == len(checks) == 1  # reused
    assert second[0]["audit_status"] == "all_supported_by_local_judge"


def test_run_batch_can_audit_preexisting_report(tmp_path):
    (tmp_path / "q_v3_a01.json").write_text(json.dumps(report()))
    def should_not_invoke(question, document):
        raise AssertionError("report should be reused")
    rows = run_batch([q()], should_not_invoke, tmp_path,
                     judge=object(), audit_fn=lambda rep, judge: audit())
    assert rows[0]["audit_status"] == "all_supported_by_local_judge"


def test_failure_saved_to_review_not_counted_as_correct(tmp_path):
    def fail(question, document):
        raise RuntimeError("local Ollama offline")
    rows = run_batch([q()], fail, tmp_path)
    assert rows[0]["status"] == "execution_error"
    assert "offline" in rows[0]["error"]
    assert not (tmp_path / "q_v3_a01.json").exists()
    assert (tmp_path / "review.csv").exists()


def test_refuse_stale_report(tmp_path):
    bad = report()
    bad["question"] = "another question"
    (tmp_path / "q_v3_a01.json").write_text(json.dumps(bad))
    rows = run_batch([q()], lambda a, b: report(), tmp_path)
    assert "different question" in rows[0]["error"]
    assert json.loads((tmp_path / "q_v3_a01.json").read_text())["question"] == "another question"


def test_csv_excel_bom_and_user_columns(tmp_path):
    path = tmp_path / "review.csv"
    write_review_csv([make_row(q(), report(), audit())], path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["reference_answer"] == "Erase Only; 1.8 ms."
    assert "manual_correct" in rows[0]


def test_limit_rejects_zero(tmp_path):
    with pytest.raises(ValueError, match="limit"):
        run_batch([q()], lambda x, y: report(), tmp_path, limit=0)


def test_run_config_prevents_mixed_experiments(tmp_path):
    ensure_run_config(tmp_path, {"method": "bm25", "model": "qwen3:4b"})
    ensure_run_config(tmp_path, {"method": "bm25", "model": "qwen3:4b"})
    with pytest.raises(ValueError, match="different run/configuration"):
        ensure_run_config(tmp_path, {"method": "hybrid", "model": "qwen3:4b"})
