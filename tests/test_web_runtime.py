"""Fast smoke checks for UI-facing helpers; no LLM, Qdrant or Streamlit needed."""
from tech_specs.web_runtime import AgentResources, format_report_json, source_evidence, status_details


def test_visible_statuses():
    for key in ("source_table_checked", "local_checks_passed", "coverage_review_recommended"):
        assert status_details(key)[2] is True
    for key in ("review_required", "citation_check_failed", "insufficient_evidence", "repair_failed"):
        assert status_details(key)[2] is False


def test_source_evidence_citations_and_metadata():
    result = {"working_result": {
        "citation_check": {"cited_chunk_ids": ["cid2"]},
        "retrieval": {"hits": [
            {"rank": 1, "chunk": {"chunk_id": "cid1", "document": "attiny85", "page": 12,
                                  "section": "A", "content": "unrelated"}},
            {"rank": 2, "chunk": {"chunk_id": "cid2", "document": "attiny85", "page": 13,
                                  "section": "B", "content": "supported"},
             "facet_matches": {"bod": {"rank": 1}}},
        ]},
    }}
    rows = source_evidence(result)
    assert len(rows) == 2
    assert rows[0]["cited"] is False
    assert rows[1]["cited"] is True and rows[1]["facets"] == ["bod"]
    assert rows[1]["page"] == 13


def test_truncate_long_evidence_and_export_json():
    result = {"status": "source_table_checked", "working_result": {
        "citation_check": {"cited_chunk_ids": ["x"]},
        "retrieval": {"hits": [{"chunk": {"chunk_id": "x", "content": "A" * 300}}]},
    }}
    assert source_evidence(result, max_chars=100)[0]["content"].endswith("…")
    assert '"source_table_checked"' in format_report_json(result)


def test_runtime_rejects_bad_request_without_loading_model(tmp_path):
    resources = AgentResources(tmp_path)
    try:
        resources.ask("   ")
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        assert False
    assert resources._graphs == {}


def test_status_labels_are_english():
    for status in ("source_table_checked", "review_required", "coverage_review_recommended"):
        label, detail, _ = status_details(status)
        assert isinstance(label, str) and isinstance(detail, str)
        assert label and detail
    assert status_details("review_required")[0] == "Answer withheld"
