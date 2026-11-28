"""Offline tests of the MCP-facing domain logic; no model/index SDK needed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# This file runs under `pip install -e .`, as intended by the repository.
from tech_specs.mcp_service import TechSpecsMCPService


class FakeRetriever:
    def __init__(self, chunk):
        self.chunk = chunk
        self.calls = []
    def search(self, query, top_k=5, document=None):
        self.calls.append((query, top_k, document))
        return [{"rank": 1, "chunk": self.chunk}]


class FakeResources:
    def __init__(self, chunk):
        self.fake_retriever = FakeRetriever(chunk)
        self.args = None
        self.closed = False
    def _retriever(self, method, doc_id=None):
        self.args = (method, doc_id)
        return self.fake_retriever
    def ask(self, question, **kwargs):
        self.args = (question, kwargs)
        return {"status": "review_required", "report": "Could not verify",
                "working_result": {"draft": "Fabricated answer", "citation_check": {"cited_chunk_ids": []},
                                   "retrieval": {"hits": []}}}
    def close(self):
        self.closed = True


@pytest.fixture
def example(tmp_path):
    p = tmp_path / "data" / "processed"
    p.mkdir(parents=True)
    chunk = {"chunk_id": "attiny85_p0001_c001", "document": "attiny85", "page": 1,
             "section": "Features", "content_type": "text", "content": "Watchdog timer available"}
    (p / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf8")
    fake = FakeResources(chunk)
    return TechSpecsMCPService(tmp_path, resources=fake), fake


def test_catalog_lists_two_builtin_docs(example):
    svc, _ = example
    docs = svc.list_datasheets()["documents"]
    assert {d["id"] for d in docs} == {"attiny85", "stm32l432kc"}
    assert all(d["kind"] == "built_in" for d in docs)


def test_search_scoped_to_builtin(example):
    svc, fake = example
    result = svc.search_datasheet("watchdog", "attiny85", method="bm25", top_k=3)
    assert result["results"][0]["pdf_page"] == 1
    assert fake.args == ("bm25", None)
    assert fake.fake_retriever.calls == [("watchdog", 3, "attiny85")]


def test_reject_unknown_and_path_traversal(example):
    svc, _ = example
    with pytest.raises(ValueError):
        svc.search_datasheet("a", "../../other")
    with pytest.raises(ValueError):
        svc.get_chunk("../../other", "x")
    with pytest.raises(ValueError):
        svc.search_datasheet("a", "attiny85", top_k=1000)
    with pytest.raises(ValueError):
        svc.search_datasheet("a", "attiny85", method="unknown")


def test_exact_chunk_read_and_wrong_document_rejected(example):
    svc, _ = example
    chunk = svc.get_chunk("attiny85", "attiny85_p0001_c001")
    assert chunk["content"] == "Watchdog timer available"
    assert chunk["pdf_page"] == 1
    with pytest.raises(ValueError):
        svc.get_chunk("stm32l432kc", "attiny85_p0001_c001")


def test_ask_withheld_is_not_exposed(example):
    svc, fake = example
    out = svc.ask_datasheet("A question", "attiny85")
    assert out["status"] == "review_required"
    assert out["answer_visible"] is False
    assert out["answer"] is None
    assert "Could not verify" in out["diagnostic"]


def test_list_custom_doc_without_vector_index(example):
    svc, fake = example
    doc = "user-example-abcdef1234"
    path = svc.root / "data" / "user_docs" / doc
    path.mkdir(parents=True)
    (path / "source.pdf").write_bytes(b"%PDF-1.4 fake")
    (path / "chunks.jsonl").write_text(json.dumps({"chunk_id": f"{doc}_p0001_t001", "document":doc,
           "page": 1, "content_type":"text", "content":"datasheet"}) + '\n', encoding='utf8')
    (path / "manifest.json").write_text(json.dumps({"schema_version":1,"id":doc,"title":"Example datasheet"}), encoding='utf8')
    catalog=svc.list_datasheets()["documents"]
    assert any(d["id"]==doc and not d["vector_indexed"] for d in catalog)
    svc.search_datasheet("processor", doc)
    assert fake.args == ("bm25", doc)
    assert fake.fake_retriever.calls[-1] == ("processor", 5, None)


def test_close_resource(example):
    svc, fake = example
    svc.close()
    assert fake.closed is True
