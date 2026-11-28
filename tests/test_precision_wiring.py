"""Ensure only imported documents gain evidence-focused search; no databases used."""
import json
import sys
import types
from tech_specs.web_runtime import AgentResources


def test_imported_doc_uses_focus_while_builtin_stays_unchanged(tmp_path, monkeypatch):
    doc_id = "user-synthetic-fake"
    folder = tmp_path / doc_id
    folder.mkdir()
    chunks = [
        {"chunk_id": "memory", "document": doc_id, "page": 35,
         "content": "CPU memory and AES cache", "content_type": "text"},
        {"chunk_id": "cpu", "document": doc_id, "page": 33,
         "content": "A 32-bit ARM Cortex-M33 CPU core with RISC instruction set", "content_type": "text"},
    ]
    (folder / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    monkeypatch.setattr("tech_specs.custom_docs.library_dir", lambda root: tmp_path)
    fake_retrieval_module = types.ModuleType("tech_specs.retrieval")
    fake_retrieval_module.load_chunks = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
    monkeypatch.setitem(sys.modules, "tech_specs.retrieval", fake_retrieval_module)

    class Fake:
        def search(self, query, top_k=5, document=None):
            return [{"rank": 1, "chunk": chunks[0]}]

    old = Fake()
    resources = AgentResources(tmp_path)
    monkeypatch.setattr(resources, "_base_retriever", lambda method, doc: old)
    custom = resources._retriever("bm25", doc_id)
    assert custom is resources._retriever("bm25", doc_id)
    assert custom.search("What CPU architecture is used?", document=doc_id)[0]["chunk"]["chunk_id"] == "cpu"
    assert resources._retriever("bm25", None) is old
    resources.close()
    assert resources._focused_retrievers == {}
