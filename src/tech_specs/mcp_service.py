"""Read-only MCP-facing API over the existing Tech-Specs document library.

No document ingestion, indexing, file paths or benchmarks are exposed to clients.
The Streamlit app and MCP server can use the same AgentResources class, but should
run in separate processes only for BM25. Qdrant Local generally requires exclusive
access: stop the Streamlit process before using vector/hybrid through MCP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

BUILTIN_DOCUMENTS = {"attiny85": "ATtiny25/45/85", "stm32l432kc": "STM32L432KC"}
ALLOWED_METHODS = {"bm25", "vector", "hybrid"}


class TechSpecsMCPService:
    """Expose bounded, document-scoped operations to an MCP transport."""

    def __init__(self, root: str | Path, *, resources: Any = None):
        self.root = Path(root).resolve()
        self._resources = resources

    @property
    def resources(self):
        if self._resources is None:
            from tech_specs.web_runtime import AgentResources
            self._resources = AgentResources(self.root)
        return self._resources

    def list_datasheets(self) -> dict:
        from tech_specs.custom_docs import list_custom_documents
        docs: list[dict] = []
        if (self.root / "data" / "processed" / "chunks.jsonl").is_file():
            indexed = (self.root / "data" / "processed" / "vector_index_manifest.json").is_file()
            for doc_id, title in BUILTIN_DOCUMENTS.items():
                docs.append({"id": doc_id, "title": title, "kind": "built_in", "vector_indexed": indexed})
        for doc in list_custom_documents(self.root):
            docs.append({"id": doc["id"], "title": doc.get("title") or doc["id"],
                         "kind": "imported", "vector_indexed": bool(doc.get("vector_indexed", False))})
        return {"documents": docs}

    def _checked_document(self, document: str) -> str:
        if not isinstance(document, str) or not document:
            raise ValueError("Select a document ID from list_datasheets")
        if document not in {doc["id"] for doc in self.list_datasheets()["documents"]}:
            raise ValueError("Unknown or unavailable datasheet ID. Call list_datasheets first.")
        return document

    @staticmethod
    def _checked_question(question: str) -> str:
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Question must contain 1–2000 characters")
        return question.strip()

    @staticmethod
    def _checked_method(method: str) -> str:
        if method not in ALLOWED_METHODS:
            raise ValueError("Method must be one of: bm25, vector, hybrid")
        return method

    def search_datasheet(self, question: str, document: str, *, method: str = "bm25", top_k: int = 5) -> dict:
        """Return original ranked snippets without asking an LLM to answer."""
        q = self._checked_question(question)
        doc = self._checked_document(document)
        method = self._checked_method(method)
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10")
        imported = doc not in BUILTIN_DOCUMENTS
        retriever = self.resources._retriever(method, doc if imported else None)
        hits = retriever.search(q, top_k=top_k, document=None if imported else doc)
        items = []
        for hit in hits:
            chunk = hit["chunk"]
            items.append({"rank": hit.get("rank"), "chunk_id": chunk["chunk_id"],
                          "document": chunk["document"], "pdf_page": chunk.get("page"),
                          "section": chunk.get("section"), "content_type": chunk.get("content_type"),
                          "excerpt": str(chunk.get("content", ""))[:1200]})
        return {"question": q, "document": doc, "method": method, "results": items,
                "note": "Retrieved snippets are candidates, not verified technical answers."}

    def get_chunk(self, document: str, chunk_id: str) -> dict:
        """Return a particular chunk's full stored content and page metadata."""
        doc = self._checked_document(document)
        if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > 256:
            raise ValueError("Invalid chunk_id")
        from tech_specs.retrieval import load_chunks
        if doc in BUILTIN_DOCUMENTS:
            corpus = self.root / "data" / "processed" / "chunks.jsonl"
        else:
            from tech_specs.custom_docs import library_dir
            corpus = library_dir(self.root) / doc / "chunks.jsonl"
        for chunk in load_chunks(corpus):
            if chunk["document"] == doc and chunk["chunk_id"] == chunk_id:
                return {"chunk_id": chunk_id, "document": doc, "pdf_page": chunk.get("page"),
                        "section": chunk.get("section"), "content_type": chunk.get("content_type"),
                        "content": chunk["content"][:12000],
                        "truncated": len(chunk["content"]) > 12000}
        raise ValueError("No chunk with this ID in the selected datasheet")

    def ask_datasheet(self, question: str, document: str, *, method: str = "bm25") -> dict:
        """Use the existing local LangGraph+Ollama answerer with original safeguards."""
        q = self._checked_question(question)
        doc = self._checked_document(document)
        method = self._checked_method(method)
        from tech_specs.web_runtime import source_evidence, status_details
        run = self.resources.ask(q, document=doc, method=method)
        label, qualifier, visible = status_details(run.get("status", ""))
        work = run.get("working_result") or run.get("original_result") or {}
        citations = (work.get("citation_check") or {}).get("cited_chunk_ids") or []
        evidence = [item for item in source_evidence(run, max_chars=850) if item["cited"]]
        return {"question": q, "document": doc, "method": method,
                "status": run.get("status", "unknown"), "status_label": label,
                "qualifier": qualifier, "answer_visible": visible,
                "answer": (work.get("draft") or "") if visible else None,
                "cited_chunk_ids": citations, "cited_evidence": evidence,
                "diagnostic": None if visible else str(run.get("report") or "Answer withheld")[:700],
                "note": "Citations and local checks are not independent guarantees of technical accuracy."}

    def close(self):
        if self._resources is not None and hasattr(self._resources, "close"):
            self._resources.close()
