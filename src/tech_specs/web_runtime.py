"""Resource lifecycle and presentation helpers for the local Tech-Specs web UI.

Uses the existing v5 graph unchanged. Loading libraries/indexes is lazy; the
Qdrant Local client is shared within one application process to avoid locks.
The service never accesses benchmark questions, anchors, or answer keys.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

VISIBLE_STATUSES = frozenset({
    "source_table_checked", "local_checks_passed", "coverage_review_recommended"
})

STATUS_INFO = {
    "source_table_checked": ("Source table cells checked", "The answer matches retrieved table cells and citation IDs; the original PDF was not independently audited."),
    "local_checks_passed": ("Local checks passed", "Local citation and judge checks passed; this is not independent proof of factual correctness."),
    "coverage_review_recommended": ("Coverage review recommended", "Cited claims passed local checks, but the response may omit requested details."),
    "review_required": ("Answer withheld", "The answer did not pass required local checks. Inspect evidence and diagnostics."),
    "citation_check_failed": ("Invalid citations", "The generated answer failed citation validation."),
    "insufficient_evidence": ("Insufficient evidence", "The local model could not support an answer from the retrieved evidence."),
    "repair_failed": ("Revision failed", "The revision could not produce an accepted answer."),
    "no_evidence": ("No evidence found", "No relevant document chunks were retrieved."),
    "missing_facet_candidates": ("Partial evidence", "One or more aspects of the question had no retrieved candidates."),
}


def status_details(status: str) -> tuple[str, str, bool]:
    """Display label, qualifier and whether it is safe to display draft."""
    label, detail = STATUS_INFO.get(status, (status or "Unknown status", "Review the result before using it."))
    return label, detail, status in VISIBLE_STATUSES


def source_evidence(state: dict, *, max_chars: int = 2400) -> list[dict]:
    """Return retrieved chunks; cited != supported, just referenced by the output."""
    work = state.get("working_result") or state.get("original_result") or {}
    retrieval = work.get("retrieval") or {}
    cited_ids = set((work.get("citation_check") or {}).get("cited_chunk_ids") or [])
    rows: list[dict] = []
    for hit in retrieval.get("hits", []):
        chunk = hit.get("chunk") or {}
        cid = str(chunk.get("chunk_id", ""))
        if not cid:
            continue
        text = str(chunk.get("content", ""))
        rows.append({
            "chunk_id": cid,
            "document": str(chunk.get("document", "")),
            "page": chunk.get("page"),
            "section": str(chunk.get("section", "")),
            "content": text[:max_chars] + ("…" if len(text) > max_chars else ""),
            "cited": cid in cited_ids,
            "rank": hit.get("rank"),
            "facets": sorted((hit.get("facet_matches") or {}).keys()),
        })
    return rows


def format_report_json(state: dict) -> str:
    """Export the full agent run for debugging, without needing file writes."""
    return json.dumps(state, ensure_ascii=False, indent=2, default=str)


class AgentResources:
    """Shared local RAG resources; original and imported PDF indexes are isolated."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.processed = self.root / "data" / "processed"
        self._lock = threading.RLock()
        self._bm25 = None
        self._vector = None
        self._hybrid = None
        self._embedding = None
        self._qdrant_client = None
        self._custom_bm25: dict[str, Any] = {}
        self._custom_vector: dict[str, Any] = {}
        self._custom_hybrid: dict[str, Any] = {}
        self._custom_clients: dict[str, Any] = {}
        self._graphs: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def _get_bm25(self, doc_id: str | None = None):
        from tech_specs.retrieval import BM25Retriever, load_chunks
        if doc_id is not None:
            from tech_specs.custom_docs import get_custom_document, library_dir
            get_custom_document(self.root, doc_id)  # ensure a registered ID
            if doc_id not in self._custom_bm25:
                corpus = library_dir(self.root) / doc_id / "chunks.jsonl"
                self._custom_bm25[doc_id] = BM25Retriever(load_chunks(corpus))
            return self._custom_bm25[doc_id]
        if self._bm25 is None:
            self._bm25 = BM25Retriever(load_chunks(self.processed / "chunks.jsonl"))
        return self._bm25

    def _get_embedding(self):
        if self._embedding is None:
            from fastembed import TextEmbedding
            from tech_specs.vector_retrieval import MODEL_NAME
            self._embedding = TextEmbedding(model_name=MODEL_NAME)
        return self._embedding

    def _get_vector(self, doc_id: str | None = None):
        from qdrant_client import QdrantClient
        from tech_specs.vector_retrieval import VectorRetriever, load_manifest, verify_index
        if doc_id is not None:
            from tech_specs.custom_docs import get_custom_document, library_dir
            info = get_custom_document(self.root, doc_id)
            if not info.get("vector_indexed"):
                raise ValueError("This imported PDF has no vector index yet. Build it in the sidebar, or use BM25.")
            if doc_id not in self._custom_vector:
                folder = library_dir(self.root) / doc_id
                corpus = folder / "chunks.jsonl"
                manifest = load_manifest(folder / "vector_index_manifest.json")
                client = QdrantClient(path=str(folder / "qdrant_local"))
                try:
                    verify_index(client, manifest, corpus_path=corpus)
                    vector = VectorRetriever(client, self._get_embedding())
                except Exception:
                    client.close()
                    raise
                self._custom_clients[doc_id] = client
                self._custom_vector[doc_id] = vector
            return self._custom_vector[doc_id]
        if self._vector is None:
            from tech_specs.vector_retrieval import load_manifest
            corpus = self.processed / "chunks.jsonl"
            manifest = load_manifest(self.processed / "vector_index_manifest.json")
            if manifest.get("partial_index"):
                raise ValueError("The original vector index is partial; rebuild it before answering questions.")
            client = QdrantClient(path=str(self.processed / "qdrant_local"))
            try:
                verify_index(client, manifest, corpus_path=corpus)
                vector = VectorRetriever(client, self._get_embedding())
            except Exception:
                client.close()
                raise
            self._qdrant_client = client
            self._vector = vector
        return self._vector

    def _retriever(self, method: str, doc_id: str | None = None):
        if method == "bm25":
            return self._get_bm25(doc_id)
        if method == "vector":
            return self._get_vector(doc_id)
        if method == "hybrid":
            from tech_specs.hybrid_retrieval import HybridRetriever
            if doc_id is not None:
                if doc_id not in self._custom_hybrid:
                    self._custom_hybrid[doc_id] = HybridRetriever(
                        self._get_bm25(doc_id), self._get_vector(doc_id))
                return self._custom_hybrid[doc_id]
            if self._hybrid is None:
                self._hybrid = HybridRetriever(self._get_bm25(), self._get_vector())
            return self._hybrid
        raise ValueError(f"Unsupported retrieval method: {method}")

    def ask(self, question: str, *, method: str = "bm25", document: str | None = None,
            model: str = "qwen3:4b", top_k: int = 6, candidates_per_facet: int = 10,
            num_ctx: int = 4096, num_predict: int = 1200,
            max_repairs: int = 1, table_first: bool = True) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please enter a nonempty question.")
        custom = document is not None and document not in {"attiny85", "stm32l432kc"}
        if custom:
            from tech_specs.custom_docs import get_custom_document
            get_custom_document(self.root, document)
        if min(top_k, candidates_per_facet, num_ctx, num_predict) < 1:
            raise ValueError("Retrieval and generation settings must be positive.")
        if max_repairs not in (0, 1):
            raise ValueError("This version supports zero or one revision attempt.")

        key = (document if custom else None, method, model, top_k,
               candidates_per_facet, num_ctx, num_predict, max_repairs, table_first)
        with self._lock:
            if key not in self._graphs:
                from tech_specs.answer_graph import build_answer_graph
                from tech_specs.faithfulness import OllamaEvidenceJudge
                from tech_specs.local_answer import LocalOllama
                from tech_specs.review_models import OllamaCoverageJudge, OllamaRepair
                from tech_specs.table_answers import TableFirstGenerator
                from tech_specs.verified_answer_graph import build_verified_answer_graph
                retriever = self._retriever(method, document if custom else None)
                generator = TableFirstGenerator(
                    LocalOllama(model=model, num_ctx=num_ctx, num_predict=num_predict),
                    enabled=table_first and not custom,
                )
                if custom:
                    from tech_specs.generic_answer_graph import build_generic_answer_graph
                    base = build_generic_answer_graph(
                        retriever, generator, document, top_k=top_k,
                        candidates_per_facet=candidates_per_facet)
                else:
                    base = build_answer_graph(retriever, generator, top_k=top_k,
                                             candidates_per_facet=candidates_per_facet)
                graph = build_verified_answer_graph(
                    base,
                    judge=OllamaEvidenceJudge(model=model, num_ctx=num_ctx),
                    coverage_judge=OllamaCoverageJudge(model=model, num_ctx=num_ctx),
                    repairer=OllamaRepair(model=model, num_ctx=num_ctx, num_predict=num_predict),
                    max_repairs=max_repairs,
                    source_cell_route=table_first and not custom,
                )
                self._graphs[key] = (graph, generator)
            graph, generator = self._graphs[key]
            generator.last_strategy = "not_run"
            generator.last_extraction = None
            result = graph.invoke({"question": question.strip(),
                                   "document_hint": None if custom else document})
            result["initial_answer_strategy"] = generator.last_strategy
            result["final_answer_strategy"] = (
                "source_table_extraction" if result.get("status") == "source_table_checked"
                else "local_llm_repair" if result.get("repairs", 0)
                     and (result.get("working_result") or {}).get("status") == "citations_checked"
                else "repair_failed_or_abstained" if result.get("repairs", 0)
                else generator.last_strategy
            )
            if generator.last_extraction is not None:
                extraction = generator.last_extraction
                result["table_extraction"] = {
                    "source_id": extraction.source_id,
                    "kind": extraction.kind,
                    "cells": list(extraction.cells),
                }
            return result

    def close(self):
        with self._lock:
            if self._qdrant_client is not None:
                self._qdrant_client.close()
                self._qdrant_client = None
            for client in self._custom_clients.values():
                client.close()
            self._custom_clients.clear()
            self._custom_vector.clear()
            self._custom_hybrid.clear()
            self._graphs.clear()
            self._vector = None
            self._hybrid = None
