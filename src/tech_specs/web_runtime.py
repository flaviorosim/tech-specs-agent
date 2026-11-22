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
    "source_table_checked": ("Células conferidas", "A resposta corresponde às células extraídas e aos IDs de fonte; o PDF original não foi auditado independentemente."),
    "local_checks_passed": ("Verificações locais aprovadas", "Citações e verificações locais passaram; isso não garante correção factual independente."),
    "coverage_review_recommended": ("Completude não confirmada", "As afirmações citadas passaram pela auditoria local, mas podem faltar partes da resposta."),
    "review_required": ("Resposta retida", "As verificações não foram satisfeitas; o texto gerado não foi apresentado como resposta."),
    "citation_check_failed": ("Citações inválidas", "A geração não passou pela validação de fontes."),
    "insufficient_evidence": ("Evidência insuficiente", "O agente não encontrou suporte suficiente para responder."),
    "repair_failed": ("Revisão interrompida", "A tentativa de correção não conseguiu produzir uma resposta aceita."),
    "no_evidence": ("Nenhuma evidência", "A pesquisa não encontrou fontes relevantes."),
    "missing_facet_candidates": ("Evidência parcial", "Uma ou mais partes da pergunta não tiveram candidatos recuperados."),
}


def status_details(status: str) -> tuple[str, str, bool]:
    """Display label, qualifier and whether it is safe to display draft."""
    label, detail = STATUS_INFO.get(status, (status or "Desconhecido", "Verifique o resultado antes de utilizá-lo."))
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
    """One local RAG service per UI process; retrievers and graphs are reused."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.processed = self.root / "data" / "processed"
        self._lock = threading.RLock()
        self._chunks = None
        self._bm25 = None
        self._vector = None
        self._hybrid = None
        self._qdrant_client = None
        self._graphs: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def _get_bm25(self):
        if self._bm25 is None:
            from tech_specs.retrieval import BM25Retriever, load_chunks
            self._chunks = load_chunks(self.processed / "chunks.jsonl")
            self._bm25 = BM25Retriever(self._chunks)
        return self._bm25

    def _get_vector(self):
        if self._vector is None:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from tech_specs.vector_retrieval import (
                MODEL_NAME, VectorRetriever, load_manifest, verify_index,
            )
            corpus = self.processed / "chunks.jsonl"
            manifest = load_manifest(self.processed / "vector_index_manifest.json")
            if manifest.get("partial_index"):
                raise ValueError("O índice vetorial é parcial; não use esse índice para responder perguntas.")
            client = QdrantClient(path=str(self.processed / "qdrant_local"))
            try:
                verify_index(client, manifest, corpus_path=corpus)
                embedding = TextEmbedding(model_name=MODEL_NAME)
                vector = VectorRetriever(client, embedding)
            except Exception:
                client.close()
                raise
            self._qdrant_client = client
            self._vector = vector
        return self._vector

    def _retriever(self, method: str):
        if method == "bm25":
            return self._get_bm25()
        if method == "vector":
            return self._get_vector()
        if method == "hybrid":
            if self._hybrid is None:
                from tech_specs.hybrid_retrieval import HybridRetriever
                self._hybrid = HybridRetriever(self._get_bm25(), self._get_vector())
            return self._hybrid
        raise ValueError(f"Método não suportado: {method}")

    def ask(
        self,
        question: str,
        *,
        method: str = "bm25",
        document: str | None = None,
        model: str = "qwen3:4b",
        top_k: int = 6,
        candidates_per_facet: int = 10,
        num_ctx: int = 4096,
        num_predict: int = 1200,
        max_repairs: int = 1,
        table_first: bool = True,
    ) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Digite uma pergunta não vazia.")
        if document not in (None, "attiny85", "stm32l432kc"):
            raise ValueError("Documento inválido.")
        if min(top_k, candidates_per_facet, num_ctx, num_predict) < 1:
            raise ValueError("Os parâmetros de busca e de geração devem ser positivos.")
        if max_repairs not in (0, 1):
            raise ValueError("Esta versão suporta zero ou uma tentativa de correção.")

        key = (method, model, top_k, candidates_per_facet, num_ctx,
               num_predict, max_repairs, table_first)
        with self._lock:
            if key not in self._graphs:
                from tech_specs.answer_graph import build_answer_graph
                from tech_specs.faithfulness import OllamaEvidenceJudge
                from tech_specs.local_answer import LocalOllama
                from tech_specs.review_models import OllamaCoverageJudge, OllamaRepair
                from tech_specs.table_answers import TableFirstGenerator
                from tech_specs.verified_answer_graph import build_verified_answer_graph
                retriever = self._retriever(method)
                generator = TableFirstGenerator(
                    LocalOllama(model=model, num_ctx=num_ctx, num_predict=num_predict),
                    enabled=table_first,
                )
                base = build_answer_graph(
                    retriever, generator, top_k=top_k,
                    candidates_per_facet=candidates_per_facet,
                )
                graph = build_verified_answer_graph(
                    base,
                    judge=OllamaEvidenceJudge(model=model, num_ctx=num_ctx),
                    coverage_judge=OllamaCoverageJudge(model=model, num_ctx=num_ctx),
                    repairer=OllamaRepair(model=model, num_ctx=num_ctx, num_predict=num_predict),
                    max_repairs=max_repairs,
                    source_cell_route=table_first,
                )
                self._graphs[key] = (graph, generator)
            graph, generator = self._graphs[key]
            # No-candidate paths never call generate(); prevent stale metadata
            # from the previous question with the same compiled graph.
            generator.last_strategy = "not_run"
            generator.last_extraction = None
            result = graph.invoke({"question": question.strip(), "document_hint": document})
            result["initial_answer_strategy"] = generator.last_strategy
            result["final_answer_strategy"] = (
                "source_table_extraction" if result.get("status") == "source_table_checked"
                else "local_llm_repair" if result.get("repairs", 0)
                     and (result.get("working_result") or {}).get("status") == "citations_checked"
                else "repair_failed_or_abstained" if result.get("repairs", 0)
                else generator.last_strategy
            )
            if generator.last_extraction is not None:
                extracted = generator.last_extraction
                result["table_extraction"] = {
                    "source_id": extracted.source_id,
                    "kind": extracted.kind,
                    "cells": list(extracted.cells),
                }
            return result

    def close(self):
        with self._lock:
            if self._qdrant_client is not None:
                self._qdrant_client.close()
                self._qdrant_client = None
                self._vector = None
                self._hybrid = None
                self._graphs.clear()
