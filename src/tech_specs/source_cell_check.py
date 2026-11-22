"""Deterministic provenance check for narrowly supported Markdown table answers.

Only accepts answers that are BYTE-FOR-BYTE (after external whitespace stripping)
reconstructed from recognized, unambiguous source tables and whose citations
are valid for all retrieval facets. It is not a general semantic entailment
checker or a verification of the original manufacturer PDF.
"""
from __future__ import annotations

import hashlib
from typing import Any

from tech_specs.local_answer import validate_citations
from tech_specs.table_answers import answer_from_tables


def check_source_cell_answer(report: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a source-table answer and verify exact output and provenance.

    This intentionally does *not* ask an LLM to grade its own generated answer.
    A missing, ambiguous, inconsistent, or modified table never qualifies.
    """
    base = {
        "applicable": False,
        "accepted": False,
        "kind": None,
        "source_id": None,
        "cells": [],
        "source_sha256": None,
        "reason": "not_a_recognized_table_answer",
        "scope": "Exact answer reconstruction from retrieved Markdown table cells and cited source IDs only",
    }
    if not isinstance(report, dict) or report.get("status") != "citations_checked":
        return {**base, "reason": "generation_not_citation_checked"}
    if not isinstance(report.get("citation_check"), dict) or report["citation_check"].get("accepted") is not True:
        return {**base, "reason": "original_citations_not_accepted"}
    question, answer = report.get("question"), report.get("draft")
    retrieval, plan = report.get("retrieval"), report.get("plan")
    if not isinstance(question, str) or not isinstance(answer, str):
        return {**base, "reason": "missing_question_or_answer"}
    if not isinstance(retrieval, dict) or not isinstance(retrieval.get("hits"), list):
        return {**base, "reason": "missing_retrieved_hits"}
    if not isinstance(plan, dict) or not isinstance(plan.get("facets"), list) or not plan["facets"]:
        return {**base, "reason": "missing_plan_facets"}
    try:
        facets = [facet["id"] for facet in plan["facets"]]
        if not all(isinstance(fid, str) and fid for fid in facets):
            return {**base, "reason": "invalid_plan_facets"}
        hits = retrieval["hits"]
        extraction = answer_from_tables(question, hits)
    except (AttributeError, TypeError, KeyError, ValueError):
        return {**base, "reason": "malformed_retrieval_or_table"}
    if extraction is None:
        return base
    check = {**base, "applicable": True, "kind": extraction.kind,
             "source_id": extraction.source_id, "cells": list(extraction.cells)}
    if answer.strip() != extraction.draft.strip():
        return {**check, "reason": "draft_differs_from_exact_source_cells"}
    try:
        citations = validate_citations(answer, hits, facet_ids=facets)
    except (KeyError, TypeError, ValueError):
        return {**check, "reason": "citation_revalidation_failed"}
    if not citations["accepted"] or citations["cited_chunk_ids"] != [extraction.source_id]:
        return {**check, "reason": "citations_or_facet_coverage_invalid"}
    # Original approval must refer to the same set of IDs. A stale approval is
    # not enough to bypass the guard for a source-cell answer.
    if report["citation_check"].get("cited_chunk_ids") != citations["cited_chunk_ids"]:
        return {**check, "reason": "stored_citations_do_not_match"}
    try:
        matching = [hit["chunk"] for hit in hits
                    if hit.get("chunk", {}).get("chunk_id") == extraction.source_id]
    except (TypeError, KeyError, AttributeError):
        return {**check, "reason": "malformed_source_hit"}
    if not matching or any(chunk.get("content") != matching[0].get("content") for chunk in matching):
        return {**check, "reason": "duplicate_source_id_has_conflicting_content"}
    text = matching[0]["content"]
    if not isinstance(text, str) or not text.strip():
        return {**check, "reason": "missing_source_text"}
    return {**check, "accepted": True, "reason": "exact_table_cells_and_citations_match",
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
