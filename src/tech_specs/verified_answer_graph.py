"""Guarded LangGraph: original answer -> audit + coverage -> at most one repair.

Never present a draft that fails these local checks as an accepted answer.
These checks are fallible and do NOT establish independent factual correctness.
"""
from __future__ import annotations

from typing import Any, TypedDict
import inspect

from langgraph.graph import START, END, StateGraph

from tech_specs.faithfulness import audit_report
from tech_specs.local_answer import ModelAbstained, citation_sources, validate_citations
from tech_specs.review_models import FRESH_REPAIR_SIGNAL, validate_coverage_result
from tech_specs.table_guard import inspect_table_claims
from tech_specs.source_cell_check import check_source_cell_answer


class GuardedState(TypedDict, total=False):
    question: str
    document_hint: str | None
    original_result: dict
    working_result: dict
    audits: list[dict]
    coverage_checks: list[dict]
    table_checks: list[dict]
    source_cell_check: dict
    repairs: int
    status: str
    report: str
    error_detail: str


def make_repair_feedback(audit: dict, coverage: dict, table_check: dict | None = None) -> str:
    # The original unsupported claim must not be echoed to the repair model:
    # small local LLMs tend to repeat the exact mistaken wording.
    table_issues = (table_check or {}).get("issues", [])
    if any(c.get("verdict") == "unsupported" for c in audit.get("claims", [])) or table_issues:
        return (
            FRESH_REPAIR_SIGNAL
            + "\nOne or more claims contradicted the source material. "
            + "Answer independently using the source table rows, exact qualifiers, "
              "and the user's question. "
            + "\nMachine-readable table conflicts (from cited source cells): "
            + "; ".join(i["message"] for i in table_issues[:3])
        )
    parts = []
    for item in audit.get("claims", []):
        if item.get("verdict") != "supported":
            parts.append(
                f"A claim was {item.get('verdict')}: {item.get('reason', '')[:450]}"
            )
    if coverage.get("verdict") != "complete":
        parts.append(
            "Question coverage was " + coverage.get("verdict", "unknown")
            + ". Possible missing aspects (verify against the actual question): "
            + "; ".join(coverage.get("missing_aspects", []))[:800]
            + ". " + coverage.get("reason", "")[:400]
        )
    return "\n".join(parts) or "Re-read all cited evidence and answer the question precisely."


def local_checks_pass(state: GuardedState) -> bool:
    audits, coverage = state.get("audits", []), state.get("coverage_checks", [])
    tables = state.get("table_checks", [])
    result = state.get("working_result") or {}
    return (
        result.get("status") == "citations_checked"
        and result.get("citation_check", {}).get("accepted") is True
        and bool(audits)
        and audits[-1].get("audit_status") == "all_supported_by_local_judge"
        and bool(coverage)
        and coverage[-1].get("verdict") == "complete"
        and bool(tables)
        and tables[-1].get("passed") is True
    )


def supported_but_coverage_unverified(state: GuardedState) -> bool:
    """Show a factually audited answer *with a warning*, never call it verified.

    Coverage is a fallible heuristic. Do not discard claims solely because it
    returns uncertain/incomplete. Explicit unanswered parts and table conflicts
    still prevent disclosure as an accepted answer.
    """
    result = state.get("working_result") or {}
    audits, tables = state.get("audits", []), state.get("table_checks", [])
    coverages = state.get("coverage_checks", [])
    return bool(
        result.get("status") == "citations_checked"
        and result.get("citation_check", {}).get("accepted")
        and audits and audits[-1].get("audit_status") == "all_supported_by_local_judge"
        and tables and tables[-1].get("passed") is True
        and coverages and coverages[-1].get("verdict") in {"incomplete", "uncertain"}
        and "Information not established by the supplied evidence:" not in result.get("draft", "")
    )


def build_verified_answer_graph(base_graph: Any, judge: Any, coverage_judge: Any,
                                repairer: Any, *, max_repairs: int = 1, source_cell_route: bool = True):
    """Wrap the existing answer graph. One revision by default; no gold labels supplied."""
    if max_repairs not in (0, 1):
        raise ValueError("This experimental graph supports max_repairs=0 or 1")

    def draft(state: GuardedState) -> dict:
        result = base_graph.invoke({
            "question": state["question"],
            "document_hint": state.get("document_hint"),
        })
        return {
            "original_result": result, "working_result": result,
            "audits": [], "coverage_checks": [], "table_checks": [],
            "repairs": 0, "error_detail": "",
            "source_cell_check": {"accepted": False, "applicable": False, "reason": "not_checked"},
        }

    def route_after_draft(state: GuardedState) -> str:
        return "source_cells" if state["working_result"].get("status") == "citations_checked" else "finish"

    def source_cells_node(state: GuardedState) -> dict:
        # Only the exact text deterministically reconstructed from actual table
        # cells can take this route. A free-form LLM answer never receives
        # source-cell status merely for citing a table.
        if not source_cell_route:
            return {"source_cell_check": {"accepted": False, "applicable": False,
                                          "reason": "source_cell_route_disabled"}}
        return {"source_cell_check": check_source_cell_answer(state["working_result"])}

    def route_after_source_cells(state: GuardedState) -> str:
        return "finish" if state["source_cell_check"]["accepted"] else "audit"

    def audit_node(state: GuardedState) -> dict:
        try:
            audit = audit_report(state["working_result"], judge)
        except RuntimeError as exc:
            audit = {"audit_status": "review_required", "claims": [],
                     "counts": {"supported": 0, "unsupported": 0, "uncertain": 0},
                     "error": f"Auditor unavailable: {exc}"}
        return {"audits": [*state.get("audits", []), audit]}

    def table_node(state: GuardedState) -> dict:
        result = inspect_table_claims(state["working_result"])
        return {"table_checks": [*state.get("table_checks", []), result]}

    def coverage_node(state: GuardedState) -> dict:
        current = state["working_result"]
        # Keep older test-double judges compatible, while the real checker gets
        # retrieved sources so it can distinguish a true gap from a table '-'.
        params = inspect.signature(coverage_judge.check).parameters
        try:
            if "hits" in params:
                result = coverage_judge.check(
                    state["question"], current["draft"],
                    hits=current["retrieval"]["hits"],
                )
            else:
                result = coverage_judge.check(state["question"], current["draft"])
        except RuntimeError as exc:
            result = {"verdict": "uncertain", "missing_aspects": [],
                      "reason": f"Coverage judge unavailable: {exc}"}
        # A safeguard independent of the judge output: explicit missing info
        # must never be accepted as a fully answered question.
        if "Information not established by the supplied evidence:" in current["draft"]:
            result = validate_coverage_result(result, answer=current["draft"])
        return {"coverage_checks": [*state.get("coverage_checks", []), result]}

    def route_after_checks(state: GuardedState) -> str:
        if (local_checks_pass(state) or state.get("repairs", 0) >= max_repairs):
            return "finish"
        # If only the heuristic coverage judge doubts an otherwise supported
        # answer, do not waste another call asking the model to rewrite it.
        if (supported_but_coverage_unverified(state)
                and state["coverage_checks"][-1].get("verdict") == "uncertain"):
            return "finish"
        return "repair"

    def repair_node(state: GuardedState) -> dict:
        original = state["working_result"]
        feedback = make_repair_feedback(
            state["audits"][-1], state["coverage_checks"][-1],
            state["table_checks"][-1],
        )
        new_result = dict(original)
        try:
            draft_text = repairer.repair(
                state["question"], original["retrieval"]["hits"],
                original["draft"], feedback,
            )
        except ModelAbstained as exc:
            new_result.update({"status": "insufficient_evidence", "draft": ""})
            return {"working_result": new_result, "repairs": state["repairs"] + 1,
                    "error_detail": f"Repair model abstained: {exc}"}
        except (ValueError, TypeError, RuntimeError) as exc:
            new_result.update({"status": "repair_failed", "draft": ""})
            return {"working_result": new_result, "repairs": state["repairs"] + 1,
                    "error_detail": f"Repair could not finish: {exc}"}

        facet_ids = [facet["id"] for facet in original["plan"]["facets"]]
        checks = validate_citations(
            draft_text, original["retrieval"]["hits"], facet_ids=facet_ids,
        )
        new_result.update({
            "draft": draft_text,
            "citation_check": checks,
            "status": "citations_checked" if checks["accepted"] else "citation_check_failed",
        })
        return {"working_result": new_result, "repairs": state["repairs"] + 1,
                "error_detail": "; ".join(checks["issues"])}

    def route_after_repair(state: GuardedState) -> str:
        return "audit" if state["working_result"].get("status") == "citations_checked" else "finish"

    def finish(state: GuardedState) -> dict:
        result = state["working_result"]
        source_check = state.get("source_cell_check") or {}
        if source_check.get("accepted"):
            sources = citation_sources(
                result["retrieval"]["hits"],
                result["citation_check"]["cited_chunk_ids"],
            )
            lines = [result["draft"], "", "Cited sources:"]
            lines.extend(
                f"- {src['chunk_id']} | {src['document']} | PDF page {src['pdf_page']}"
                for src in sources
            )
            lines.extend([
                "",
                "SOURCE TABLE CELLS CHECKED: exact regenerated answer matches the",
                "retrieved table's headers, rows and requested cells; citation IDs were revalidated.",
                "No local LLM factual/coverage judge was run for this deterministic answer.",
                "This is NOT independent verification of the original PDF or universal semantic correctness.",
            ])
            return {"status": "source_table_checked", "report": "\n".join(lines)}
        verified = local_checks_pass(state)
        provisional = supported_but_coverage_unverified(state)
        if verified or provisional:
            sources = citation_sources(
                result["retrieval"]["hits"],
                result["citation_check"]["cited_chunk_ids"],
            )
            lines = [result["draft"], "", "Cited sources:"]
            lines.extend(f"- {s['chunk_id']} | {s['document']} | PDF page {s['pdf_page']}"
                         for s in sources)
            lines.append("")
            if verified:
                lines.append("LOCAL CHECKS PASSED: citations, local claim judge, and heuristic coverage.")
            else:
                lines.extend([
                    "COVERAGE REVIEW RECOMMENDED: cited claims passed the local factual judge,",
                    "but the coverage judge was uncertain/incomplete. This answer may omit requirements.",
                    "Missing/uncertain: " + "; ".join(state["coverage_checks"][-1].get("missing_aspects", [])),
                ])
            lines.append("Local checks are NOT independent proof of factual correctness.")
            return {"status": "local_checks_passed" if verified else "coverage_review_recommended",
                    "report": "\n".join(lines)}
        status = result.get("status", "review_required")
        if status == "citations_checked":
            status = "review_required"
        last_audit = state.get("audits", [])[-1] if state.get("audits") else {}
        last_coverage = state.get("coverage_checks", [])[-1] if state.get("coverage_checks") else {}
        reasons = [
            f"current generation status={result.get('status', 'unknown')}",
            f"claim audit={last_audit.get('audit_status', 'not_run')}",
            f"coverage={last_coverage.get('verdict', 'not_run')}",
            f"table_conflicts={len((state.get('table_checks') or [{}])[-1].get('issues', []))}",
        ]
        if state.get("error_detail"):
            reasons.append(state["error_detail"])
        return {
            "status": status,
            "report": (
                "Answer withheld: local review did not pass. " + "; ".join(reasons)
                + ". Inspect the saved JSON for the original and revised drafts."
            ),
        }

    builder = StateGraph(GuardedState)
    for name, func in (
        ("draft", draft), ("source_cells", source_cells_node),
        ("audit", audit_node), ("table", table_node),
        ("coverage", coverage_node),
        ("repair", repair_node), ("finish", finish),
    ):
        builder.add_node(name, func)
    builder.add_edge(START, "draft")
    builder.add_conditional_edges("draft", route_after_draft, {"source_cells": "source_cells", "finish": "finish"})
    builder.add_conditional_edges("source_cells", route_after_source_cells, {"finish": "finish", "audit": "audit"})
    builder.add_edge("audit", "table")
    builder.add_edge("table", "coverage")
    builder.add_conditional_edges("coverage", route_after_checks, {"repair": "repair", "finish": "finish"})
    builder.add_conditional_edges("repair", route_after_repair, {"audit": "audit", "finish": "finish"})
    builder.add_edge("finish", END)
    return builder.compile()
