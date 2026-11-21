"""First LangGraph workflow: automatic plan -> balanced retrieval -> evidence report.

This is a retrieval workflow, not yet an LLM answer generator or a semantic
fact verifier. The status only describes whether each facet produced candidates.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from tech_specs.auto_planner import make_plan
from tech_specs.facet_retrieval import search_facets


class AgentState(TypedDict, total=False):
    question: str
    document_hint: str | None
    plan: dict
    retrieval: dict
    assessment: dict
    report: str


def assess_candidates(result: dict) -> dict:
    """Check candidate presence only; this is NOT factual verification."""
    facets = result["facets"]
    missing = [f["id"] for f in facets if f["selected_matches"] == 0]
    return {
        "facets_total": len(facets),
        "facets_with_candidates": len(facets) - len(missing),
        "facets_without_candidates": missing,
        "has_any_candidates": bool(result["hits"]),
        "candidate_coverage_only": True,
    }


def format_report(result: dict, assessment: dict) -> str:
    """Format evidence IDs for auditability; do not synthesize an answer."""
    lines = [
        "Evidence candidates (NOT a verified technical answer)",
        f"Document: {result['document'] or 'not identified'}",
        ("Facet candidates: "
         f"{assessment['facets_with_candidates']}/{assessment['facets_total']}"),
    ]
    for facet in result["facets"]:
        lines.append(f"  - {facet['id']}: {facet['query']}")
    if not result["hits"]:
        lines.append("No candidates retrieved; do not answer from these sources.")
    for hit in result["hits"]:
        chunk = hit["chunk"]
        tags = ", ".join(sorted(hit["facet_matches"]))
        lines.append(
            f"#{hit['rank']} {chunk['chunk_id']} "
            f"({chunk['document']}, PDF page {chunk['page']}; facets={tags})"
        )
    lines.append("Caution: candidate presence does not establish answer completeness.")
    return "\n".join(lines)


def build_agent_graph(retriever: Any, *, top_k: int = 6, candidates_per_facet: int = 10):
    """Compile a local, deterministic LangGraph using an existing retriever."""
    if top_k < 1 or candidates_per_facet < 1:
        raise ValueError("top_k and candidates_per_facet must be >= 1")

    def plan_node(state: AgentState) -> dict:
        return {"plan": make_plan(state["question"], state.get("document_hint"))}

    def retrieve_node(state: AgentState) -> dict:
        return {
            "retrieval": search_facets(
                retriever,
                state["plan"],
                top_k=top_k,
                candidates_per_facet=candidates_per_facet,
            )
        }

    def assess_node(state: AgentState) -> dict:
        return {"assessment": assess_candidates(state["retrieval"])}

    def report_node(state: AgentState) -> dict:
        return {"report": format_report(state["retrieval"], state["assessment"])}

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess", assess_node)
    builder.add_node("report", report_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_edge("assess", "report")
    builder.add_edge("report", END)
    return builder.compile()
