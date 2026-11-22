"""LangGraph workflow: automatic retrieval -> local answer -> citation verification.

Uses the existing planner and facet-balanced retriever. The LLM is injected so
unit tests can run without starting Ollama or downloading a model.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from tech_specs.agent_graph import assess_candidates
from tech_specs.auto_planner import make_plan
from tech_specs.facet_retrieval import search_facets
from tech_specs.local_answer import ModelAbstained, citation_sources, validate_citations


class AnswerState(TypedDict, total=False):
    question: str
    document_hint: str | None
    plan: dict
    retrieval: dict
    assessment: dict
    draft: str
    generation_notice: str
    citation_check: dict
    status: str
    report: str


def build_answer_graph(retriever: Any, llm: Any, *, top_k: int = 6, candidates_per_facet: int = 10):
    if top_k < 1 or candidates_per_facet < 1:
        raise ValueError("top_k and candidates_per_facet must be >= 1")

    def plan_node(state: AnswerState) -> dict:
        return {"plan": make_plan(state["question"], state.get("document_hint"))}

    def retrieve_node(state: AnswerState) -> dict:
        result = search_facets(
            retriever, state["plan"],
            top_k=top_k, candidates_per_facet=candidates_per_facet,
        )
        return {"retrieval": result}

    def assess_node(state: AnswerState) -> dict:
        return {"assessment": assess_candidates(state["retrieval"])}

    def route_after_assessment(state: AnswerState) -> str:
        assessment = state["assessment"]
        if not assessment["has_any_candidates"] or assessment["facets_without_candidates"]:
            return "report"
        return "generate"

    def generate_node(state: AnswerState) -> dict:
        try:
            draft = llm.generate(state["question"], state["retrieval"]["hits"])
        except ModelAbstained as exc:
            return {"draft": "", "status": "insufficient_evidence", "generation_notice": str(exc)}
        return {"draft": draft}

    def route_after_generation(state: AnswerState) -> str:
        return "report" if state.get("status") == "insufficient_evidence" else "validate"

    def validate_node(state: AnswerState) -> dict:
        checks = validate_citations(
            state["draft"],
            state["retrieval"]["hits"],
            facet_ids=[f["id"] for f in state["plan"]["facets"]],
        )
        return {
            "citation_check": checks,
            "status": "citations_checked" if checks["accepted"] else "citation_check_failed",
        }

    def report_node(state: AnswerState) -> dict:
        if state.get("status") == "insufficient_evidence":
            return {"report": "Model abstained: " + state["generation_notice"]}
        assessment = state["assessment"]
        if not assessment["has_any_candidates"]:
            return {"status": "no_evidence", "report": "No source evidence retrieved. No answer generated."}
        if assessment["facets_without_candidates"]:
            missing = ", ".join(assessment["facets_without_candidates"])
            return {
                "status": "missing_facet_candidates",
                "report": f"Missing candidates for facets: {missing}. No answer generated.",
            }
        if state["status"] != "citations_checked":
            issues = "; ".join(state["citation_check"]["issues"])
            return {
                "report": "Generated text failed citation checks and is withheld. " + issues,
            }
        sources = citation_sources(
            state["retrieval"]["hits"], state["citation_check"]["cited_chunk_ids"]
        )
        lines = [state["draft"], "", "Sources (from retrieved chunks):"]
        lines.extend(
            f"- {s['chunk_id']} | {s['document']} | PDF page {s['pdf_page']}"
            for s in sources
        )
        lines.extend([
            "",
            "Warning: citations were checked for valid IDs and facet coverage only; "
            "the factual correctness of each claim has NOT been independently verified.",
        ])
        return {"report": "\n".join(lines)}

    builder = StateGraph(AnswerState)
    for name, node in (
        ("plan", plan_node), ("retrieve", retrieve_node),
        ("assess", assess_node), ("generate", generate_node),
        ("validate", validate_node), ("report", report_node),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges(
        "assess", route_after_assessment,
        {"generate": "generate", "report": "report"},
    )
    builder.add_conditional_edges("generate", route_after_generation, {"validate": "validate", "report": "report"})
    builder.add_edge("validate", "report")
    builder.add_edge("report", END)
    return builder.compile()
