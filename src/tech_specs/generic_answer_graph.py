"""Generic single-document LangGraph route for arbitrary imported PDF datasheets.

Intentionally does not call the two-device glossary planner. This route never
uses source-specific table extraction or benchmark anchors.
"""
from __future__ import annotations

from typing import Any, TypedDict



class GenericAnswerState(TypedDict, total=False):
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


def make_generic_plan(question: str, document_id: str) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("An imported document must be selected")
    return {"question": question.strip(), "document": document_id,
            "facets": [{"id": "general", "query": question.strip()}]}


def build_generic_answer_graph(retriever: Any, llm: Any, document_id: str, *,
                               top_k: int = 6, candidates_per_facet: int = 10):
    from langgraph.graph import END, START, StateGraph
    from tech_specs.agent_graph import assess_candidates
    from tech_specs.facet_retrieval import search_facets
    from tech_specs.local_answer import ModelAbstained, citation_sources, validate_citations

    if top_k < 1 or candidates_per_facet < 1:
        raise ValueError("Retrieval depths must be positive")

    def plan_node(state: GenericAnswerState) -> dict:
        # Explicit document selection wins. A new PDF is not recognized by
        # the built-in STM32/ATtiny device-name glossary.
        return {"plan": make_generic_plan(state["question"], document_id)}

    def retrieve_node(state: GenericAnswerState) -> dict:
        return {"retrieval": search_facets(retriever, state["plan"], top_k=top_k,
                                          candidates_per_facet=candidates_per_facet)}

    def assess_node(state: GenericAnswerState) -> dict:
        return {"assessment": assess_candidates(state["retrieval"])}

    def after_assessment(state: GenericAnswerState) -> str:
        assessment = state["assessment"]
        return "report" if not assessment["has_any_candidates"] or assessment["facets_without_candidates"] else "generate"

    def generate_node(state: GenericAnswerState) -> dict:
        try:
            draft = llm.generate(state["question"], state["retrieval"]["hits"])
        except ModelAbstained as exc:
            return {"draft": "", "status": "insufficient_evidence", "generation_notice": str(exc)}
        return {"draft": draft}

    def after_generate(state: GenericAnswerState) -> str:
        return "report" if state.get("status") == "insufficient_evidence" else "validate"

    def validate_node(state: GenericAnswerState) -> dict:
        checked = validate_citations(state["draft"], state["retrieval"]["hits"], facet_ids=["general"])
        return {"citation_check": checked,
                "status": "citations_checked" if checked["accepted"] else "citation_check_failed"}

    def report_node(state: GenericAnswerState) -> dict:
        if state.get("status") == "insufficient_evidence":
            return {"report": "Model abstained: " + state["generation_notice"]}
        if not state["assessment"]["has_any_candidates"]:
            return {"status": "no_evidence", "report": "No evidence was retrieved; no answer generated."}
        if state["assessment"]["facets_without_candidates"]:
            return {"status": "missing_facet_candidates", "report": "No candidates for one or more question aspects."}
        if state["status"] != "citations_checked":
            return {"report": "Citation verification failed: " + "; ".join(state["citation_check"]["issues"])}
        sources = citation_sources(state["retrieval"]["hits"], state["citation_check"]["cited_chunk_ids"])
        lines = [state["draft"], "", "Sources (retrieved PDF chunks):"]
        lines.extend(f"- {src['chunk_id']} | PDF page {src['pdf_page']}" for src in sources)
        lines.append("Citation IDs were validated, not independently fact-checked.")
        return {"report": "\n".join(lines)}

    builder = StateGraph(GenericAnswerState)
    for name, func in (("plan", plan_node), ("retrieve", retrieve_node),
                       ("assess", assess_node), ("generate", generate_node),
                       ("validate", validate_node), ("report", report_node)):
        builder.add_node(name, func)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges("assess", after_assessment,
                                  {"generate": "generate", "report": "report"})
    builder.add_conditional_edges("generate", after_generate,
                                  {"validate": "validate", "report": "report"})
    builder.add_edge("validate", "report")
    builder.add_edge("report", END)
    return builder.compile()
