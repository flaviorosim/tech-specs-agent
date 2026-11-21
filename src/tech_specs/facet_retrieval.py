"""Transparent, balanced retrieval for explicitly supplied subquestions.

No automatic planning or relevance verification is performed here.
The same BM25, vector, or hybrid retriever can be used for every facet.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def validate_plan(plan: dict) -> dict:
    """Validate a small JSON-compatible retrieval plan."""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")

    question = plan.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Plan needs a nonempty 'question'")

    document = plan.get("document")
    if document is not None and (
        not isinstance(document, str) or not document.strip()
    ):
        raise ValueError("'document' must be a nonempty string or null")

    facets = plan.get("facets")
    if not isinstance(facets, list) or not facets:
        raise ValueError("Plan needs a nonempty 'facets' array")

    ids = set()
    clean_facets = []
    for i, facet in enumerate(facets, start=1):
        if not isinstance(facet, dict):
            raise ValueError(f"Facet {i} must be an object")
        facet_id = facet.get("id")
        query = facet.get("query")
        if not isinstance(facet_id, str) or not facet_id.strip():
            raise ValueError(f"Facet {i} needs a nonempty 'id'")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Facet {i} needs a nonempty 'query'")
        facet_id = facet_id.strip()
        if facet_id in ids:
            raise ValueError(f"Duplicate facet ID: {facet_id}")
        ids.add(facet_id)
        clean_facets.append({"id": facet_id, "query": query.strip()})

    return {
        "question": question.strip(),
        "document": document.strip() if document is not None else None,
        "facets": clean_facets,
    }


def search_facets(
    retriever: Any,
    plan: dict,
    *,
    top_k: int = 6,
    candidates_per_facet: int = 10,
) -> dict:
    """Retrieve each facet separately, then interleave its candidate rankings.

    Round-robin makes it harder for one facet to consume all available slots.
    Duplicate chunk IDs are emitted once, with every matching facet recorded.
    A retrieved chunk is not automatically a *verified* answer.
    """
    if top_k < 1 or candidates_per_facet < 1:
        raise ValueError("top_k and candidates_per_facet must be >= 1")

    checked = validate_plan(plan)
    result_lists = {}
    membership = defaultdict(dict)

    # Fetch candidates first, so cross-facet memberships are known before
    # choosing the final (possibly short) list.
    for facet in checked["facets"]:
        fid = facet["id"]
        raw_hits = retriever.search(
            facet["query"],
            top_k=candidates_per_facet,
            document=checked["document"],
        )
        unique_hits = []
        seen = set()
        for hit in raw_hits:
            chunk = hit["chunk"]
            cid = chunk["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            unique_hits.append(hit)
            membership[cid][fid] = {
                "rank": len(unique_hits),
                "original_score": hit.get("score"),
            }
        result_lists[fid] = unique_hits

    selected = []
    selected_ids = set()
    for position in range(candidates_per_facet):
        for facet in checked["facets"]:
            hits = result_lists[facet["id"]]
            if position >= len(hits):
                continue
            chunk = hits[position]["chunk"]
            cid = chunk["chunk_id"]
            if cid not in selected_ids:
                selected_ids.add(cid)
                selected.append({
                    "rank": len(selected) + 1,
                    "chunk": chunk,
                    "facet_matches": dict(membership[cid]),
                })
                if len(selected) >= top_k:
                    break
        if len(selected) >= top_k:
            break

    facet_summary = []
    for facet in checked["facets"]:
        fid = facet["id"]
        hits = result_lists[fid]
        chosen = sum(fid in hit["facet_matches"] for hit in selected)
        facet_summary.append({
            "id": fid,
            "query": facet["query"],
            "candidates": len(hits),
            "selected_matches": chosen,
            "first_candidate_id": hits[0]["chunk"]["chunk_id"] if hits else None,
        })

    return {
        "question": checked["question"],
        "document": checked["document"],
        "facets": facet_summary,
        "hits": selected,
    }


def anchor_coverage(result: dict, expected_chunk_ids: list[str]) -> dict:
    """Score only explicitly supplied reference chunk IDs, never answer quality."""
    if not expected_chunk_ids:
        raise ValueError("expected_chunk_ids cannot be empty")
    if len(expected_chunk_ids) != len(set(expected_chunk_ids)):
        raise ValueError("Duplicate reference chunk IDs")
    retrieved = {hit["chunk"]["chunk_id"] for hit in result["hits"]}
    found = [cid for cid in expected_chunk_ids if cid in retrieved]
    missing = [cid for cid in expected_chunk_ids if cid not in retrieved]
    return {
        "found": found,
        "missing": missing,
        "fraction": len(found) / len(expected_chunk_ids),
    }
