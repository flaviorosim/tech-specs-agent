"""Reproducible answer-level evaluation helpers for the v3 development set.

The reference answer and gold anchors are used ONLY after graph invocation.
They never modify planning, retrieval, model prompting, or citation validation.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable


ID_PATTERN = re.compile(r"^q_v3_[a-z][0-9]{2}$")


def load_dev_set(questions_path: Path, key_path: Path) -> list[dict]:
    """Validate question/key consistency; refuse holdout or non-dev questions."""
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a nonempty questions JSON list")
    entries = key.get("questions")
    if not isinstance(entries, list):
        raise ValueError("Answer key must contain a questions list")
    keyed = {item["id"]: item for item in entries}
    if len(keyed) != len(entries):
        raise ValueError("Duplicate IDs in answer key")
    seen = set()
    rows = []
    for item in questions:
        qid, query, document = item.get("id"), item.get("query"), item.get("document")
        if not isinstance(qid, str) or not ID_PATTERN.fullmatch(qid) or qid in seen:
            raise ValueError(f"Duplicate or invalid question ID: {qid!r}")
        seen.add(qid)
        if qid not in keyed or keyed[qid].get("split") != "dev":
            raise ValueError(f"Refusing non-development question: {qid}")
        if not isinstance(query, str) or not query.strip() or document not in ("attiny85", "stm32l432kc"):
            raise ValueError(f"Invalid question or document: {qid}")
        gold = keyed[qid]
        if gold.get("query") != query or gold.get("document") != document:
            raise ValueError(f"Question/key mismatch: {qid}")
        anchors = item.get("relevant_chunk_ids")
        if not isinstance(anchors, list) or not anchors or any(not isinstance(x, str) for x in anchors):
            raise ValueError(f"Missing gold anchors: {qid}")
        if len(set(anchors)) != len(anchors):
            raise ValueError(f"Duplicate gold anchors: {qid}")
        rows.append({
            "id": qid, "query": query, "document": document,
            "category": item.get("category", ""),
            "anchors": anchors,
            "reference_answer": gold["reference_answer"],
            "pdf_review_recommended": bool(gold.get("pdf_review_recommended")),
        })
    return rows


def make_row(question: dict, report: dict | None, audit: dict | None = None, error: str = "") -> dict:
    """One question/answer pair for human review, not an automatic accuracy verdict."""
    report = report or {}
    chunks = {
        hit["chunk"]["chunk_id"]
        for hit in report.get("retrieval", {}).get("hits", [])
    }
    cited = set(report.get("citation_check", {}).get("cited_chunk_ids", []))
    anchors = set(question["anchors"])
    counts = audit.get("counts", {}) if audit else {}
    draft = report.get("draft", "")
    return {
        "question_id": question["id"],
        "document": question["document"],
        "category": question["category"],
        "status": report.get("status", "execution_error" if error else "not_run"),
        "audit_status": audit.get("audit_status", "not_run") if audit else "not_run",
        "gold_anchors_retrieved": len(chunks & anchors),
        "gold_anchors_cited": len(cited & anchors),
        "gold_anchors_total": len(anchors),
        "cited_source_count": len(cited),
        "claims_supported_by_judge": counts.get("supported", ""),
        "claims_unsupported_by_judge": counts.get("unsupported", ""),
        "claims_uncertain_by_judge": counts.get("uncertain", ""),
        "incomplete_answer_declared": bool(
            "Information not established by the supplied evidence:" in draft
        ),
        "pdf_review_recommended": question["pdf_review_recommended"],
        "question": question["query"],
        "reference_answer": question["reference_answer"],
        "generated_answer": draft,
        "error": error,
        "manual_correct": "",  # fill YES/NO/PARTIAL after review
        "manual_complete": "",  # fill YES/NO after review
        "review_notes": "",
    }


def write_review_csv(rows: list[dict], path: Path) -> None:
    """Excel-friendly UTF-8 BOM; format with Data > From Text/CSV if necessary."""
    if not rows:
        raise ValueError("Cannot export empty review")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_run_config(output_dir: Path, config: dict) -> None:
    """Prevent accidental reuse of reports with different experimental settings."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "_run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                "Existing output directory belongs to a different run/configuration. "
                "Choose a fresh --output-dir instead."
            )
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def run_batch(
    questions: list[dict],
    invoke: Callable[[str, str], dict],
    output_dir: Path,
    *,
    judge: Any | None = None,
    audit_fn: Callable[[dict, Any], dict] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Incrementally persist reports, audit when possible, resume without redoing successes.

    invoke(question, document) must NOT receive reference answers/anchor IDs.
    Failures are captured in the review CSV; unsuccessful generations are not treated
    as successful answers. Existing per-question JSON files are never overwritten.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if judge is not None and audit_fn is None:
        raise ValueError("judge requires audit_fn")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    selected = questions[:limit] if limit else questions
    rows = []
    for index, question in enumerate(selected, start=1):
        qid = question["id"]
        path = folder / f"{qid}.json"
        audit_path = folder / f"{qid}_faithfulness.json"
        error = ""
        report = None
        audit = None
        try:
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("question") != question["query"]:
                    raise ValueError("Existing report has different question; refusing reuse")
                if report.get("plan", {}).get("document") != question["document"]:
                    raise ValueError("Existing report targets a different document")
            else:
                report = invoke(question["query"], question["document"])
                if report.get("question") != question["query"]:
                    raise ValueError("Graph returned a mismatched question")
                if report.get("plan", {}).get("document") != question["document"]:
                    raise ValueError("Graph returned a mismatched document")
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if judge is not None and report.get("status") == "citations_checked":
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    if audit.get("question") != question["query"]:
                        raise ValueError("Existing audit has a different question")
                else:
                    audit = audit_fn(report, judge)
                    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        except (RuntimeError, ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = make_row(question, report, audit, error)
        rows.append(row)
        print(f"[{index}/{len(selected)}] {qid}: {row['status']} | audit={row['audit_status']}" +
              (f" | ERROR: {error}" if error else ""), flush=True)
        write_review_csv(rows, folder / "review.csv")
    return rows
