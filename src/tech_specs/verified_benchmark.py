"""Resumable benchmark for the *guarded* v5 Tech-Specs Agent.

This is NOT a correctness grader. Human reference answers/anchors are joined
only AFTER each graph invocation, never passed to planning, retrieval or LLM.
The 12-question holdout is excluded by load_dev_set's split validation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable


DISPLAYABLE = frozenset({
    "source_table_checked", "local_checks_passed", "coverage_review_recommended",
})


def _unique_ordered(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _hits(report: dict) -> list[dict]:
    retrieval = (report.get("working_result") or {}).get("retrieval") or {}
    return retrieval.get("hits", []) if isinstance(retrieval.get("hits", []), list) else []


def _citations(report: dict) -> list[str]:
    ccheck = (report.get("working_result") or {}).get("citation_check") or {}
    return ccheck.get("cited_chunk_ids", []) if isinstance(ccheck.get("cited_chunk_ids", []), list) else []


def make_verified_row(question: dict, report: dict | None, error: str = "") -> dict[str, Any]:
    """One manual-review row; never turn local checks into an accuracy label."""
    rep = report or {}
    work = rep.get("working_result") or {}
    original = rep.get("original_result") or {}
    audits = rep.get("audits") or []
    statuses = [a.get("audit_status", "") for a in audits]
    last_counts = (audits[-1].get("counts") or {}) if audits else {}
    coverages = [c.get("verdict", "") for c in rep.get("coverage_checks", [])]
    table = rep.get("source_cell_check") or {}
    cited = _unique_ordered(_citations(rep))
    found = _unique_ordered([
        hit.get("chunk", {}).get("chunk_id", "")
        for hit in _hits(rep)
        if isinstance(hit, dict) and isinstance(hit.get("chunk"), dict)
    ])
    gold = question.get("anchors", [])
    status = rep.get("status", "execution_error" if error else "not_run")
    displayed = status in DISPLAYABLE and (status != "source_table_checked" or table.get("accepted") is True)
    # A coverage warning must remain distinct from a fully checked answer.
    return {
        "question_id": question["id"],
        "document": question["document"],
        "category": question.get("category", ""),
        "status": status,
        "visible_answer": displayed,
        "source_cell_checked": status == "source_table_checked" and table.get("accepted") is True,
        "answer_strategy_initial": rep.get("initial_answer_strategy", ""),
        "answer_strategy_final": rep.get("final_answer_strategy", ""),
        "repair_attempts": rep.get("repairs", 0),
        "latest_claim_audit": statuses[-1] if statuses else "not_run",
        "claims_supported_by_judge": last_counts.get("supported", ""),
        "claims_unsupported_by_judge": last_counts.get("unsupported", ""),
        "claims_uncertain_by_judge": last_counts.get("uncertain", ""),
        "source_cell_kind": table.get("kind", ""),
        "source_cells_checked": len(table.get("cells") or []),
        "latest_coverage": coverages[-1] if coverages else "not_run",
        "table_conflicts": len((rep.get("table_checks") or [{}])[-1].get("issues", [])),
        "gold_anchors_retrieved": len(set(found) & set(gold)),
        "gold_anchors_cited": len(set(cited) & set(gold)),
        "gold_anchors_total": len(gold),
        "cited_source_ids": "; ".join(cited),
        "question": question["query"],
        "reference_answer": question["reference_answer"],
        "generated_answer_for_review": work.get("draft", ""),
        "answer_displayed": work.get("draft", "") if displayed else "",
        "error": error or rep.get("error_detail", ""),
        "manual_correct": "",
        "manual_complete": "",
        "review_notes": "",
    }


def write_review_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty review")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    counts = {}
    for row in rows:
        key = row["status"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "processed": len(rows),
        "status_counts": counts,
        "source_table_checked": counts.get("source_table_checked", 0),
        "local_checks_passed": counts.get("local_checks_passed", 0),
        "coverage_review_recommended": counts.get("coverage_review_recommended", 0),
        "withheld": sum(1 for r in rows if not r["visible_answer"]),
        "execution_errors": sum(1 for r in rows if r["error"] and r["status"] == "execution_error"),
        "repairs_total": sum(int(r["repair_attempts"] or 0) for r in rows),
        "accuracy_evaluated": False,
    }


def run_verified_batch(
    questions: list[dict],
    invoke: Callable[[str, str], dict],
    output_dir: Path,
    *,
    limit: int | None = None,
) -> list[dict]:
    """Run or resume, saving each graph state exclusively. No silent reruns.

    A failed invoke is recorded in review.csv, not given a fake answer file.
    Existing graph state is reused only when question and device match exactly.
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    selected = questions[:limit] if limit is not None else questions
    rows = []
    for idx, q in enumerate(selected, 1):
        path = folder / (q["id"] + ".json")
        report, error = None, ""
        try:
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
            else:
                # Never send gold anchors, reference answers or grades to graph.
                report = invoke(q["query"], q["document"])
            if not isinstance(report, dict):
                raise ValueError("Graph report must be a JSON object")
            if report.get("question") != q["query"]:
                raise ValueError("Question in graph report does not match benchmark")
            plan = (report.get("original_result") or {}).get("plan") or {}
            if plan.get("document") != q["document"]:
                raise ValueError("Document in graph report does not match benchmark")
            if not isinstance(report.get("status"), str):
                raise ValueError("Graph status is missing")
            if (report["status"] == "source_table_checked"
                    and (report.get("source_cell_check") or {}).get("accepted") is not True):
                raise ValueError("Source-table status lacks accepted source-cell check")
            if not path.exists():
                # Saving only after validation; exclusive open protects past runs.
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(report, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            # Keep batch moving; preserve the error and never overwrite a prior file.
            error = f"{type(exc).__name__}: {exc}"
            report = None
        row = make_verified_row(q, report, error)
        rows.append(row)
        print(
            f"[{idx}/{len(selected)}] {q['id']}: {row['status']} "
            f"| strategy={row['answer_strategy_final'] or '-'}"
            + (f" | ERROR={error}" if error else ""),
            flush=True,
        )
        write_review_csv(rows, folder / "review.csv")
        (folder / "summary.json").write_text(
            json.dumps(summarize(rows), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return rows
