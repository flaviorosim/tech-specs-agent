"""Export diverse, previously unused source chunks for a new RAG benchmark.

No search is performed and no relevance labels are invented. The exported JSON
contains verbatim source chunks from the *current* chunks.jsonl snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_JUDGMENTS = ROOT / "evaluation" / "judgments_v2_reviewed.csv"
DEFAULT_OUTPUT = ROOT / "evaluation" / "benchmark_v3_sources.json"


def load_chunks(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = row["chunk_id"]
            if chunk_id in seen:
                raise ValueError(f"Duplicate chunk_id at line {line_number}: {chunk_id}")
            seen.add(chunk_id)
            rows.append(row)
    if not rows:
        raise ValueError("Corpus is empty")
    return rows


def load_previously_judged(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Reviewed judgments not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "chunk_id" not in reader.fieldnames:
            raise ValueError("Reviewed CSV must have a chunk_id column")
        return {r["chunk_id"].strip() for r in reader if r.get("chunk_id")}


def suitable(chunk: dict, excluded: set[str], skip_pages: int, max_chars: int) -> bool:
    if chunk["chunk_id"] in excluded or int(chunk["page"]) <= skip_pages:
        return False
    section = (chunk.get("section") or "").strip()
    content = chunk["content"].strip()
    kind = chunk["content_type"]
    if not section or re.search(r"(?:\.\s*){4,}", section):  # TOC leaders
        return False
    if len(content) > max_chars or len(content) < (100 if kind == "table" else 160):
        return False
    if kind not in {"table", "text"}:
        return False
    if not re.search(r"[A-Za-z]{3,}", content):
        return False
    if kind == "table":
        if chunk.get("metadata", {}).get("review_required"):
            return False
        if re.search(r"\bCol[2-9]\b", content):
            return False
        # At least one header line and one populated data line.
        rows = [s for s in content.splitlines() if s.lstrip().startswith("|")
                and not re.fullmatch(r"[\s|:\-]+", s)]
        if len(rows) < 2:
            return False
    return True


def stable_order(chunk: dict, seed: int) -> str:
    value = f"{seed}/{chunk['chunk_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def pick_diverse(pool: list[dict], amount: int, selected: list[dict],
                 seed: int, page_max: int) -> None:
    """Choose across six page bands, limiting repetition per page/section."""
    buckets = defaultdict(list)
    for chunk in pool:
        band = min(5, (int(chunk["page"]) - 1) * 6 // max(page_max, 1))
        buckets[band].append(chunk)
    for values in buckets.values():
        values.sort(key=lambda c: stable_order(c, seed))

    used = {c["chunk_id"] for c in selected}
    section_counts = Counter(c["section"] for c in selected)
    page_counts = Counter(c["page"] for c in selected)
    target = len(selected) + amount

    # First pass balances sections and pages; second pass fills any shortfall.
    for relaxed in (False, True):
        while len(selected) < target:
            progress = False
            for band in range(6):
                for chunk in buckets[band]:
                    if chunk["chunk_id"] in used:
                        continue
                    if not relaxed and (section_counts[chunk["section"]] >= 2
                                        or page_counts[chunk["page"]] >= 2):
                        continue
                    selected.append(chunk)
                    used.add(chunk["chunk_id"])
                    section_counts[chunk["section"]] += 1
                    page_counts[chunk["page"]] += 1
                    progress = True
                    break
                if len(selected) >= target:
                    break
            if not progress:
                break


def select_sources(chunks: list[dict], excluded: set[str], *,
                   per_document: int = 30, tables_per_document: int = 8,
                   skip_pages: int = 8, max_chars: int = 2500,
                   seed: int = 2026) -> list[dict]:
    if per_document < 1 or not 0 <= tables_per_document <= per_document:
        raise ValueError("Invalid sample quotas")
    if skip_pages < 0 or max_chars < 160:
        raise ValueError("Invalid page or content limits")

    selected = []
    documents = sorted({c["document"] for c in chunks})
    for document in documents:
        all_doc = [c for c in chunks if c["document"] == document]
        page_max = max(int(c["page"]) for c in all_doc)
        pool = [c for c in all_doc if suitable(c, excluded, skip_pages, max_chars)]
        doc_selection = []
        tables = [c for c in pool if c["content_type"] == "table"]
        text = [c for c in pool if c["content_type"] == "text"]
        pick_diverse(tables, tables_per_document, doc_selection, seed, page_max)
        pick_diverse(text, per_document - tables_per_document, doc_selection, seed, page_max)
        # Fill missing table quota with text, or vice versa.
        if len(doc_selection) < per_document:
            pick_diverse(pool, per_document - len(doc_selection), doc_selection, seed, page_max)
        selected.extend(sorted(doc_selection, key=lambda c: (int(c["page"]), c["chunk_id"])))
    return selected


def build_export(chunks: list[dict], selected: list[dict], corpus_path: Path,
                 *, per_document: int, seed: int) -> dict:
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    candidates = []
    for index, chunk in enumerate(selected, 1):
        candidates.append({
            "candidate_id": f"SRC{index:03d}",
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "section": chunk.get("section"),
            "content_type": chunk["content_type"],
            "content": chunk["content"],  # full, not truncated
        })
    return {
        "schema_version": 1,
        "purpose": "new benchmark question drafting; NOT relevance judgments",
        "corpus_sha256": digest,
        "seed": seed,
        "target_per_document": per_document,
        "counts_by_document": dict(Counter(c["document"] for c in candidates)),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-document", type=int, default=30)
    parser.add_argument("--tables-per-document", type=int, default=8)
    parser.add_argument("--skip-pages", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Won't overwrite existing file: {args.output}")
    chunks = load_chunks(args.corpus)
    excluded = load_previously_judged(args.judgments)
    selected = select_sources(
        chunks, excluded,
        per_document=args.per_document,
        tables_per_document=args.tables_per_document,
        skip_pages=args.skip_pages,
        max_chars=args.max_chars,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No eligible chunks; adjust sampling limits")
    result = build_export(chunks, selected, args.corpus,
                          per_document=args.per_document, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Previously judged chunk IDs excluded: {len(excluded)}")
    print(f"New source candidates: {len(selected)}")
    for doc, count in result["counts_by_document"].items():
        print(f"  {doc}: {count}")
    print(f"Corpus SHA256: {result['corpus_sha256'][:16]}...")
    print(f"Saved to: {args.output}")
    print("Send this JSON to ChatGPT to draft 20–30 new grounded questions.")


if __name__ == "__main__":
    main()
