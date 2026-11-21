"""Validate the source-anchored benchmark v3 without installing dependencies.

Place in scripts/validate_benchmark_v3.py and run from the repository root:
    python scripts/validate_benchmark_v3.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_benchmark(sources: dict, dev: list, test: list, answer_key: dict) -> dict:
    """Check IDs, split, source anchors and verbatim evidence."""
    source_items = sources.get("candidates", [])
    if not source_items:
        raise ValueError("Source candidate list is empty")
    source_by_id = {item["candidate_id"]: item for item in source_items}
    if len(source_by_id) != len(source_items):
        raise ValueError("Duplicate source candidate_id")

    records = dev + test
    if len(records) != 30 or len(dev) != 18 or len(test) != 12:
        raise ValueError("Expected 18 development and 12 holdout questions")
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs overlap or repeat")
    if len({item["query"].strip().casefold() for item in records}) != len(records):
        raise ValueError("Queries overlap or repeat")

    answers = answer_key.get("questions", [])
    answer_by_id = {a["id"]: a for a in answers}
    if len(answer_by_id) != len(answers) or set(ids) != set(answer_by_id):
        raise ValueError("Answer-key question IDs do not match query files")
    if answer_key.get("corpus_sha256") != sources.get("corpus_sha256"):
        raise ValueError("Answer key was created for a different corpus snapshot")

    stats = Counter()
    for split, questions in (("dev", dev), ("test", test)):
        for item in questions:
            qid = item["id"]
            answer = answer_by_id[qid]
            refs = answer["source_evidence"]
            anchors = [item["chunk_id"] for item in refs]
            if not refs or not item["relevant_chunk_ids"]:
                raise ValueError(f"{qid}: no source anchors")
            if len(set(anchors)) != len(anchors):
                raise ValueError(f"{qid}: duplicate evidence anchor")
            if (
                item["relevant_chunk_ids"] != anchors
                or item["query"] != answer["query"]
                or item["document"] != answer["document"]
                or split != answer["split"]
            ):
                raise ValueError(f"{qid}: question/answer-key mismatch")
            if not answer.get("reference_answer", "").strip():
                raise ValueError(f"{qid}: missing reference answer")
            for ref in refs:
                source_id = ref["candidate_id"]
                if source_id not in source_by_id:
                    raise ValueError(f"{qid}: missing source {source_id}")
                source = source_by_id[source_id]
                for key in ("chunk_id", "document", "page", "section", "content_type"):
                    if ref[key] != source[key]:
                        raise ValueError(f"{qid}: {key} mismatch for {source_id}")
                if source["document"] != item["document"]:
                    raise ValueError(f"{qid}: cross-document anchor")
                if not ref.get("verbatim_excerpts"):
                    raise ValueError(f"{qid}: missing evidence excerpt")
                for excerpt in ref["verbatim_excerpts"]:
                    if excerpt not in source["content"]:
                        raise ValueError(f"{qid}: evidence not verbatim in {source_id}: {excerpt!r}")
            stats[(split, item["document"])] += 1
    for split, amount in (("dev", 9), ("test", 6)):
        for document in ("attiny85", "stm32l432kc"):
            if stats[(split, document)] != amount:
                raise ValueError(f"Incorrect balance: {split}/{document}")
    return {
        "questions": len(records),
        "development_questions": len(dev),
        "holdout_questions": len(test),
        "source_anchors": sum(len(q["relevant_chunk_ids"]) for q in records),
        "multi_source_questions": sum(len(q["relevant_chunk_ids"]) > 1 for q in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "evaluation/benchmark_v3_sources.json")
    parser.add_argument("--dev", type=Path, default=ROOT / "evaluation/questions_v3_dev.json")
    parser.add_argument("--test", type=Path, default=ROOT / "evaluation/questions_v3_test.json")
    parser.add_argument("--answers", type=Path, default=ROOT / "evaluation/benchmark_v3_answer_key.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/chunks.jsonl")
    parser.add_argument("--skip-corpus-hash", action="store_true", help="Validate against exported sources only")
    args = parser.parse_args()

    sources = read_json(args.sources)
    dev = read_json(args.dev)
    test = read_json(args.test)
    answers = read_json(args.answers)
    report = validate_benchmark(sources, dev, test, answers)

    if not args.skip_corpus_hash:
        if not args.corpus.is_file():
            raise FileNotFoundError(f"Corpus missing: {args.corpus}")
        actual = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
        if actual != sources["corpus_sha256"]:
            raise ValueError("Corpus changed since source sampling; review IDs and regenerate benchmark")

    print("Benchmark v3 source validation: PASS")
    for name, value in report.items():
        print(f"  {name}: {value}")
    print("Note: extracted-chunk evidence verified; original manufacturer PDFs not independently checked.")


if __name__ == "__main__":
    main()
