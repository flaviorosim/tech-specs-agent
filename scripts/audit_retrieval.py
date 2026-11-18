
"""Build a review CSV by pooling top-k results from retrieval reports.

This script does not change the corpus or any evaluation reports.
"""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corpus: {path}")

    chunks = {}

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = json.loads(line)
                chunk_id = chunk["chunk_id"]

                if chunk_id in chunks:
                    raise ValueError(
                        f"Duplicate chunk ID: {chunk_id}"
                    )

                chunks[chunk_id] = chunk

    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data/processed/chunks.jsonl",
    )

    parser.add_argument(
        "--bm25",
        type=Path,
        default=ROOT / "data/processed/bm25_scoped_eval.json",
    )

    parser.add_argument(
        "--vector",
        type=Path,
        default=ROOT / "data/processed/vector_scoped_eval.json",
    )

    parser.add_argument(
        "--hybrid",
        type=Path,
        default=ROOT / "data/processed/hybrid_scoped_eval.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/judgments_v2.csv",
    )

    args = parser.parse_args()

    reports = {
        "bm25": read_json(args.bm25),
        "vector": read_json(args.vector),
        "hybrid": read_json(args.hybrid),
    }

    chunks = load_corpus(args.corpus)

    question_sets = {
        name: {
            q["id"]: q
            for q in report["questions"]
        }
        for name, report in reports.items()
    }

    ids = set(question_sets["bm25"])

    if any(
        set(questions) != ids
        for questions in question_sets.values()
    ):
        raise ValueError(
            "Evaluation reports have different question IDs"
        )

    rows = []

    for question_id in sorted(ids):
        base = question_sets["bm25"][question_id]
        candidate_ids = set()

        for method, questions in question_sets.items():
            question = questions[question_id]

            if (
                question["query"] != base["query"]
                or question.get("document")
                != base.get("document")
            ):
                raise ValueError(
                    f"Query or document differs for "
                    f"{question_id} ({method})"
                )

            candidate_ids.update(
                question["retrieved_chunk_ids"]
            )

        for chunk_id in sorted(candidate_ids):
            if chunk_id not in chunks:
                raise ValueError(
                    f"Chunk not in corpus: {chunk_id}"
                )

            chunk = chunks[chunk_id]

            rows.append({
                "question_id": question_id,
                "query": base["query"],
                "expected_document": (
                    base.get("document") or ""
                ),
                "chunk_id": chunk_id,
                "document": chunk["document"],
                "page": chunk["page"],
                "section": chunk.get("section") or "",
                "content_type": chunk["content_type"],
                "content": chunk["content"],
                "relevance": "",
                "notes": "",
            })

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}. "
            "Choose another --output to avoid "
            "overwriting judgments."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else [],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Questions: {len(ids)}")
    print(
        f"Unique question/chunk pairs "
        f"to review: {len(rows)}"
    )
    print(f"Review file: {args.output}")


if __name__ == "__main__":
    main()
