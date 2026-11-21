"""Search independently supplied evidence facets using existing retrievers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_specs.facet_retrieval import (
    anchor_coverage,
    search_facets,
    validate_plan,
)
from tech_specs.retrieval import BM25Retriever, load_chunks

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--method", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates-per-facet", type=int, default=10)
    parser.add_argument("--corpus", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--db-path", type=Path, default=PROCESSED / "qdrant_local")
    parser.add_argument("--manifest", type=Path, default=PROCESSED / "vector_index_manifest.json")
    parser.add_argument("--expected", action="append", default=[], metavar="CHUNK_ID")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--full", action="store_true", help="Show full chunk content")
    args = parser.parse_args()

    if args.top_k < 1 or args.candidates_per_facet < 1:
        parser.error("--top-k and --candidates-per-facet must be >= 1")

    plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8-sig")))
    chunks = load_chunks(args.corpus)
    client = None
    try:
        bm25 = BM25Retriever(chunks) if args.method in {"bm25", "hybrid"} else None
        if args.method == "bm25":
            searcher = bm25
        else:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from tech_specs.vector_retrieval import (
                MODEL_NAME, VectorRetriever, load_manifest, verify_index,
            )
            client = QdrantClient(path=str(args.db_path))
            manifest = load_manifest(args.manifest)
            if manifest.get("partial_index"):
                raise ValueError("Vector collection is a partial smoke-test index")
            verify_index(client, manifest, corpus_path=args.corpus)
            vector = VectorRetriever(client, TextEmbedding(model_name=MODEL_NAME))
            if args.method == "vector":
                searcher = vector
            else:
                from tech_specs.hybrid_retrieval import HybridRetriever
                searcher = HybridRetriever(bm25, vector)

        result = search_facets(
            searcher,
            plan,
            top_k=args.top_k,
            candidates_per_facet=args.candidates_per_facet,
        )

        print(f"Question: {result['question']}")
        print(f"Device: {result['document'] or '(all)'} | Method: {args.method}")
        print("NOTE: The facets are supplied in the plan; they are not generated automatically.")
        print("\nPer-facet candidates:")
        for facet in result["facets"]:
            print(
                f"  {facet['id']}: candidates={facet['candidates']}, "
                f"selected={facet['selected_matches']}, "
                f"first={facet['first_candidate_id']}"
            )

        print("\nBalanced results:")
        for hit in result["hits"]:
            chunk = hit["chunk"]
            matches = ", ".join(
                f"{fid}=#{match['rank']}"
                for fid, match in hit["facet_matches"].items()
            )
            print("\n" + "=" * 64)
            print(f"#{hit['rank']} | {chunk['chunk_id']} | {matches}")
            print(
                f"Document: {chunk['document']} | page: {chunk['page']} "
                f"| section: {chunk.get('section')}"
            )
            print(chunk["content"] if args.full else chunk["content"][:400])

        if args.expected:
            evidence = anchor_coverage(result, args.expected)
            result["expected_anchor_coverage"] = evidence
            print(
                "\nReference anchor coverage (not verified answer completeness): "
                f"{len(evidence['found'])}/{len(args.expected)}"
            )
            if evidence["missing"]:
                print("Missing: " + ", ".join(evidence["missing"]))

        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(f"Refusing to overwrite: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nReport saved to: {args.output}")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
