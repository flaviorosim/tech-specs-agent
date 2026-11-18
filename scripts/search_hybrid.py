
"""Search technical PDFs using BM25 + vector RRF."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from tech_specs.hybrid_retrieval import HybridRetriever
from tech_specs.retrieval import BM25Retriever, load_chunks
from tech_specs.vector_retrieval import (
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST,
    MODEL_NAME,
    VectorRetriever,
    load_manifest,
    verify_index,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CORPUS = (
    ROOT_DIR / "data" / "processed" / "chunks.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument(
        "--document",
        choices=["stm32l432kc", "attiny85"],
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument("--full", action="store_true")

    args = parser.parse_args()

    if min(
        args.top_k, args.candidate_k, args.rrf_k
    ) < 1:
        parser.error(
            "top-k, candidate-k and rrf-k must be >= 1"
        )

    chunks = load_chunks(args.corpus)

    manifest = load_manifest(args.manifest)

    if manifest.get("partial_index"):
        raise ValueError(
            "This is a partial index; "
            "rebuild the complete index first"
        )

    client = QdrantClient(
        path=str(args.db_path)
    )

    try:
        verify_index(
            client,
            manifest,
            corpus_path=args.corpus,
        )

        searcher = HybridRetriever(
            BM25Retriever(chunks),
            VectorRetriever(
                client,
                TextEmbedding(model_name=MODEL_NAME),
            ),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )

        hits = searcher.search(
            args.query,
            top_k=args.top_k,
            document=args.document,
        )

        print(
            f"Query: {args.query!r} "
            f"| Results: {len(hits)}"
        )

        for hit in hits:
            chunk = hit["chunk"]

            source_ranks = ", ".join(
                f"{name}=#{data['rank']}"
                for name, data in sorted(
                    hit["sources"].items()
                )
            )

            print("\n" + "=" * 64)

            print(
                f"#{hit['rank']} "
                f"| RRF: {hit['score']:.6f} "
                f"| {source_ranks}"
            )

            print(
                f"Chunk: {chunk['chunk_id']} "
                f"| {chunk['content_type']}"
            )

            print(
                f"Document: {chunk['document']} "
                f"| PDF page: {chunk['page']}"
            )

            print(
                f"Section: {chunk.get('section')}"
            )

            print("-" * 64)

            if args.full:
                print(chunk["content"])
            else:
                print(chunk["content"][:500])

    finally:
        client.close()


if __name__ == "__main__":
    main()
