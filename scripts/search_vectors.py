"""Search the persisted local Qdrant index with semantic embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from tech_specs.vector_retrieval import (
    DEFAULT_DB_PATH, DEFAULT_MANIFEST, MODEL_NAME,
    VectorRetriever, load_manifest, verify_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language or technical query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document", choices=["stm32l432kc", "attiny85"])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.jsonl",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest.get("partial_index"):
        print("WARNING: This is a partial smoke-test index.")
    client = QdrantClient(path=str(args.db_path))
    try:
        verify_index(client, manifest, corpus_path=args.corpus)
        embedding_model = TextEmbedding(model_name=MODEL_NAME)
        retriever = VectorRetriever(client, embedding_model)
        hits = retriever.search(args.query, top_k=args.top_k, document=args.document)
        print(f"Query: {args.query!r} | Results: {len(hits)}")
        for hit in hits:
            chunk = hit["chunk"]
            print("\n" + "=" * 64)
            print(f"#{hit['rank']} | Cosine similarity: {hit['score']:.4f}")
            print(f"Chunk: {chunk['chunk_id']} | Type: {chunk['content_type']}")
            print(f"Document: {chunk['document']} | PDF page: {chunk['page']}")
            print(f"Section: {chunk.get('section')}")
            print("-" * 64)
            print(chunk["content"] if args.full else chunk["content"][:500])
    finally:
        client.close()


if __name__ == "__main__":
    main()
