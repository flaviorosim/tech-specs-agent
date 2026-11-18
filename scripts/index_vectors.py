"""Generate FastEmbed passage vectors and persist chunks in local Qdrant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from tech_specs.retrieval import load_chunks
from tech_specs.vector_retrieval import (
    COLLECTION_NAME,
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST,
    MODEL_NAME,
    VECTOR_SIZE,
    corpus_sha256,
    embedding_text,
    point_id,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT_DIR / "data" / "processed" / "chunks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=None, help="For isolated smoke tests")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild this collection")
    args = parser.parse_args()

    if args.batch_size < 1 or (args.max_chunks is not None and args.max_chunks < 1):
        parser.error("Batch size and max-chunks must be positive")
    chunks = load_chunks(args.corpus)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    args.db_path.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(args.db_path))
    try:
        already_exists = client.collection_exists(COLLECTION_NAME)
        if already_exists and not args.rebuild:
            raise RuntimeError(
                "Collection already exists. To replace it, re-run with --rebuild. "
                "Close any other scripts using the local database first."
            )
        if already_exists:
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )

        print(f"Model: {MODEL_NAME} (first run downloads model files)")
        embedder = TextEmbedding(model_name=MODEL_NAME)
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]
            passages = [embedding_text(chunk) for chunk in batch]
            vectors = list(embedder.passage_embed(passages))
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding count differs from chunk count")
            points = []
            for chunk, vector in zip(batch, vectors):
                if len(vector) != VECTOR_SIZE:
                    raise RuntimeError(f"Unexpected vector size for {chunk['chunk_id']}")
                points.append(models.PointStruct(
                    id=point_id(chunk["chunk_id"]),
                    vector=vector.tolist(),
                    payload={"document": chunk["document"], "chunk": chunk},
                ))
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            print(f"Indexed {start + len(batch)}/{len(chunks)} chunks")

        actual = client.count(COLLECTION_NAME, exact=True).count
        if actual != len(chunks):
            raise RuntimeError(f"Point count mismatch: {actual} vs {len(chunks)}")
        manifest = {
            "model_name": MODEL_NAME,
            "collection_name": COLLECTION_NAME,
            "vector_size": VECTOR_SIZE,
            "corpus_sha256": corpus_sha256(args.corpus),
            "chunks_indexed": len(chunks),
            "partial_index": args.max_chunks is not None,
        }
        temp_path = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp_path.replace(args.manifest)
        print(f"Index saved: {args.db_path}")
        print(f"Manifest saved: {args.manifest}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
