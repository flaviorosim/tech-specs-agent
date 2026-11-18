"""Local semantic retrieval over the same chunks indexed by BM25."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
COLLECTION_NAME = "tech_specs_bge_small_en_v15"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "processed" / "qdrant_local"
DEFAULT_MANIFEST = ROOT_DIR / "data" / "processed" / "vector_index_manifest.json"


def point_id(chunk_id: str) -> str:
    """Qdrant point IDs are UUIDs or integers, not arbitrary chunk_id strings."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tech-specs-agent/{chunk_id}"))


def embedding_text(chunk: dict) -> str:
    """Use consistent chunk text for all semantic indexing runs."""
    content = re.sub(r"<br\s*/?>", " ", chunk["content"], flags=re.IGNORECASE)
    section = chunk.get("section") or ""
    # The section provides context to technical tables and short paragraphs.
    return f"Section: {section}\n{content}".strip()


def corpus_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Vector index manifest missing: {path}. Run scripts/index_vectors.py first."
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_index(
    client: QdrantClient,
    manifest: dict,
    *,
    corpus_path: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Stop early if a saved collection does not match the expected configuration."""
    if manifest.get("model_name") != MODEL_NAME:
        raise ValueError("Embedding model differs from index manifest; rebuild the index")
    if manifest.get("collection_name") != collection_name:
        raise ValueError("Collection differs from index manifest; rebuild the index")
    if manifest.get("vector_size") != VECTOR_SIZE:
        raise ValueError("Vector dimension differs from index manifest; rebuild the index")
    if not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection not found: {collection_name}")
    indexed_count = client.count(collection_name, exact=True).count
    if indexed_count != manifest.get("chunks_indexed"):
        raise ValueError(
            f"Index count mismatch: expected {manifest.get('chunks_indexed')}, got {indexed_count}"
        )
    if corpus_path is not None:
        if corpus_sha256(corpus_path) != manifest.get("corpus_sha256"):
            raise ValueError("Corpus has changed since indexing. Rebuild vectors with --rebuild")


class VectorRetriever:
    """Retrieve original chunks from a persisted Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        embedding_model: TextEmbedding,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    def search(
        self,
        query: str,
        top_k: int = 5,
        document: str | None = None,
    ) -> list[dict]:
        """Return the same {'rank', 'score', 'chunk'} shape as BM25Retriever."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query.strip():
            return []

        vectors = self.embedding_model.query_embed(query.strip())
        query_vector = next(iter(vectors)).tolist()
        if len(query_vector) != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE}-D query embedding")

        query_filter = None
        if document is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="document", match=models.MatchValue(value=document)
                )]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "rank": rank,
                "score": round(float(point.score), 6),
                "chunk": point.payload["chunk"],
            }
            for rank, point in enumerate(response.points, start=1)
        ]
