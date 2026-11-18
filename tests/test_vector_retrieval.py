"""Unit tests using synthetic vectors and a real local in-memory Qdrant index."""

import json
import math

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
pytest.importorskip("fastembed")

from qdrant_client import QdrantClient, models
from tech_specs.vector_retrieval import (
    COLLECTION_NAME, VECTOR_SIZE, VectorRetriever, corpus_sha256,
    embedding_text, point_id, verify_index, MODEL_NAME,
)


class StubEmbeddings:
    def query_embed(self, query):
        # No model download needed: a valid 384-D vector.
        return [FakeVector([1.0] + [0.0] * (VECTOR_SIZE - 1))]


class FakeVector(list):
    def tolist(self):
        return list(self)


def test_point_ids_stable():
    assert point_id("abc") == point_id("abc")
    assert point_id("abc") != point_id("def")


def test_embedding_text_and_hash(tmp_path):
    data = {"content": "|TIM2|32<br>bit|", "section": "3.21 Timers"}
    assert embedding_text(data) == "Section: 3.21 Timers\n|TIM2|32 bit|"
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("first", encoding="utf-8")
    old_hash = corpus_sha256(corpus)
    corpus.write_text("changed", encoding="utf-8")
    assert corpus_sha256(corpus) != old_hash


def test_vector_search_filter_and_manifest(tmp_path):
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        chunks = [
            {"chunk_id": "c_stm", "document": "stm32l432kc", "page": 38,
             "section": "3.21", "content_type": "text", "content": "TIM2 resolution"},
            {"chunk_id": "c_att", "document": "attiny85", "page": 9,
             "section": "4.4", "content_type": "text", "content": "ATtiny SREG"},
        ]
        client.upsert(collection_name=COLLECTION_NAME, wait=True, points=[
            models.PointStruct(
                id=point_id(chunk["chunk_id"]),
                vector=[1.0, 0.0] + [0.0] * (VECTOR_SIZE - 2) if i == 0 else
                       [0.0, 1.0] + [0.0] * (VECTOR_SIZE - 2),
                payload={"document": chunk["document"], "chunk": chunk},
            ) for i, chunk in enumerate(chunks)
        ])
        manifest = {"model_name": MODEL_NAME, "collection_name": COLLECTION_NAME,
                    "vector_size": VECTOR_SIZE, "chunks_indexed": 2}
        verify_index(client, manifest)
        retriever = VectorRetriever(client, StubEmbeddings())
        assert retriever.search("", top_k=5) == []
        with pytest.raises(ValueError):
            retriever.search("TIM2", top_k=0)
        hits = retriever.search("TIM2", top_k=2)
        assert hits[0]["chunk"]["chunk_id"] == "c_stm"
        assert hits[0]["rank"] == 1
        filtered = retriever.search("TIM2", document="attiny85")
        assert len(filtered) == 1
        assert filtered[0]["chunk"]["document"] == "attiny85"
        with pytest.raises(ValueError):
            verify_index(client, {**manifest, "chunks_indexed": 4})
    finally:
        client.close()
