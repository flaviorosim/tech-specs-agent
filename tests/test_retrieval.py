"""Unit tests for JSONL loading and technical BM25 retrieval."""

import json

import pytest

from tech_specs.retrieval import BM25Retriever, load_chunks, tokenize


def sample_chunks():
    records = [
        ("stm32l432kc", "TIM2", "TIM2 32-bit timer has DMA requests"),
        ("attiny85", "SREG", "SREG is at address 0x3F with eight flags"),
        ("attiny85", "R26", "R26 uses address 0x1A X-register Low Byte"),
        ("stm32l432kc", "RNG", "Random number generator produces random values"),
        ("attiny85", "ADC", "Analog to digital converter input channels"),
    ]
    return [
        {
            "chunk_id": f"chunk_{index}",
            "document": document,
            "page": index,
            "section": label,
            "content_type": "text",
            "content": content,
            "metadata": {},
        }
        for index, (document, label, content) in enumerate(records, start=1)
    ]


def test_tokenizer_preserves_technical_identifiers():
    tokens = tokenize("TIM2, 0x3F, R/W, 16-bit GPIOA_MODER and 32-bit<br>counter")
    assert "tim2" in tokens
    assert "0x3f" in tokens
    assert "r/w" in tokens
    assert "16-bit" in tokens
    assert "gpioa_moder" in tokens
    assert "32-bit" in tokens
    assert "br" not in tokens


def test_exact_identifier_search():
    engine = BM25Retriever(sample_chunks())
    assert engine.search("TIM2", top_k=1)[0]["chunk"]["chunk_id"] == "chunk_1"
    assert engine.search("0x3F", top_k=1)[0]["chunk"]["chunk_id"] == "chunk_2"
    assert engine.search("R26 0x1A", top_k=1)[0]["chunk"]["chunk_id"] == "chunk_3"


def test_document_filter_and_missing_terms():
    engine = BM25Retriever(sample_chunks())
    hits = engine.search("TIM2", document="attiny85")
    assert hits == []
    assert engine.search("completely_unknown_identifier_zxy") == []
    assert engine.search("!!!") == []
    with pytest.raises(ValueError):
        engine.search("TIM2", top_k=0)


def test_jsonl_loading_and_duplicate_detection(tmp_path):
    path = tmp_path / "chunks.jsonl"
    record = sample_chunks()[0]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_chunks(path)[0]["chunk_id"] == "chunk_1"

    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        load_chunks(path)
