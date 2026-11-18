"""Unit tests for corpus quality filtering."""

from tech_specs.corpus import prepare_chunk


def sample_chunk(document="stm32l432kc", content_type="text", content="Example"):
    return {
        "chunk_id": "sample_p001_c001",
        "document": document,
        "page": 1,
        "section": "3.21 Timers and watchdogs",
        "content_type": content_type,
        "content": content,
        "bbox": [10, 20, 30, 40],
        "element_ids": ["sample_p001_e001"],
        "metadata": {"oversized": False},
    }


def test_excludes_review_required_table_by_default():
    chunk = sample_chunk(content_type="table", content="|I|T|H|")
    chunk["metadata"]["review_required"] = True

    assert prepare_chunk(chunk) is None

    included = prepare_chunk(chunk, include_review_tables=True)
    assert included is not None
    assert included["content"] == "|I|T|H|"


def test_preserves_valid_table():
    chunk = sample_chunk(content_type="table", content="|TIM2|32-bit|")
    chunk["metadata"]["review_required"] = False

    assert prepare_chunk(chunk)["content"] == "|TIM2|32-bit|"


def test_cleans_known_stm32_header_without_mutating_original():
    chunk = sample_chunk(
        content="Functional overview\nSTM32L432KB STM32L432KC\n\n"
                "The main features of the touch sensing controller..."
    )

    cleaned = prepare_chunk(chunk)

    assert cleaned["content"].startswith("The main features")
    assert cleaned["metadata"]["header_removed"] is True
    assert chunk["content"].startswith("Functional overview")
    assert "header_removed" not in chunk["metadata"]


def test_does_not_remove_other_document_content():
    chunk = sample_chunk(
        document="attiny85",
        content="Functional overview\nSTM32L432KB STM32L432KC\n\nKeep this",
    )
    assert prepare_chunk(chunk)["content"].startswith("Functional overview")
