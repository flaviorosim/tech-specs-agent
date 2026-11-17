
import pytest

from tech_specs.chunking import chunk_page_elements


def make_element(
    index,
    content,
    content_type="text",
    section="3.21 Timers and watchdogs",
    y=100,
    page=38,
):
    """Build a sample structured element."""

    return {
        "element_id": f"sample_p{page:03d}_e{index:03d}",
        "document": "sample",
        "page": page,
        "section": section,
        "content_type": content_type,
        "content": content,
        "bbox": [60, y, 500, y + 10],
        "metadata": {},
    }


def test_does_not_mix_sections():
    elements = [
        make_element(
            1, "Random number generator",
            section="3.20 RNG",
        ),
        make_element(
            2, "RNG description",
            section="3.20 RNG",
        ),
        make_element(
            3, "TIM2 description",
            section="3.21 Timers",
        ),
    ]

    chunks = chunk_page_elements(elements)

    assert len(chunks) == 2
    assert chunks[0]["section"] == "3.20 RNG"
    assert chunks[1]["section"] == "3.21 Timers"

    assert "TIM2" not in chunks[0]["content"]


def test_table_caption_is_preserved():
    elements = [
        make_element(
            1,
            "Table 9. Timer feature comparison",
            y=100,
        ),
        make_element(
            2,
            "|Timer|DMA|\n|---|---|\n|TIM2|Yes|",
            content_type="table",
            y=125,
        ),
    ]

    chunks = chunk_page_elements(elements)

    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "table"

    assert "Table 9." in chunks[0]["content"]
    assert "TIM2" in chunks[0]["content"]

    assert len(chunks[0]["element_ids"]) == 2


def test_long_text_is_split():
    text = " ".join(["datasheet"] * 400)

    elements = [
        make_element(1, text),
    ]

    chunks = chunk_page_elements(
        elements,
        max_chars=200,
    )

    assert len(chunks) > 1

    assert all(
        len(chunk["content"]) <= 200
        for chunk in chunks
    )

    # All words must survive the splitting.
    reconstructed = " ".join(
        chunk["content"]
        for chunk in chunks
    )

    assert reconstructed == text


def test_oversized_table_is_preserved():
    table_content = "|Parameter|Value|\n" * 100

    elements = [
        make_element(
            1,
            table_content,
            content_type="table",
        ),
    ]

    chunks = chunk_page_elements(
        elements,
        max_chars=200,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == table_content.strip()
    assert chunks[0]["metadata"]["oversized"]


def test_different_pages_are_rejected():
    elements = [
        make_element(1, "Page 38"),
        make_element(2, "Page 39", page=39),
    ]

    with pytest.raises(ValueError):
        chunk_page_elements(elements)
