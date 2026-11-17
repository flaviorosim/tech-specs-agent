
import pymupdf

from tech_specs.structure import (
    build_page_elements,
    is_footer,
)


def test_footer_detection():
    """Known footer patterns near the bottom must be removed."""

    page_height = 842

    assert is_footer(
        "38/156\nDS11451 Rev 4",
        [67, 743, 329, 754],
        page_height,
    )

    assert is_footer(
        "9 ATtiny25/45/85 [DATASHEET]",
        [404, 738, 576, 750],
        page_height,
    )

    # The same pattern in the main content is not a footer.
    assert not is_footer(
        "38/156 DS11451 Rev 4",
        [67, 100, 329, 120],
        page_height,
    )

    # Ordinary text near the bottom is not a footer.
    assert not is_footer(
        "Important electrical characteristics",
        [67, 740, 329, 752],
        page_height,
    )


def test_table_text_is_not_duplicated():
    """Words inside a valid table should not appear as text."""

    with pymupdf.open() as document:

        page = document.new_page(
            width=595,
            height=842,
        )

        # Build a small synthetic document.
        page.insert_text(
            (72, 100),
            "Table 9. Timer comparison",
        )

        page.insert_text(
            (72, 160),
            "TIM2 32-bit Yes",
        )

        page.insert_text(
            (72, 210),
            "Explanation text outside table.",
        )

        tables = [{
            "bbox": [65, 140, 510, 180],
            "rows": 2,
            "columns": 3,
            "markdown": (
                "|Timer|Resolution|DMA|\n"
                "|---|---|---|\n"
                "|TIM2|32-bit|Yes|"
            ),
        }]

        elements = build_page_elements(
            page=page,
            document_name="sample",
            page_number=1,
            tables=tables,
        )

    text_elements = [
        element for element in elements
        if element["content_type"] == "text"
    ]

    table_elements = [
        element for element in elements
        if element["content_type"] == "table"
    ]

    # Exactly one structured table must be created.
    assert len(table_elements) == 1

    # Table information must remain in the table.
    assert "TIM2" in table_elements[0]["content"]

    # Table information must not be duplicated as text.
    assert all(
        "TIM2" not in element["content"]
        for element in text_elements
    )

    # Text outside the table must be preserved.
    assert any(
        "Explanation text outside table" in element["content"]
        for element in text_elements
    )

    # The table caption must also be preserved.
    assert any(
        "Table 9. Timer comparison" in element["content"]
        for element in text_elements
    )


def test_incomplete_table_keeps_original_text():
    """A single-row table must not mask its source text."""

    with pymupdf.open() as document:

        page = document.new_page(
            width=595,
            height=842,
        )

        page.insert_text(
            (72, 130),
            "0x3F I T H S V N Z C SREG",
        )

        tables = [{
            "bbox": [110, 110, 455, 140],
            "rows": 1,
            "columns": 8,
            "markdown": (
                "|I|T|H|S|V|N|Z|C|\n"
                "|---|---|---|---|---|---|---|---|"
            ),
        }]

        elements = build_page_elements(
            page=page,
            document_name="sample",
            page_number=1,
            tables=tables,
        )

    text_content = "\n".join(
        element["content"]
        for element in elements
        if element["content_type"] == "text"
    )

    assert "0x3F" in text_content
    assert "SREG" in text_content

    table_elements = [
        element for element in elements
        if element["content_type"] == "table"
    ]

    assert len(table_elements) == 1
    assert table_elements[0]["metadata"]["review_required"]
