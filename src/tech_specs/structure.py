
"""Convert PDF pages into structured content elements."""

import re
from collections import defaultdict

import pymupdf


FOOTER_PATTERNS = (
    re.compile(r"\b\d+\s*/\s*\d+\b"),
    re.compile(r"\bDS\d+\s+Rev\s+\d+\b", re.IGNORECASE),
    re.compile(
        r"\bATtiny\d+(?:/\d+)+\s+\[DATASHEET\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{4}[A-Z][–—-]AVR[–—-]\d{2}/\d{4}\b",
        re.IGNORECASE,
    ),
)


def is_footer(
    content: str,
    bbox: list,
    page_height: float,
) -> bool:
    """Identify footer content using position and known patterns."""

    near_bottom = bbox[1] >= page_height * 0.85

    if not near_bottom:
        return False

    has_footer_pattern = any(
        pattern.search(content)
        for pattern in FOOTER_PATTERNS
    )

    return has_footer_pattern


def round_bbox(coords) -> list:
    return [round(float(value), 2) for value in coords]


def build_page_elements(
    page: pymupdf.Page,
    document_name: str,
    page_number: int,
    tables: list[dict],
) -> list[dict]:
    """
    Build ordered text and table elements.

    Avoid duplicate text inside reasonably complete
    tables while preserving content outside their bounds.
    """

    # Do not mask text for potentially incomplete
    # single-row table detections.
    table_masks = [
        pymupdf.Rect(table["bbox"])
        for table in tables
        if table["rows"] >= 2 and table["columns"] >= 2
    ]

    # Group surviving words by PDF block and line.
    lines_by_block = defaultdict(
        lambda: defaultdict(list)
    )

    for word in page.get_text("words", sort=False):
        x0, y0, x1, y1, text, block_no, line_no, word_no = word

        center = pymupdf.Point(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
        )

        # Remove words already represented by a table.
        if any(
            rect.contains(center)
            for rect in table_masks
        ):
            continue

        lines_by_block[block_no][line_no].append(word)

    elements = []

    # Reconstruct text blocks from remaining words.
    for block_no, lines in lines_by_block.items():

        remaining_words = []
        reconstructed_lines = []

        for line_no in sorted(lines):

            line_words = sorted(
                lines[line_no],
                key=lambda word: word[7],
            )

            reconstructed_lines.append(
                " ".join(word[4] for word in line_words)
            )

            remaining_words.extend(line_words)

        content = "\n".join(reconstructed_lines).strip()

        if not content:
            continue

        bbox = round_bbox((
            min(w[0] for w in remaining_words),
            min(w[1] for w in remaining_words),
            max(w[2] for w in remaining_words),
            max(w[3] for w in remaining_words),
        ))

        if is_footer(content, bbox, page.rect.height):
            continue

        elements.append({
            "document": document_name,
            "page": page_number,
            "content_type": "text",
            "content": content,
            "bbox": bbox,
            "metadata": {
                "block_no": block_no,
            },
        })

    # Add detected tables as separate elements.
    for table in tables:

        elements.append({
            "document": document_name,
            "page": page_number,
            "content_type": "table",
            "content": table["markdown"],
            "bbox": table["bbox"],
            "metadata": {
                "rows": table["rows"],
                "columns": table["columns"],
                "review_required": table["rows"] == 1,
                "extraction_strategy": "lines",
            },
        })

    # Basic spatial reading order.
    elements.sort(
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        )
    )

    # Assign stable identifiers for the same extraction.
    for index, element in enumerate(elements, start=1):
        element["element_id"] = (
            f"{document_name}_p{page_number:03d}_e{index:03d}"
        )

    return elements
