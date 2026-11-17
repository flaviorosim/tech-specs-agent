
"""Create retrieval chunks from structured PDF elements."""

import re


TABLE_CAPTION_PATTERN = re.compile(
    r"^Table\s+\d+[A-Za-z]?(?:[.:-]|\s)",
    re.IGNORECASE,
)


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split long text at whitespace whenever possible."""

    remaining = text.strip()
    pieces = []

    while len(remaining) > max_chars:
        window = remaining[:max_chars + 1]

        break_at = max(
            window.rfind("\n"),
            window.rfind(" "),
        )

        # Fallback for unusually long strings
        # without convenient whitespace.
        if break_at < max_chars // 2:
            break_at = max_chars

        pieces.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def union_bbox(elements: list[dict]) -> list[float]:
    """Calculate the bounding box enclosing all elements."""

    boxes = [element["bbox"] for element in elements]

    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def chunk_page_elements(
    elements: list[dict],
    max_chars: int = 1200,
) -> list[dict]:
    """
    Chunk elements from a single document page.

    Rules:
    - Do not combine different sections.
    - Preserve complete tables.
    - Attach nearby table captions.
    - Split oversized text at whitespace.
    """

    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")

    if not elements:
        return []

    # This version processes one page at a time.
    origin = {
        (element["document"], element["page"])
        for element in elements
    }

    if len(origin) != 1:
        raise ValueError(
            "Pass elements from a single document page"
        )

    document, page = next(iter(origin))

    chunks = []
    buffer = []
    active_section = None

    def emit(
        content_type: str,
        parts: list[tuple[dict, str]],
        metadata: dict | None = None,
    ):
        """Create a chunk from one or more elements."""

        if not parts:
            return

        content = "\n\n".join(
            text for _, text in parts
        )

        source_elements = [
            element for element, _ in parts
        ]

        # Remove repeated source IDs while keeping order.
        element_ids = list(dict.fromkeys(
            element["element_id"]
            for element in source_elements
        ))

        chunks.append({
            "chunk_id": (
                f"{document}_p{page:03d}_"
                f"c{len(chunks) + 1:03d}"
            ),
            "document": document,
            "page": page,
            "section": source_elements[0].get("section"),
            "content_type": content_type,
            "content": content,
            "bbox": union_bbox(source_elements),
            "element_ids": element_ids,
            "metadata": {
                "oversized": len(content) > max_chars,
                **(metadata or {}),
            },
        })

    def flush_text():
        """Save pending text as a chunk."""
        nonlocal buffer

        emit("text", buffer)
        buffer = []

    index = 0

    while index < len(elements):
        element = elements[index]
        section = element.get("section")

        # Do not mix elements from different sections.
        if section != active_section:
            flush_text()
            active_section = section

        # Check whether a text element is the caption
        # immediately preceding a detected table.
        if (
            element["content_type"] == "text"
            and TABLE_CAPTION_PATTERN.match(
                element["content"].strip()
            )
            and index + 1 < len(elements)
        ):
            candidate = elements[index + 1]

            vertical_gap = (
                candidate["bbox"][1]
                - element["bbox"][3]
            )

            if (
                candidate["content_type"] == "table"
                and candidate.get("section") == section
                and 0 <= vertical_gap <= 35
            ):
                flush_text()

                emit(
                    "table",
                    [
                        (element, element["content"].strip()),
                        (candidate, candidate["content"].strip()),
                    ],
                    {
                        "caption": element["content"].strip(),
                        "review_required": candidate.get(
                            "metadata", {}
                        ).get("review_required", False),
                        "rows": candidate.get(
                            "metadata", {}
                        ).get("rows"),
                        "columns": candidate.get(
                            "metadata", {}
                        ).get("columns"),
                    },
                )

                index += 2
                continue

        if element["content_type"] == "table":
            flush_text()

            table_metadata = element.get("metadata", {})

            emit(
                "table",
                [(element, element["content"].strip())],
                {
                    "review_required": table_metadata.get(
                        "review_required", False
                    ),
                    "rows": table_metadata.get("rows"),
                    "columns": table_metadata.get("columns"),
                },
            )

        elif element["content_type"] == "text":
            parts = split_long_text(
                element["content"],
                max_chars,
            )

            for part in parts:
                current_length = sum(
                    len(text)
                    for _, text in buffer
                )

                if buffer:
                    current_length += 2 * (len(buffer) - 1)

                if (
                    buffer
                    and current_length + 2 + len(part) > max_chars
                ):
                    flush_text()

                buffer.append((element, part))

        # Other content types are not yet indexed.
        index += 1

    # Save the last pending text chunk.
    flush_text()

    return chunks
