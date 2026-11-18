"""Quality filters applied to chunks before indexing."""

import re


STM32_HEADER = re.compile(
    r"\AFunctional overview\r?\n"
    r"STM32L432KB STM32L432KC(?:\r?\n)+"
)


def prepare_chunk(
    chunk: dict,
    include_review_tables: bool = False,
) -> dict | None:
    """Return a cleaned copy, or None if the chunk must be excluded.

    A review-required table is excluded by default because its extraction
    may be incomplete. Text that came from the same PDF page is unaffected.
    """
    metadata = chunk.get("metadata", {})

    if (
        chunk["content_type"] == "table"
        and metadata.get("review_required", False)
        and not include_review_tables
    ):
        return None

    prepared = {
        **chunk,
        "metadata": dict(metadata),
    }
    content = prepared["content"].strip()

    # A conservative, source-specific filter; not a general header detector.
    if (
        prepared["document"] == "stm32l432kc"
        and prepared["content_type"] == "text"
    ):
        content, removed = STM32_HEADER.subn("", content, count=1)
        if removed:
            prepared["metadata"]["header_removed"] = True

    content = content.strip()
    if not content:
        return None

    prepared["content"] = content
    return prepared
