import re


SECTION_PATTERN = re.compile(
    r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2}){1,4})"
    r"[ \t]*(?:\n[ \t]*|[ \t]+)"
    r"(?P<title>[^\n]+)"
)


def detect_section(element: dict) -> dict | None:
    """
    Detect a numbered section heading at the beginning
    of a text element.
    """

    if element["content_type"] != "text":
        return None

    content = element["content"]
    match = SECTION_PATTERN.match(content)

    if not match:
        return None

    number = match.group("number")
    title = " ".join(match.group("title").split())

    # Reject candidates without a meaningful title.
    if not any(char.isalpha() for char in title):
        return None

    if len(title) > 120:
        return None

    return {
        "number": number,
        "title": title,
    }


def annotate_sections(
    elements: list[dict],
    active_section: dict | None = None,
) -> tuple[list[dict], dict | None]:
    """
    Assign the current section to each element.

    The active section can be carried to the next page.
    """

    current_section = (
        active_section.copy()
        if active_section is not None
        else None
    )

    annotated = []

    for element in elements:

        # Create a copy to avoid modifying original data.
        updated = {
            **element,
            "metadata": dict(element.get("metadata", {})),
        }

        detected = detect_section(updated)

        if detected is not None:
            current_section = detected

        updated["metadata"]["starts_section"] = (
            detected is not None
        )

        if current_section is not None:
            updated["section"] = (
                f"{current_section['number']} "
                f"{current_section['title']}"
            )
        else:
            updated["section"] = None

        annotated.append(updated)

    return annotated, current_section
