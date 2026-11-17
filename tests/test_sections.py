
from tech_specs.sections import (
    annotate_sections,
    detect_section,
)


def make_element(content, content_type="text"):
    """Create a minimal element for testing."""
    return {
        "content_type": content_type,
        "content": content,
        "metadata": {},
    }


def test_heading_detection():

    stm = make_element(
        "3.21\nTimers and watchdogs"
    )

    attiny = make_element(
        "4.4.1\nSREG – AVR Status Register"
    )

    assert detect_section(stm) == {
        "number": "3.21",
        "title": "Timers and watchdogs",
    }

    assert detect_section(attiny) == {
        "number": "4.4.1",
        "title": "SREG – AVR Status Register",
    }

    # These are not section headings.
    assert detect_section(
        make_element("0x3F\nI T H S V N Z C")
    ) is None

    assert detect_section(
        make_element("Table 9. Timer comparison")
    ) is None


def test_section_changes():

    elements = [
        make_element("3.20\nRandom number generator"),
        make_element("RNG generates random numbers"),
        make_element("3.21\nTimers and watchdogs"),
        make_element("TIM2 timer information"),
        make_element(
            "|Timer|Resolution|\n|---|---|\n|TIM2|32-bit|",
            "table",
        ),
    ]

    annotated, last_section = annotate_sections(elements)

    assert annotated[1]["section"] == (
        "3.20 Random number generator"
    )

    assert annotated[3]["section"] == (
        "3.21 Timers and watchdogs"
    )

    assert annotated[4]["section"] == (
        "3.21 Timers and watchdogs"
    )

    assert last_section["number"] == "3.21"

    assert annotated[2]["metadata"]["starts_section"]
    assert not annotated[4]["metadata"]["starts_section"]


def test_section_continues_between_pages():

    page1 = [
        make_element("3.21\nTimers and watchdogs"),
        make_element("First paragraph"),
    ]

    page2 = [
        make_element("Table continuation", "table"),
        make_element("More information about timers"),
    ]

    _, last_section = annotate_sections(page1)

    annotated_page2, _ = annotate_sections(
        page2,
        active_section=last_section,
    )

    assert all(
        element["section"] == "3.21 Timers and watchdogs"
        for element in annotated_page2
    )
