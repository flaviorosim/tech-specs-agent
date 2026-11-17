
from pathlib import Path

import pytest

from tech_specs.ingestion import extract_page


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"

# Identify tests that require real PDF files.
pytestmark = pytest.mark.pdf


@pytest.fixture
def pdf_path():
    """Provide local datasheet paths for integration tests."""

    def resolve(document_name: str) -> Path:
        path = RAW_DIR / f"{document_name}.pdf"

        if not path.is_file():
            pytest.skip(
                f"PDF not available locally: {path}"
            )

        return path

    return resolve


def test_stm32_timer_table(pdf_path):
    """STM32 page 38 must preserve the timer comparison table."""

    result = extract_page(
        pdf_path("stm32l432kc"),
        page_number=38,
    )

    elements = result["elements"]

    tables = [
        element for element in elements
        if element["content_type"] == "table"
        and "TIM2" in element["content"]
    ]

    assert len(tables) == 1

    table = tables[0]

    assert "TIM1" in table["content"]
    assert "TIM2" in table["content"]
    assert "TIM15" in table["content"]

    assert table["metadata"]["columns"] == 8
    assert table["metadata"]["rows"] == 4

    # The original table caption must be preserved.
    assert any(
        "Table 9. Timer feature comparison" in element["content"]
        for element in elements
        if element["content_type"] == "text"
    )

    # The timer information must not appear twice.
    assert not any(
        "TIM2" in element["content"]
        for element in elements
        if element["content_type"] == "text"
    )

    # Footer must not appear in structured elements.
    assert not any(
        "38/156" in element["content"]
        for element in elements
    )


def test_attiny_sreg_preservation(pdf_path):
    """ATtiny85 page 9 must preserve SREG information."""

    result = extract_page(
        pdf_path("attiny85"),
        page_number=9,
    )

    elements = result["elements"]

    text_content = "\n".join(
        element["content"]
        for element in elements
        if element["content_type"] == "text"
    )

    # Important register identifiers must survive.
    assert "SREG" in text_content
    assert "0x3F" in text_content
    assert "Global Interrupt Enable" in text_content

    # The incomplete table should be flagged for review.
    tables = [
        element for element in elements
        if element["content_type"] == "table"
    ]

    assert any(
        element["metadata"]["review_required"]
        and "I" in element["content"]
        and "C" in element["content"]
        for element in tables
    )


def test_attiny_register_addresses(pdf_path):
    """ATtiny85 page 10 must preserve register/address pairs."""

    result = extract_page(
        pdf_path("attiny85"),
        page_number=10,
    )

    contents = [
        element["content"]
        for element in result["elements"]
    ]

    expected_registers = [
        ("R26", "0x1A", "X-register Low Byte"),
        ("R27", "0x1B", "X-register High Byte"),
        ("R28", "0x1C", "Y-register Low Byte"),
        ("R29", "0x1D", "Y-register High Byte"),
        ("R30", "0x1E", "Z-register Low Byte"),
        ("R31", "0x1F", "Z-register High Byte"),
    ]

    for register, address, description in expected_registers:

        # All three values should be present
        # in the same structured element.
        assert any(
            register in content
            and address in content
            and description in content
            for content in contents
        ), f"Missing register mapping: {register}"
