"""Functional PDF-import tests using tiny generated PDFs; no Ollama or Qdrant."""
import json
from pathlib import Path

import pymupdf
import pytest

from tech_specs.custom_docs import (
    _split_text, get_custom_document, import_pdf_bytes,
    list_custom_documents, library_dir,
)
from tech_specs.generic_answer_graph import make_generic_plan


def sample_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((40, 60), "ACME123 microcontroller features a 12-bit ADC and UART.")
    page.insert_text((40, 80), "The ADC can be configured using the ADC_CTRL register.")
    page = pdf.new_page()
    page.insert_text((40, 60), "Maximum operating voltage is 3.3 V. Pin P1 supports UART TX.")
    data = pdf.tobytes()
    pdf.close()
    return data


def test_import_and_catalog_and_page_numbers(tmp_path):
    data = sample_pdf()
    doc = import_pdf_bytes(tmp_path, "ACME123.pdf", data, display_name="ACME 123 Reference")
    assert doc["id"].startswith("user-acme123-")
    assert doc["title"] == "ACME 123 Reference"
    assert doc["pages"] == doc["text_pages"] == 2
    assert doc["chunks"] >= 2
    assert doc["vector_indexed"] is False
    base = library_dir(tmp_path) / doc["id"]
    assert (base / "source.pdf").read_bytes() == data
    chunks = [json.loads(line) for line in (base / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(chunks) == doc["chunks"]
    assert {chunk["page"] for chunk in chunks} == {1, 2}
    assert all(chunk["document"] == doc["id"] for chunk in chunks)
    assert all(chunk["chunk_id"].startswith(doc["id"] + "_") for chunk in chunks)
    assert all(chunk["section"].startswith("PDF page") for chunk in chunks)
    assert get_custom_document(tmp_path, doc["id"])["sha256"] == doc["sha256"]
    assert len(list_custom_documents(tmp_path)) == 1
    # Importing the same bytes twice is idempotent and does not overwrite.
    doc_again = import_pdf_bytes(tmp_path, "ACME123.pdf", data, display_name="Different name")
    assert doc_again["id"] == doc["id"]
    assert len(list_custom_documents(tmp_path)) == 1


def test_original_corpus_untouched(tmp_path):
    baseline = tmp_path / "data" / "processed" / "chunks.jsonl"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("original corpus sentinel", encoding="utf-8")
    import_pdf_bytes(tmp_path, "new_device.pdf", sample_pdf())
    assert baseline.read_text(encoding="utf-8") == "original corpus sentinel"
    assert not (tmp_path / "data" / "processed" / "vector_index_manifest.json").exists()


def test_reject_invalid_pdf_empty_scan_and_large_file(tmp_path):
    with pytest.raises(ValueError, match="PDF header"):
        import_pdf_bytes(tmp_path, "bad.pdf", b"this is a fake PDF" * 2)
    with pytest.raises(ValueError, match="Only .pdf"):
        import_pdf_bytes(tmp_path, "bad.docx", sample_pdf())
    with pytest.raises(ValueError, match="30 MB"):
        import_pdf_bytes(tmp_path, "huge.pdf", b"%PDF-" + b"0" * (30 * 1024 * 1024))
    pdf = pymupdf.open()
    pdf.new_page()
    blank = pdf.tobytes()
    pdf.close()
    with pytest.raises(ValueError, match="No searchable text"):
        import_pdf_bytes(tmp_path, "scan.pdf", blank)
    assert list_custom_documents(tmp_path) == []


def test_slug_sanitization_and_manifest_scan(tmp_path):
    doc = import_pdf_bytes(tmp_path, "../../Super Device (v3).pdf", sample_pdf())
    assert ".." not in doc["id"] and "/" not in doc["id"]
    assert doc["id"].startswith("user-super-device-v3")
    folder = library_dir(tmp_path) / "arbitrary_unregistered"
    folder.mkdir()
    (folder / "manifest.json").write_text('{"id": "fake", "schema_version": 1}')
    assert len(list_custom_documents(tmp_path)) == 1


def test_generic_plan_uses_explicit_document_and_query():
    plan = make_generic_plan("How many PWM channels are provided?", "user-new123")
    assert plan["document"] == "user-new123"
    assert plan["facets"] == [{"id": "general", "query": "How many PWM channels are provided?"}]
    # New documents are not forced into the two-device glossary.
    assert make_generic_plan("Does the new STM32 model support SPI?", "user-other")["document"] == "user-other"


def test_split_text_length_boundaries():
    text = "This MCU has a GPIO interrupt. " * 120
    spans = _split_text(text, max_chars=300, overlap=45)
    assert len(spans) > 2
    assert all(0 < len(span) <= 300 for span in spans)
    assert any("GPIO" in span for span in spans)


def test_best_effort_grid_table_extraction(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page(width=500, height=500)
    for x in [40, 190, 360]:
        page.draw_line((x, 100), (x, 180))
    for y in [100, 140, 180]:
        page.draw_line((40, y), (360, y))
    for x, y, value in [(55, 125, "Mode"), (210, 125, "Value"),
                         (55, 165, "STOP2"), (210, 165, "No")]:
        page.insert_text((x, y), value)
    data = pdf.tobytes()
    pdf.close()
    doc = import_pdf_bytes(tmp_path, "Grid.pdf", data)
    chunks = [json.loads(line) for line in (library_dir(tmp_path) / doc["id"] / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(c["content_type"] == "table" and "STOP2" in c["content"] for c in chunks)
    assert any(c["content_type"] == "text" for c in chunks)
    assert doc["table_chunks"] >= 1
