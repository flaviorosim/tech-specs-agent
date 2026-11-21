"""Regression tests for reproducible sampling of benchmark source chunks."""

import csv

import pytest

from scripts.sample_benchmark_sources import (
    load_previously_judged,
    select_sources,
    suitable,
)


def fake_chunk(document, page, kind="text", section="4.2 Timers"):
    return {
        "chunk_id": f"{document}_p{page:03d}_{kind}",
        "document": document,
        "page": page,
        "section": section,
        "content_type": kind,
        "content": (
            "|Register|Address|Function|\n|---|---|---|\n"
            "|TIM2|0x42|Timer counter resolution: 32-bit, up/down count, input capture, output compare and PWM|"
            if kind == "table" else
            "The timer is capable of measuring the time between external "
            "events and generating output compare signals. " * 3
        ),
        "metadata": {"review_required": False},
    }


def test_skips_old_judgments_and_noise():
    old = fake_chunk("stm32l432kc", 18)
    assert not suitable(old, {old["chunk_id"]}, 8, 2500)
    toc = {**old, "section": "4.2 Timers . . . . . . . 52"}
    assert not suitable(toc, set(), 8, 2500)
    footer = {**old, "content": "148"}
    assert not suitable(footer, set(), 8, 2500)
    assert suitable(old, set(), 8, 2500)


def test_deterministic_and_balanced_selection():
    chunks = []
    for doc in ("attiny85", "stm32l432kc"):
        for page in range(10, 70):
            kind = "table" if page % 4 == 0 else "text"
            chunks.append(fake_chunk(doc, page, kind, f"{page // 6}.1 Register section"))
    first = select_sources(chunks, set(), per_document=12, tables_per_document=4)
    second = select_sources(list(reversed(chunks)), set(), per_document=12,
                            tables_per_document=4)
    assert {c["chunk_id"] for c in first} == {c["chunk_id"] for c in second}
    assert len(first) == 24
    for doc in ("attiny85", "stm32l432kc"):
        chosen = [c for c in first if c["document"] == doc]
        assert len(chosen) == 12
        assert sum(c["content_type"] == "table" for c in chosen) == 4
        assert len({c["page"] // 10 for c in chosen}) >= 3


def test_preserves_full_content_and_excludes_judged_chunk():
    chunks = [fake_chunk("attiny85", p) for p in range(10, 30)]
    exclude = {chunks[0]["chunk_id"]}
    sample = select_sources(chunks, exclude, per_document=3, tables_per_document=0)
    assert len(sample) == 3
    assert all(c["chunk_id"] not in exclude for c in sample)
    assert all(c["content"].endswith("signals. ") for c in sample)


def test_reads_utf8_bom_judgments(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_id", "relevance"])
        writer.writeheader()
        writer.writerow({"chunk_id": "old_1", "relevance": "2"})
    assert load_previously_judged(path) == {"old_1"}


def test_invalid_quotas_fail():
    with pytest.raises(ValueError):
        select_sources([], set(), per_document=4, tables_per_document=5)
