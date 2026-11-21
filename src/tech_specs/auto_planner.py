"""Deterministic first-pass planner for technical datasheet retrieval.

No LLM is involved. The planner recognizes a small, explicit glossary and
creates one targeted query per recognized topic. Unsupported questions fall
back to a single search of the original user question.

This module uses no retrieval results or gold chunk IDs to make its plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


DOCUMENT_NAMES = {
    "attiny85": "ATtiny25/45/85",
    "stm32l432kc": "STM32L432xx",
}

_DEVICE_PATTERNS = {
    "attiny85": re.compile(r"\b(?:attiny(?:25|45|85)?(?:/45/85)?|avr)\b", re.I),
    "stm32l432kc": re.compile(r"\b(?:stm32(?:l432(?:kc|kb|k[bx]|xx)?)?)\b", re.I),
}


@dataclass(frozen=True)
class Topic:
    id: str
    pattern: re.Pattern[str]
    search_terms: str


# These are deliberately narrow. The graph must not invent capabilities,
# specific bit values, or electrical limits when rewriting the query.
_TOPICS = (
    Topic("adc", re.compile(r"\bADC\d*\b|analog[ -]to[ -]digital(?: converter)?", re.I), "ADC Analog to Digital Converter"),
    Topic("bod", re.compile(r"\bBOD\b|brown[ -]out(?: detector| detection| reset)?", re.I), "Brown-out Detector BOD"),
    Topic("watchdog", re.compile(r"\bWDT\b|watchdog", re.I), "Watchdog Timer WDT"),
    Topic("eeprom", re.compile(r"\bEEPROM\b", re.I), "EEPROM"),
    Topic("gpio", re.compile(r"\bGPIO\w*\b|\bI/O pins?\b|\bport pins?\b", re.I), "GPIO I/O port pins"),
    Topic("timer", re.compile(r"\bTIM\d+\b|\bTimer\b|\bCounter\b", re.I), "timer counter"),
    Topic("usi", re.compile(r"\bUSI\b|\bthree[ -]wire\b|\btwo[ -]wire\b", re.I), "USI serial interface"),
    Topic("dma", re.compile(r"\bDMA\b", re.I), "DMA"),
    Topic("flash", re.compile(r"\bFlash\b", re.I), "Flash memory"),
    Topic("rng", re.compile(r"\bRNG\b|random number generator", re.I), "RNG random number generator"),
)

# Terms describing the *shared task*, used only when splitting 2+ topics.
# Keep this simple, deterministic, and auditable.
_CONTEXT_PATTERN = re.compile(
    r"\b(?:sleep(?:-mode)?|power(?:-down)?|low[ -]power|consumption|"
    r"current|disable[ds]?|enable[ds]?|reset|interrupt|"
    r"clock|frequency|prescaler|calibration|voltage|"
    r"output|input|reference|programming|resolution|"
    r"conversion|wake(?:up)?|PWM|accuracy|mode|modes)\b",
    re.I,
)


def detect_document(question: str, document_hint: str | None = None) -> str | None:
    """Choose a known datasheet only when clearly specified.

    Comparison questions mentioning both families need separate explicit handling;
    v1 refuses to silently choose one device.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty")
    if document_hint is not None and document_hint not in DOCUMENT_NAMES:
        raise ValueError(f"Unknown document: {document_hint}")

    mentioned = {
        doc for doc, pattern in _DEVICE_PATTERNS.items()
        if pattern.search(question)
    }
    if len(mentioned) > 1:
        raise ValueError(
            "Question references both device families; cross-device "
            "comparisons are not supported by the automatic planner yet"
        )
    if document_hint and mentioned and document_hint not in mentioned:
        raise ValueError("Document override conflicts with device mentioned in question")
    return document_hint or (next(iter(mentioned)) if mentioned else None)


def make_plan(question: str, document_hint: str | None = None, max_facets: int = 3) -> dict:
    """Create an automatic facet plan without using retrieved evidence.

    Multi-topic queries are split into concise, topic-centered queries.
    Single/unknown-topic queries retain the original wording.
    """
    if max_facets < 1:
        raise ValueError("max_facets must be >= 1")
    document = detect_document(question, document_hint)
    question = question.strip()
    matched = [t for t in _TOPICS if t.pattern.search(question)]

    if len(matched) > max_facets:
        raise ValueError(
            f"Planner detected {len(matched)} topics, exceeding max_facets={max_facets}; "
            "refusing to silently discard requested topics"
        )

    if len(matched) < 2:
        return {
            "question": question,
            "document": document,
            "facets": [{"id": matched[0].id if matched else "general", "query": question}],
        }

    # Only match shared context markers that already occur in the user's question.
    # Preserve order; remove duplicates. Never insert fabricated facts.
    context = []
    for m in _CONTEXT_PATTERN.finditer(question):
        word = m.group(0).lower()
        if word not in context:
            context.append(word)
    shared_context = " ".join(context[:10])
    prefix = DOCUMENT_NAMES[document] if document else ""

    facets = []
    for topic in matched:
        query = " ".join(part for part in (prefix, topic.search_terms, shared_context) if part)
        facets.append({"id": topic.id, "query": query})

    return {"question": question, "document": document, "facets": facets}
