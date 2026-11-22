"""Conservative, source-only checks for mode-availability Markdown tables.

This detects a subset of contradictory claims using a cited table row. It
never invents facts or assumes that every dash in every table means 'no'.
If the structural match is ambiguous, it abstains instead of deciding.
"""
from __future__ import annotations

import re

from tech_specs.faithfulness import get_claims

_MODE_LABELS = {
    "low-power run": ("low-power run", "low power run"),
    "low-power sleep": ("low-power sleep", "low power sleep"),
    "stop 2": ("stop 2", "stop2"),
    "stop 0 / stop 1": ("stop 0 / stop 1", "stop 0/1"),
}


def _cell(value: str) -> str:
    value = re.sub(r"(?i)<br\s*/?>", " ", value)
    return " ".join(value.split()).strip()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:\d+|x)?", _cell(value).lower()))


def _component_mentioned(value: str, claim: str) -> bool:
    """Require table source/destination component IDs in the *claim*.

    Only normalize the common x and numeric suffix variants. Do not infer
    semantically equivalent peripherals (e.g., RTC vs TIMx).
    """
    c = _tokens(claim)
    field = [t for t in _tokens(value) if t not in {"and", "or"}]
    if not field:
        return False
    for token in field:
        stem = re.sub(r"(?:\d+|x)$", "", token)
        if not any(x == token or (len(stem) >= 3 and (x == stem or re.sub(r"(?:\d+|x)$", "", x) == stem))
                   for x in c):
            return False
    return True


def _parse_mode_rows(content: str) -> list[tuple[dict, dict]]:
    """Return rows only from interconnect availability matrices with Y/- cells."""
    lines = content.splitlines()
    results = []
    header = None
    for line in lines:
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cols = [_cell(x) for x in line.strip().strip("|").split("|")]
        if len(cols) < 6:
            continue
        if any("interconnect source" in x.lower() for x in cols) and any("stop 2" in x.lower() for x in cols):
            header = cols
            continue
        if not header or len(cols) != len(header):
            continue
        if all(re.fullmatch(r":?-{3,}:?", x or "") for x in cols):
            continue
        cells = dict(zip([x.lower() for x in header], cols))
        src = next((v for k, v in cells.items() if "interconnect source" in k), "")
        dst = next((v for k, v in cells.items() if "interconnect destination" in k), "")
        if not src or not dst:
            continue
        modes = {}
        for mode in _MODE_LABELS:
            value = cells.get(mode, "")
            if value in {"Y", "-"}:
                modes[mode] = value
        if modes:
            action = next((v for k, v in cells.items() if "interconnect action" in k), "")
            results.append(({"source": src, "destination": dst,
                             "action": action, "modes": modes}, cells))
    return results


def _claim_polarity(claim: str, mode: str) -> bool | None:
    """True=asserts available; False=asserts unavailable; None=ambiguous."""
    text = claim.lower().replace("‑", "-")
    variants = _MODE_LABELS[mode]
    positions = [text.find(x) for x in variants if x in text]
    if not positions:
        return None
    index = min(positions)
    # Mixed affirmative and negative sentences require a more sophisticated
    # parser; skip rather than accept or reject on ambiguous evidence.
    prior = text[max(0, index - 130):index]
    # Contrast markers separate different mode claims within one sentence.
    prior = re.split(r"\b(?:but|however|whereas)\b|[.;]", prior)[-1]
    negative = bool(re.search(
        r"\b(?:not|not supported|isn't supported|not available|unavailable|unsupported|"
        r"cannot|can't|does not|is not|are not|not possible|no connection)\b", prior
    ))
    positive = bool(re.search(
        r"\b(?:can|supported|available|works|possible|enabled|triggers?|allow(?:s|ed)?)\b", prior
    ))
    if negative:
        return False
    if positive:
        return True
    return None


def inspect_table_claims(report: dict) -> dict:
    """Check cited claims against matching interconnect matrix cells, if present.

    No relevant/matchable table -> checked=0, not an approval.
    Cited chunks are prioritized, then other retrieved chunks from that device.
    """
    hits = report.get("retrieval", {}).get("hits", [])
    sources = {h["chunk"]["chunk_id"]: h["chunk"] for h in hits}
    issues, checked = [], 0
    for claim in get_claims(report):
        cited_docs = {sources[cid].get("document") for cid in claim.source_ids}
        # A wrong citation can point to a nearby table row while the correct
        # conflicting row is another retrieved candidate. Check same-document
        # evidence too, but never the benchmark answer key or external facts.
        candidate_ids = list(dict.fromkeys([
            *claim.source_ids,
            *[cid for cid, chunk in sources.items()
              if chunk.get("document") in cited_docs],
        ]))
        for cid in candidate_ids:
            for row, _ in _parse_mode_rows(sources[cid]["content"]):
                if not (_component_mentioned(row["source"], claim.text)
                        and _component_mentioned(row["destination"], claim.text)):
                    continue
                # A repeated identifier in a source->same-destination row is
                # not sufficient: a claim mentioning TIMx once may instead
                # concern TIMx -> ADCx. Require an action-specific clue.
                if row["source"].lower() == row["destination"].lower():
                    if not any(t in claim.text.lower() for t in _tokens(row["action"]) if len(t) > 5):
                        continue
                if "conversion" in claim.text.lower() and "conversion" not in row["action"].lower():
                    continue
                for mode, value in row["modes"].items():
                    polarity = _claim_polarity(claim.text, mode)
                    if polarity is None:
                        continue
                    checked += 1
                    expected = (value == "Y")
                    if polarity != expected:
                        issues.append({
                            "source_id": cid, "claim": claim.text, "mode": mode,
                            "table_value": value,
                            "source_component": row["source"],
                            "destination_component": row["destination"],
                            "message": (
                                f"Cited interconnect table: {row['source']} -> {row['destination']}, "
                                f"{mode}={value}; claim asserts "
                                f"{'available' if polarity else 'unavailable'}."
                            ),
                        })
    return {"checked_cells": checked, "issues": issues, "passed": not issues,
            "scope": "Only matched retrieved interconnect cells; not general claim verification"}
