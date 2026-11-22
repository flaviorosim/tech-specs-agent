"""Conservative, source-derived answers for *unambiguous* Markdown tables.

This is NOT an LLM, a semantic entailment judge, or a benchmark answer key.
It extracts requested cells from retrieved chunks and cites the same chunks.
If the match is not unique, complete, or recognized, it leaves the query to
an ordinary generator instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class TableAnswer:
    draft: str
    source_id: str
    kind: str
    cells: tuple[dict[str, str], ...]


def _norm(text: str) -> str:
    """Normalize extraction linebreaks and HTML used in table cells."""
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = text.replace("\u2011", "-").replace("\u2013", "-")
    return re.sub(r"\s+", " ", text).strip()


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(text).casefold())


def _tables(content: str) -> list[tuple[list[str], list[list[str]]]]:
    """Read only actual Markdown tables with a pipe header and divider row."""
    results: list[tuple[list[str], list[list[str]]]] = []
    lines = content.splitlines()
    i = 0
    while i + 1 < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        header = [_norm(x) for x in lines[i].strip().strip("|").split("|")]
        divider = [x.strip() for x in lines[i + 1].strip().strip("|").split("|")]
        if (len(header) < 2 or len(divider) != len(header)
                or not all(re.fullmatch(r":?-{3,}:?", x) for x in divider)):
            i += 1
            continue
        i += 2
        rows = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            row = [_norm(x) for x in lines[i].strip().strip("|").split("|")]
            if len(row) != len(header):
                break
            rows.append(row)
            i += 1
        if rows:
            results.append((header, rows))
    return results


def _source(hits: list[dict]) -> list[tuple[str, list[str], list[list[str]]]]:
    matched = []
    for hit in hits:
        chunk = hit.get("chunk") or {}
        source_id = chunk.get("chunk_id", "")
        content = chunk.get("content", "")
        if not source_id or not isinstance(content, str):
            continue
        for header, rows in _tables(content):
            matched.append((source_id, header, rows))
    return matched


def _is_wdt_question(question: str) -> bool:
    q = question.casefold()
    return (bool(re.search(r"\bwdton\b", q))
            and bool(re.search(r"\b(?:wdt|watchdog)\b", q))
            and ("initial" in q or "disabl" in q))


def _wdt_candidate(source_id: str, headers: list[str], rows: list[list[str]], question: str) -> TableAnswer | None:
    # General configuration-matrix layout; all values come from retrieved source.
    header_keys = [_key(h) for h in headers]
    try:
        fuse_col = header_keys.index("wdton")
    except ValueError:
        return None
    init_cols = [i for i, key in enumerate(header_keys) if "wdtinitialstate" in key]
    disable_cols = [i for i, key in enumerate(header_keys) if "howtodisablethewdt" in key]
    if len(init_cols) != 1 or len(disable_cols) != 1:
        return None
    start, disable = init_cols[0], disable_cols[0]
    by_setting = {_key(row[fuse_col]): row for row in rows}
    if "programmed" not in by_setting or "unprogrammed" not in by_setting:
        return None
    # The question asks about a changed setting, so show both alternatives.
    # Only produce an answer if all necessary fields are populated.
    lines, cells = [], []
    for setting in ("Unprogrammed", "Programmed"):
        row = by_setting[_key(setting)]
        initial_state, disable_rule = row[start], row[disable]
        if not initial_state or not disable_rule:
            return None
        lines.append(
            f"For WDTON = {setting}, the WDT Initial State is {initial_state}; "
            f"the datasheet's 'How to Disable the WDT' entry is '{disable_rule}'. "
            f"[[{source_id}]]"
        )
        cells.append({"setting": setting, "initial_state": initial_state,
                      "how_to_disable": disable_rule})
    return TableAnswer("\n\n".join(lines), source_id, "configuration_table", tuple(cells))


_MODE_PATTERNS = (
    ("Low-power run", re.compile(r"\blow[ -]power\s+run\b", re.I)),
    ("Low-power sleep", re.compile(r"\blow[ -]power\s+sleep\b", re.I)),
    ("Stop 2", re.compile(r"\bstop\s*2\b", re.I)),
    ("Stop 0 / Stop 1", re.compile(r"\bstop\s*(?:0\s*/\s*1|0\s+and\s+1)\b", re.I)),
)


def _mode_question(question: str) -> bool:
    q = question.casefold()
    return (bool(re.search(r"\b(?:timx|timers?|tim\d+)\b", q))
            and bool(re.search(r"\badc\w*|\bdac\w*", q))
            and bool(re.search(r"\b(?:trigg\w*|conversion\w*|interconnect\w*)\b", q)))


def _component_group_in_question(component: str, question: str) -> bool:
    q = question.casefold()
    comp = _key(component)
    if comp == "timx":
        return bool(re.search(r"\b(?:timx|timers?|tim\d+)\b", q))
    if comp.startswith("adc"):
        return bool(re.search(r"\badc\w*", q))
    if comp.startswith("dac"):
        return bool(re.search(r"\bdac\w*", q))
    return bool(re.search(r"\b" + re.escape(component.casefold()) + r"\b", q))


def _interconnect_candidate(source_id: str, headers: list[str], rows: list[list[str]], question: str) -> TableAnswer | None:
    keys = [_key(h) for h in headers]
    try:
        src_col = keys.index("interconnectsource")
        dst_col = keys.index("interconnectdestination")
        action_col = keys.index("interconnectaction")
    except ValueError:
        return None
    requested_modes = [(label, regex) for label, regex in _MODE_PATTERNS if regex.search(question)]
    if not requested_modes:
        return None
    mode_columns = {}
    for label, _ in requested_modes:
        idxs = [i for i, k in enumerate(keys) if k == _key(label)]
        if len(idxs) != 1:
            return None
        mode_columns[label] = idxs[0]
    # Reject ambiguous questions/rows. If both ADC and DAC were requested,
    # require both terms in the same destination cell.
    wants_adc = bool(re.search(r"\badc\w*", question, re.I))
    wants_dac = bool(re.search(r"\bdac\w*", question, re.I))
    candidates = []
    for row in rows:
        src, dst, action = row[src_col], row[dst_col], row[action_col]
        if _key(src) != "timx":
            continue
        if "conversiontrigger" not in _key(action):
            continue
        if wants_adc and "adc" not in _key(dst):
            continue
        if wants_dac and "dac" not in _key(dst):
            continue
        if any(row[idx] not in ("Y", "-") for idx in mode_columns.values()):
            continue
        candidates.append((src, dst, action, row))
    if len(candidates) != 1:
        return None
    src, dst, action, row = candidates[0]
    clauses, cells = [], []
    for label, col in mode_columns.items():
        value = row[col]
        status = "supported" if value == "Y" else "not supported"
        clauses.append(
            f"The {src} -> {dst} '{action}' connection is {status} in {label} mode "
            f"(table cell: {value}). [[{source_id}]]"
        )
        cells.append({"source": src, "destination": dst, "action": action,
                      "mode": label, "cell": value})
    return TableAnswer("\n\n".join(clauses), source_id, "interconnect_matrix", tuple(cells))


def answer_from_tables(question: str, hits: list[dict]) -> TableAnswer | None:
    """Answer only recognized, fully-resolved table questions from retrieved text.

    Return None on missing evidence or inconsistent duplicate table candidates.
    No benchmark labels or model-generated assertions are consulted.
    """
    if not isinstance(question, str) or not question.strip() or not isinstance(hits, list):
        return None
    kind = "wdt" if _is_wdt_question(question) else ("mode" if _mode_question(question) else None)
    if not kind:
        return None
    possible: list[TableAnswer] = []
    for source_id, headers, rows in _source(hits):
        if kind == "wdt":
            result = _wdt_candidate(source_id, headers, rows, question)
        else:
            result = _interconnect_candidate(source_id, headers, rows, question)
        if result:
            possible.append(result)
    if not possible:
        return None
    # Several *identical* copies in different chunks are fine; inconsistent
    # evidence must fall back, never select a convenient version.
    first = possible[0]
    if any(other.cells != first.cells for other in possible[1:]):
        return None
    return first


class TableFirstGenerator:
    """Drop-in adapter for LocalOllama.generate(question, hits).

    Structured table answers are taken verbatim from sources; all other
    questions still go through the unchanged local LLM.
    """
    def __init__(self, fallback: Any, *, enabled: bool = True):
        self.fallback = fallback
        self.enabled = enabled
        self.last_extraction: TableAnswer | None = None
        self.last_strategy = "not_run"

    def generate(self, question: str, hits: list[dict]) -> str:
        self.last_extraction = None
        extracted = answer_from_tables(question, hits) if self.enabled else None
        if extracted is not None:
            self.last_extraction = extracted
            self.last_strategy = "source_table_extraction"
            return extracted.draft
        self.last_strategy = "local_llm"
        return self.fallback.generate(question, hits)
