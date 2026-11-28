"""Evidence-focused retrieval for newly imported microcontroller datasheets.

No datasheet-specific answer keys, truth labels, LLM calls, or network requests.
Uses the EXISTING retriever (BM25/vector/hybrid) and imported chunks.jsonl.

For two broad, reusable technical question classes, it:
  * issues answer-free query paraphrases;
  * pools results by weighted reciprocal rank;
  * considers a narrow PDF-page neighborhood;
  * promotes passages that state the requested property rather than related
    but different metrics (e.g. nominal ADC resolution vs. measured ENOB).

For other questions, it transparently delegates to the original retriever.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

_CPU = re.compile(r"\b(?:cpu|processor|microprocessor|core|instruction\s+set|isa)\b", re.I)
_ARCH = re.compile(r"\b(?:architecture|instruction\s+set|isa|processor\s+type|cpu\s+type|core\s+type)\b", re.I)
_ADC = re.compile(r"\b(?:adc|analog(?:ue)?[ -]+to[ -]+digital(?:[ -]+converter)?)\b", re.I)
_RESOLUTION = re.compile(r"\b(?:resolution|bit[ -]+depth|bit[ -]+width|how\s+many\s+bits|\d+[ -]?bit)\b", re.I)
_ENOB = re.compile(r"\b(?:enob|effective\s+(?:number\s+of\s+)?bits|effective\s+resolution)\b", re.I)
_CPU_ISA = re.compile(r"\b(?:risc[ -]?v|rv32\w*|armv\w*|cortex[ -]?\w+|xtensa|mips|pic\d+|avr|8051|arc\s+em|tensilica)\b", re.I)
_N_BIT = re.compile(r"\b(?:\d{1,2}[ -]?bit|\d{1,2}\s+bits)\b", re.I)
_ERRATA = re.compile(r"\b(?:errata|workaround|enob|dnl|inl|effective\s+number\s+of\s+bits)\b", re.I)
_STOP = {
    "what", "which", "does", "the", "its", "is", "are", "use", "used", "of", "a", "an",
    "for", "in", "on", "and", "to", "it", "this", "that", "how", "many", "with", "has",
}
_TOKEN = re.compile(r"[\w]+(?:-[\w]+)*", re.UNICODE)


def question_focus(query: str) -> str | None:
    """Detect the type of requested *property*, not its answer."""
    if _ADC.search(query) and _ENOB.search(query):
        return "adc_effective"
    if _ADC.search(query) and _RESOLUTION.search(query):
        return "adc_nominal"
    if _CPU.search(query) and _ARCH.search(query):
        return "cpu_architecture"
    return None


def query_variants(query: str) -> list[str]:
    """General-purpose rephrasings; none embeds a device's expected answer."""
    question = query.strip()
    focus = question_focus(question)
    variants = [question]
    # Preserve part numbers explicitly present in a question (e.g. ESP32-C3,
    # RP2040). This is not an answer hint: it only preserves document context.
    part_numbers = [t for t in _TOKEN.findall(question)
                    if 4 <= len(t) <= 32 and re.search(r"[a-z]", t, re.I)
                    and re.search(r"\d", t)]
    context = " ".join(dict.fromkeys(part_numbers[:2]))
    prefix = (context + " ") if context else ""
    if focus == "cpu_architecture":
        variants.extend([
            prefix + "microprocessor high-performance CPU core instruction set architecture",
            prefix + "CPU processor core architecture instruction set overview",
        ])
    elif focus == "adc_nominal":
        variants.extend([
            prefix + "analog to digital converter ADC nominal resolution bit width",
            prefix + "ADC conversion resolution bits hardware SAR converter",
        ])
    elif focus == "adc_effective":
        variants.extend([
            prefix + "ADC ENOB effective number of bits measured resolution",
            prefix + "ADC effective resolution ENOB characterization measurement",
        ])
    return variants


def _tokens(text: str) -> set[str]:
    return {x.casefold() for x in _TOKEN.findall(text) if len(x) > 2 and x.casefold() not in _STOP}


def _local_window(content: str, patterns: tuple[re.Pattern, ...], radius: int = 230) -> list[str]:
    windows = []
    for pattern in patterns:
        for match in list(pattern.finditer(content))[:8]:
            windows.append(content[max(0, match.start() - radius):match.end() + radius])
    return windows or [content[:450]]


def evidence_quality(query: str, chunk: dict) -> float:
    """A small explainable heuristic over text, not a factual correctness model."""
    focus = question_focus(query)
    if not focus:
        return 0.0
    content = str(chunk.get("content", ""))
    if not content:
        return -3.0
    windows = _local_window(content, (_ADC,) if focus.startswith("adc") else (_CPU,))
    question_terms = _tokens(query)
    best = 0.0
    for window in windows:
        score = 0.0
        lower = window.casefold()
        common = len(question_terms & _tokens(window))
        score += min(common, 4) * 0.23
        if focus == "cpu_architecture":
            if _CPU.search(window):
                score += 1.5
            if _ARCH.search(window):
                score += 1.8
            if _CPU_ISA.search(window) and _CPU.search(window):
                score += 5.0
            if "microprocessor" in lower and "high-performance" in lower:
                score += 1.2
            if ("xts-aes" in lower or "encryption" in lower or "cache" in lower) and not _CPU_ISA.search(window):
                score -= 1.5
        elif focus == "adc_nominal":
            has_adc = _ADC.search(window) is not None
            has_number = _N_BIT.search(window) is not None
            if has_adc:
                score += 1.6
            if _RESOLUTION.search(window):
                score += 2.0
            if has_adc and has_number:
                score += 4.8
            if _ENOB.search(window):
                score -= 5.5
            if _ERRATA.search(window) and not _RESOLUTION.search(window):
                score -= 2.0
        else:  # adc_effective
            if _ADC.search(window):
                score += 1.3
            if _ENOB.search(window):
                score += 5.0
            if _N_BIT.search(window):
                score += 1.3
        best = max(best, score)
    # A table of errata about ENOB is not evidence of the nominal ADC bit depth.
    if focus == "adc_nominal" and _ENOB.search(content) and not (
        re.search(r"\bnominal\b", content, re.I) and _RESOLUTION.search(content)
    ):
        best -= 5.0
    return best


class EvidenceFocusedRetriever:
    """Drop-in .search() wrapper around existing retrievers for imported PDFs."""

    def __init__(self, base: Any, chunks: list[dict], *, pool_k: int = 26,
                 neighbor_radius: int = 2, max_neighbor_candidates: int = 150):
        if pool_k < 2 or neighbor_radius not in (0, 1, 2) or max_neighbor_candidates < 0:
            raise ValueError("Invalid retrieval enhancement settings")
        self.base = base
        self.pool_k = pool_k
        self.neighbor_radius = neighbor_radius
        self.max_neighbor_candidates = max_neighbor_candidates
        self._pages: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for chunk in chunks:
            if isinstance(chunk.get("page"), int):
                self._pages[(str(chunk["document"]), chunk["page"])].append(chunk)
        self.last_diagnostics: dict = {}

    def search(self, query: str, top_k: int = 5, document: str | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("Query must be nonempty and top_k >= 1")
        focus = question_focus(query)
        if focus is None:  # all other technical questions retain the prior behavior
            result = self.base.search(query, top_k=top_k, document=document)
            self.last_diagnostics = {"mode": "original", "queries": [query], "focus": None}
            return result
        queries = query_variants(query)
        pool: dict[str, dict] = {}
        rank_score: dict[str, float] = defaultdict(float)
        primary: list[dict] = []
        for query_index, variant in enumerate(queries):
            hits = self.base.search(variant, top_k=max(top_k, self.pool_k), document=document)
            if query_index == 0:
                primary = hits
            weight = 1.4 if query_index == 0 else 1.0
            for position, hit in enumerate(hits, 1):
                chunk = hit["chunk"]
                cid = str(chunk["chunk_id"])
                if document is not None and chunk.get("document") != document:
                    continue
                pool[cid] = chunk
                rank_score[cid] += weight / (35 + position)
        # Context near original/expanded search results often contains the actual
        # property stated in a preceding section heading or adjacent PDF page.
        neighbor_count = 0
        if self.neighbor_radius:
            seeds = sorted(pool.values(), key=lambda c: -rank_score[c["chunk_id"]])[:12]
            seen_ids = set(pool)
            for seed in seeds:
                if neighbor_count >= self.max_neighbor_candidates:
                    break
                d, page = str(seed.get("document")), seed.get("page")
                if not isinstance(page, int):
                    continue
                for other_page in range(max(1, page - self.neighbor_radius), page + self.neighbor_radius + 1):
                    for chunk in self._pages.get((d, other_page), []):
                        cid = str(chunk["chunk_id"])
                        if cid in seen_ids:
                            continue
                        seen_ids.add(cid)
                        if evidence_quality(query, chunk) > 2.4:
                            pool[cid] = chunk
                            neighbor_count += 1
                            if neighbor_count >= self.max_neighbor_candidates:
                                break
                    if neighbor_count >= self.max_neighbor_candidates:
                        break
        scored = []
        for cid, chunk in pool.items():
            quality = evidence_quality(query, chunk)
            score = rank_score[cid] + quality * 0.018
            scored.append((score, rank_score[cid], cid, chunk))
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
        self.last_diagnostics = {
            "mode": "focused", "focus": focus, "queries": queries,
            "candidate_count": len(pool), "neighbor_candidates": neighbor_count,
            "top_signals": [(row[2], round(evidence_quality(query, row[3]), 3)) for row in scored[:top_k]],
        }
        return [
            {"rank": pos, "score": round(row[0], 7), "chunk": row[3]}
            for pos, row in enumerate(scored[:top_k], 1)
        ]
