"""Offline tests: retrieval precision without model, vector DB or answer keys."""
from tech_specs.evidence_search import (
    EvidenceFocusedRetriever, evidence_quality, question_focus, query_variants,
)


def chunk(cid, page, text, doc="user-fake"):
    return {"chunk_id": cid, "document": doc, "page": page,
            "section": f"PDF page {page}", "content_type": "text", "content": text}


class FakeRetriever:
    """Rank the baseline query poorly, and the semantic paraphrase more usefully."""

    def __init__(self, base, boosted=None):
        self.baseline = base
        self.boosted = boosted or base
        self.calls = []

    def search(self, query, top_k=5, document=None):
        self.calls.append((query, top_k, document))
        hits = self.boosted if "instruction set" in query.lower() or "nominal" in query.lower() else self.baseline
        return [{"rank": i, "chunk": c, "score": 1.0 / i}
                for i, c in enumerate(hits, 1) if document is None or c["document"] == document][:top_k]


def test_detects_intents_but_not_cpu_answer():
    assert question_focus("Which CPU architecture does this device use?") == "cpu_architecture"
    assert question_focus("What is the resolution of its ADC?") == "adc_nominal"
    assert question_focus("What is the effective ADC resolution (ENOB)?") == "adc_effective"
    assert question_focus("How much SRAM is available?") is None
    assert "risc-v" not in " ".join(query_variants("Which CPU architecture does it use?")).lower()
    assert "12-bit" not in " ".join(query_variants("What is the ADC resolution?")).lower()


def test_cpu_neighbor_page_promoted_without_architecture_hardcoding():
    # In real PDFs, the actual CPU subsection often precedes memory sections.
    memory = chunk("memory", 35, "CPU instruction memory space maps to flash via XTS-AES encryption")
    flash = chunk("flash", 34, "CPU access flash memory with a 16 KB cache")
    cpu = chunk("cpu", 33, "The high-performance CPU has a 32-bit ARM Cortex-M33 core with an instruction set.")
    retriever = FakeRetriever([memory, flash])
    focused = EvidenceFocusedRetriever(retriever, [memory, flash, cpu])
    result = focused.search("Which CPU architecture does this microcontroller use?", top_k=3, document="user-fake")
    assert result[0]["chunk"]["chunk_id"] == "cpu"
    assert focused.last_diagnostics["neighbor_candidates"] >= 1
    assert focused.last_diagnostics["focus"] == "cpu_architecture"


def test_cpu_paraphrase_source_even_without_neighbor_expansion():
    wrong = chunk("aes", 45, "The CPU flash bus uses XTS-AES security")
    right = chunk("real_cpu", 8, "Microprocessor: single-core 32-bit RISC-V CPU architecture.")
    retriever = FakeRetriever([wrong], [right, wrong])
    focused = EvidenceFocusedRetriever(retriever, [right, wrong], neighbor_radius=0)
    ids = [item["chunk"]["chunk_id"] for item in focused.search(
        "Which CPU architecture does this chip use?", top_k=2, document="user-fake")]
    assert ids[0] == "real_cpu"
    assert len(retriever.calls) == 3


def test_adc_nominal_bit_depth_preferred_to_enob_errata():
    errata = chunk("errata", 606, "RP2040 ADC errata: ENOB reduced to 8.7 bits. DNL workaround none.")
    context = chunk("adc_context", 536, "SAR ADC block diagram 500 ksps conversion")
    nominal = chunk("adc_nominal", 535, "The analog-to-digital converter ADC has 12-bit resolution.")
    retriever = FakeRetriever([errata, context])
    focused = EvidenceFocusedRetriever(retriever, [errata, context, nominal])
    result = focused.search("What is the resolution of its ADC?", top_k=3, document="user-fake")
    assert result[0]["chunk"]["chunk_id"] == "adc_nominal"
    assert evidence_quality("What is the resolution of its ADC?", nominal) > evidence_quality(
        "What is the resolution of its ADC?", errata)


def test_enob_question_keeps_measured_value():
    effective = chunk("enob", 606, "ADC has measured ENOB of 8.7 bits at 250 ksps")
    nominal = chunk("nominal", 535, "ADC has a nominal resolution of 12-bit")
    retriever = FakeRetriever([nominal, effective])
    focused = EvidenceFocusedRetriever(retriever, [nominal, effective])
    result = focused.search("What is ADC effective resolution ENOB?", top_k=2, document="user-fake")
    assert result[0]["chunk"]["chunk_id"] == "enob"


def test_unrelated_queries_keep_exact_baseline_and_single_call():
    passage = chunk("x", 7, "The MCU has two UART controllers")
    retriever = FakeRetriever([passage])
    focused = EvidenceFocusedRetriever(retriever, [passage])
    result = focused.search("How many UART controllers are available?", top_k=5, document="user-fake")
    assert result[0]["chunk"]["chunk_id"] == "x"
    assert focused.last_diagnostics["mode"] == "original"
    assert len(retriever.calls) == 1


def test_document_scope_and_dedup():
    local = chunk("local", 1, "The CPU has an ARM Cortex-M4 architecture", "doc-a")
    other = chunk("other", 1, "The CPU uses RISC-V instruction set architecture", "doc-b")
    retriever = FakeRetriever([other, local])
    focused = EvidenceFocusedRetriever(retriever, [local, other])
    results = focused.search("What CPU architecture is used?", top_k=5, document="doc-a")
    assert [hit["chunk"]["chunk_id"] for hit in results] == ["local"]
    assert all(hit["chunk"]["document"] == "doc-a" for hit in results)


def test_empty_query_and_invalid_config_rejected():
    fake = FakeRetriever([])
    try:
        EvidenceFocusedRetriever(fake, [], neighbor_radius=3)
        assert False
    except ValueError:
        pass
    try:
        EvidenceFocusedRetriever(fake, []).search("  ")
        assert False
    except ValueError:
        pass


def test_technical_prompt_distinctions():
    from tech_specs.local_answer import SYSTEM_PROMPT
    assert "nominal hardware resolution" in SYSTEM_PROMPT
    assert "Memory mapping" in SYSTEM_PROMPT
