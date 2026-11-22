"""Tech-Specs Agent — English-language local Streamlit app.

From repository root:
    python -m streamlit run scripts/web_app.py --server.address 127.0.0.1
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tech_specs.custom_docs import import_pdf_bytes, index_custom_vectors, list_custom_documents
from tech_specs.web_runtime import AgentResources, format_report_json, source_evidence, status_details

st.set_page_config(page_title="Tech-Specs Agent", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 1.75rem;}
.app-kicker {letter-spacing: .14em; text-transform: uppercase; color: #64748b; font-size: .75rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_resources(root: str) -> AgentResources:
    return AgentResources(root)


def ollama_reachable() -> bool:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0) as response:
            return response.status == 200
    except (URLError, OSError, TimeoutError):
        return False


def show_sources(state: dict) -> None:
    evidence = source_evidence(state)
    if not evidence:
        st.caption("No retrieved evidence available.")
        return
    cited = [row for row in evidence if row["cited"]]
    if cited:
        st.caption("Cited sources (citation IDs validated, not independent factual proof):")
        for row in cited:
            with st.expander(f"📄 {row['chunk_id']} · PDF page {row['page']}"):
                st.caption(f"Document: {row['document']} · {row['section']}")
                st.text(row["content"])
    others = [row for row in evidence if not row["cited"]]
    if others:
        with st.expander(f"Other retrieved passages ({len(others)})"):
            for row in others:
                st.markdown(f"**{row['chunk_id']}** · PDF page {row['page']}")
                if row["section"]:
                    st.caption(row["section"])
                st.text(row["content"])
                st.divider()


def show_diagnostics(state: dict) -> None:
    original = state.get("original_result") or {}
    working = state.get("working_result") or {}
    plan = original.get("plan") or {}
    st.write(f"**Answer strategy:** {state.get('final_answer_strategy', 'not provided')}")
    st.write(f"**Revision attempts:** {state.get('repairs', 0)}")
    st.write(f"**Document scope:** {plan.get('document') or 'automatic / original corpus'}")
    if plan.get("facets"):
        st.write("**Retrieval queries:**")
        for facet in plan["facets"]:
            st.code(f"{facet.get('id', '')}: {facet.get('query', '')}", language=None)
    source_check = state.get("source_cell_check") or {}
    if source_check.get("applicable"):
        st.write(f"**Checked table cells:** {len(source_check.get('cells') or [])} · "
                 f"accepted={source_check.get('accepted')} · {source_check.get('reason', '')}")
    audits = state.get("audits") or []
    if audits:
        st.write(f"**Claim audit:** {audits[-1].get('audit_status', 'unknown')}")
        st.json(audits[-1].get("counts") or {})
    coverage = state.get("coverage_checks") or []
    if coverage:
        st.write(f"**Coverage check:** {coverage[-1].get('verdict', 'unknown')}")
        missing = coverage[-1].get("missing_aspects") or []
        if missing:
            st.caption("Possible omissions: " + "; ".join(map(str, missing)))
    checks = state.get("table_checks") or []
    if checks:
        issues = checks[-1].get("issues") or []
        st.write(f"**Table conflicts:** {len(issues)}")
        for issue in issues:
            st.warning(issue.get("message", "A table conflict was found"))
    if working.get("status") != "citations_checked" and state.get("status") != "source_table_checked":
        st.caption(f"Generation status: {working.get('status', 'unknown')}")


def show_response(message: dict) -> None:
    state = message.get("result")
    if not state:
        st.error(message.get("error", "An unknown error occurred."))
        return
    label, detail, visible = status_details(state.get("status", ""))
    if state.get("status") in {"source_table_checked", "local_checks_passed"}:
        st.success(label)
    elif state.get("status") == "coverage_review_recommended":
        st.warning(label)
    else:
        st.error(label)
    if visible:
        draft = (state.get("working_result") or {}).get("draft", "")
        if draft:
            st.markdown(draft)
        else:
            st.write(state.get("report", "No answer text available."))
    else:
        st.write(state.get("report", "The generated answer was withheld by the checks."))
    st.caption(detail)
    a, b, c = st.columns(3)
    a.caption("Method: " + message.get("method", "unknown"))
    b.caption("Model: " + message.get("model", "unknown"))
    c.caption("Cited chunks: " + str(sum(row["cited"] for row in source_evidence(state))))
    with st.expander("📚 Sources and PDF pages", expanded=visible):
        show_sources(state)
    with st.expander("🛠️ Agent diagnostics"):
        show_diagnostics(state)
    st.download_button("Download JSON report", format_report_json(state),
                       file_name=f"tech_specs_{message['index']:03d}.json",
                       mime="application/json", key=f"download_{message['index']}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_index" not in st.session_state:
    st.session_state.next_index = 1
if "pending_example" not in st.session_state:
    st.session_state.pending_example = ""

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    with st.expander("➕ Add a PDF datasheet", expanded=False):
        st.caption("Import a text-based PDF into its own local collection. BM25 is available immediately; vector/hybrid search requires a separate index.")
        uploaded = st.file_uploader("PDF datasheet", type=["pdf"], key="pdf_upload")
        display_name = st.text_input("Display name (optional)", key="new_display_name")
        if st.button("Import PDF", disabled=uploaded is None, use_container_width=True):
            try:
                with st.spinner("Extracting document text and indexing BM25 passages..."):
                    info = import_pdf_bytes(ROOT, uploaded.name, uploaded.getvalue(),
                                            display_name=display_name)
                st.session_state["selected_document"] = info["id"]
                st.success(f"Imported: {info['title']} · {info['pages']} pages · {info['chunks']} chunks")
                if info.get("warnings"):
                    for warning in info["warnings"]:
                        st.warning(warning)
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    custom_docs = list_custom_documents(ROOT)
    choices = {"auto": "Auto-detect (original datasheets)",
               "attiny85": "ATtiny25/45/85 — original",
               "stm32l432kc": "STM32L432KC — original"}
    for doc in custom_docs:
        choices[doc["id"]] = f"{doc['title']} — imported ({doc['pages']} pages)"
    if st.session_state.get("selected_document") not in choices:
        st.session_state["selected_document"] = "auto"
    selected = st.selectbox("Datasheet", list(choices),
                            format_func=lambda value: choices[value],
                            key="selected_document")
    document = None if selected == "auto" else selected
    method_label = st.selectbox("Retrieval method", ["BM25", "Hybrid (RRF)", "Vector"], index=0)
    method = {"BM25": "bm25", "Hybrid (RRF)": "hybrid", "Vector": "vector"}[method_label]
    if custom_docs and document and document not in {"attiny85", "stm32l432kc"}:
        info = next(doc for doc in custom_docs if doc["id"] == document)
        with st.expander("📄 Imported document details", expanded=True):
            st.caption(f"PDF pages: {info['pages']} · Text pages: {info['text_pages']} · "
                       f"Chunks: {info['chunks']} · Table chunks: {info['table_chunks']}")
            for warning in info.get("warnings") or []:
                st.warning(warning)
            if not info.get("vector_indexed"):
                st.info("Vector/Hybrid requires an index. BM25 can be used now.")
                if st.button("Build vector index", use_container_width=True):
                    try:
                        with st.spinner("Embedding document passages locally (may be slow on the first run)..."):
                            indexed = index_custom_vectors(ROOT, document)
                        st.success(f"Vector index ready: {indexed['chunks']} chunks")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Vector indexing failed: {exc}. Close other programs accessing this index and try again.")
            else:
                st.caption("Vector index ready")
    model = st.text_input("Ollama model", value="qwen3:4b")
    with st.expander("Advanced settings"):
        top_k = st.slider("Answer context chunks", 3, 12, 6)
        candidates = st.slider("Candidates per retrieval query", 5, 20, 10)
        num_ctx = st.select_slider("Model context window", [2048, 4096, 8192], value=4096)
        num_predict = st.slider("Maximum generation tokens", 400, 2000, 1200, step=200)
        repairs = st.checkbox("Allow one revision attempt", value=True)
        table_first = st.checkbox("Direct table-cell extraction (original docs only)", value=True)
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.next_index = 1
        st.rerun()
    st.caption("🟢 Ollama is reachable" if ollama_reachable()
               else "🟠 Ollama is offline at 127.0.0.1:11434")
    st.caption("Local only · Your PDF files and indexes stay on this computer")

st.markdown('<div class="app-kicker">RAG · EMBEDDED SYSTEMS · LOCAL AI</div>', unsafe_allow_html=True)
st.title("🔬 Tech-Specs Agent")
st.write("Ask grounded questions about the original ATtiny and STM32 datasheets, or **import another PDF datasheet** to test a new device.")
st.caption("A validated citation is not proof that every technical statement is correct. For imported PDFs, tables are extracted on a best-effort basis and the generic single-document planner is used.")

examples = [
    ("TIM2 resolution", "What is TIM2 counter resolution on STM32L432KC?"),
    ("WDTON fuse", "How does programming the WDTON fuse change the watchdog initial state and disable behavior on ATtiny25/45/85?"),
    ("Stop 2 triggers", "Can STM32L432xx timers trigger ADC/DAC conversions in Low-power Run and Low-power Sleep, and is that connection supported in Stop 2?"),
]
if not st.session_state.messages and document in {None, "attiny85", "stm32l432kc"}:
    st.markdown("##### Try an example question")
    for col, (label, query) in zip(st.columns(3), examples):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_example = query

for message in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(message["question"])
    with st.chat_message("assistant"):
        show_response(message)

question = st.chat_input("Ask a question about the selected datasheet...")
if not question and st.session_state.pending_example:
    question = st.session_state.pending_example
    st.session_state.pending_example = ""
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the PDF and checking supporting evidence..."):
                state = get_resources(str(ROOT)).ask(
                    question, method=method, document=document,
                    model=model.strip() or "qwen3:4b", top_k=top_k,
                    candidates_per_facet=candidates, num_ctx=num_ctx,
                    num_predict=num_predict, max_repairs=1 if repairs else 0,
                    table_first=table_first)
            item = {"question": question, "result": state, "method": method,
                    "model": model.strip() or "qwen3:4b", "index": st.session_state.next_index}
            st.session_state.messages.append(item)
            st.session_state.next_index += 1
            show_response(item)
        except Exception as exc:
            msg = str(exc)
            if "11434" in msg or "Connection refused" in msg:
                msg = "Cannot reach local Ollama. Start Ollama and run 'ollama list'. Details: " + msg
            elif "storage folder" in msg or "already accessed" in msg or "already open" in msg:
                msg = "Qdrant Local appears to be open in another process. Close other programs using this index. Details: " + msg
            item = {"question": question, "error": msg, "index": st.session_state.next_index}
            st.session_state.messages.append(item)
            st.session_state.next_index += 1
            st.error(msg)
