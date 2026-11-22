"""Local, read-only Streamlit UI for the existing Tech-Specs Agent v5.

Launch from the repository root:
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

from tech_specs.web_runtime import AgentResources, format_report_json, source_evidence, status_details

st.set_page_config(
    page_title="Tech-Specs Agent", page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1080px; padding-top: 2rem;}
.app-kicker {letter-spacing: .17em; text-transform: uppercase; color: #64748b; font-size: .72rem; font-weight: 700;}
.soft-note {color: #64748b; font-size: .88rem;}
.small-tag {font-size: .78rem; color: #64748b; margin-bottom: .8rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_resources(root: str) -> AgentResources:
    # One shared Qdrant Local client and embedding model per Streamlit process.
    # AgentResources serializes queries under a lock.
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
        st.caption("Nenhum trecho recuperado disponível.")
        return
    cited = [row for row in evidence if row["cited"]]
    if cited:
        st.caption("Fontes citadas pela resposta (IDs confirmados, não garantia de suporte factual):")
        for row in cited:
            subtitle = f"{row['document']} · página {row['page']}"
            with st.expander(f"📄 {row['chunk_id']} — {subtitle}"):
                if row["section"]:
                    st.caption(row["section"])
                st.text(row["content"])
    with st.expander(f"Outros trechos recuperados ({len(evidence) - len(cited)})"):
        for row in evidence:
            if row["cited"]:
                continue
            st.markdown(f"**{row['chunk_id']}** · {row['document']} · p. {row['page']}")
            if row["section"]:
                st.caption(row["section"])
            st.text(row["content"])
            st.divider()


def show_diagnostics(state: dict) -> None:
    origin = state.get("original_result") or {}
    work = state.get("working_result") or {}
    plan = origin.get("plan") or {}
    st.write(f"**Estratégia:** {state.get('final_answer_strategy', 'não informada')}")
    st.write(f"**Revisões:** {state.get('repairs', 0)}")
    if plan.get("facets"):
        st.write("**Subconsultas:**")
        for facet in plan["facets"]:
            st.code(f"{facet.get('id', '')}: {facet.get('query', '')}", language=None)
    cell_check = state.get("source_cell_check") or {}
    if cell_check.get("applicable"):
        st.write(
            f"**Células verificadas:** {len(cell_check.get('cells') or [])} · "
            f"aceita={cell_check.get('accepted')} · {cell_check.get('reason', '')}"
        )
    audits = state.get("audits") or []
    if audits:
        st.write(f"**Auditoria de afirmações:** {audits[-1].get('audit_status', '—')}")
        st.json(audits[-1].get("counts") or {})
    coverage = state.get("coverage_checks") or []
    if coverage:
        st.write(f"**Completude (juiz local):** {coverage[-1].get('verdict', '—')}")
        missing = coverage[-1].get("missing_aspects") or []
        if missing:
            st.caption("Possíveis lacunas: " + "; ".join(map(str, missing)))
    table_checks = state.get("table_checks") or []
    if table_checks:
        issues = table_checks[-1].get("issues") or []
        st.write(f"**Conflitos com tabelas:** {len(issues)}")
        for issue in issues:
            st.warning(issue.get("message", "Conflito detectado"))
    if work.get("status") != "citations_checked" and state.get("status") != "source_table_checked":
        st.caption(f"Estado da geração: {work.get('status', '—')}")


def show_response(message: dict) -> None:
    state = message.get("result")
    if not state:
        st.error(message.get("error", "Falha desconhecida"))
        return
    label, detail, visible = status_details(state.get("status", ""))
    if state.get("status") == "source_table_checked":
        st.success(label)
    elif state.get("status") == "local_checks_passed":
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
            st.write(state.get("report", "Nenhum texto disponível"))
    else:
        # Never reveal a withheld answer as an accepted answer.
        st.write(state.get("report", "A resposta foi retida pelas verificações."))
    st.caption(detail)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Método: " + message.get("method", "—"))
    with col2:
        st.caption("Modelo: " + message.get("model", "—"))
    with col3:
        st.caption("Fontes: " + str(sum(row["cited"] for row in source_evidence(state))))
    with st.expander("📚 Fontes e páginas do datasheet", expanded=visible):
        show_sources(state)
    with st.expander("🛠️ Diagnóstico do agente"):
        show_diagnostics(state)
    st.download_button(
        "Baixar relatório JSON", format_report_json(state),
        file_name=f"tech_specs_{message['index']:03d}.json",
        mime="application/json", key=f"download_{message['index']}",
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_index" not in st.session_state:
    st.session_state.next_index = 1
if "pending_example" not in st.session_state:
    st.session_state.pending_example = ""

with st.sidebar:
    st.markdown("### ⚙️ Configuração")
    method_label = st.selectbox("Recuperação", ["BM25", "Híbrido (RRF)", "Vetorial"], index=0)
    method = {"BM25": "bm25", "Híbrido (RRF)": "hybrid", "Vetorial": "vector"}[method_label]
    document_label = st.selectbox("Datasheet", ["Detecção automática", "ATtiny25/45/85", "STM32L432KC"])
    document = {
        "Detecção automática": None,
        "ATtiny25/45/85": "attiny85",
        "STM32L432KC": "stm32l432kc",
    }[document_label]
    model = st.text_input("Modelo Ollama", value="qwen3:4b")
    with st.expander("Parâmetros avançados"):
        top_k = st.slider("Trechos na resposta", 3, 12, 6)
        candidates = st.slider("Candidatos por subconsulta", 5, 20, 10)
        num_ctx = st.select_slider("Contexto do modelo", options=[2048, 4096, 8192], value=4096)
        num_predict = st.slider("Limite de tokens da geração", 400, 2000, 1200, step=200)
        repairs = st.checkbox("Permitir uma tentativa de correção", value=True)
        table_first = st.checkbox("Extração direta de tabelas", value=True)
    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.next_index = 1
        st.rerun()
    status_online = ollama_reachable()
    st.caption("🟢 Ollama local acessível" if status_online else "🟠 Ollama não responde em 127.0.0.1:11434")
    st.caption("Execução local · PDFs e índices permanecem no seu computador")

st.markdown('<div class="app-kicker">RAG • EMBEDDED SYSTEMS • LOCAL AI</div>', unsafe_allow_html=True)
st.title("🔬 Tech-Specs Agent")
st.write("Pergunte sobre os datasheets de **ATtiny25/45/85** e **STM32L432KC**, com fontes rastreáveis e verificações explícitas.")
st.caption("As verificações são locais e falíveis. 'Fonte citada' não é sinônimo de 'afirmação verdadeira'.")

examples = [
    ("TIM2 — resolução", "What is TIM2 counter resolution on STM32L432KC?"),
    ("Watchdog WDTON", "How does programming the WDTON fuse change the watchdog initial state and disable behavior on ATtiny25/45/85?"),
    ("Stop 2 — triggers", "Can STM32L432xx timers trigger ADC/DAC conversions in Low-power Run and Low-power Sleep, and is that connection supported in Stop 2?"),
]
if not st.session_state.messages:
    st.markdown("##### Experimente uma pergunta")
    cols = st.columns(3)
    for col, (label, query) in zip(cols, examples):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_example = query

for item in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        show_response(item)

question = st.chat_input("Ex.: What is TIM2 counter resolution on STM32L432KC?")
if not question and st.session_state.pending_example:
    question = st.session_state.pending_example
    st.session_state.pending_example = ""
if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando os datasheets e verificando evidências..."):
                resources = get_resources(str(ROOT))
                state = resources.ask(
                    question, method=method, document=document, model=model.strip() or "qwen3:4b",
                    top_k=top_k, candidates_per_facet=candidates, num_ctx=num_ctx,
                    num_predict=num_predict, max_repairs=1 if repairs else 0,
                    table_first=table_first,
                )
            item = {
                "question": question, "result": state, "method": method,
                "model": model.strip() or "qwen3:4b", "index": st.session_state.next_index,
            }
            st.session_state.messages.append(item)
            st.session_state.next_index += 1
            show_response(item)
        except Exception as exc:
            message = str(exc)
            if "11434" in message or "Connection refused" in message or "Failed to establish" in message:
                message = "Não foi possível contatar o Ollama local. Abra o aplicativo Ollama e confirme 'ollama list'. Detalhes: " + message
            elif "storage folder" in message or "already accessed" in message or "already open" in message:
                message = "O banco Qdrant Local pode estar aberto em outro processo. Encerre outros scripts Python usando o índice. Detalhes: " + message
            item = {"question": question, "error": message, "index": st.session_state.next_index}
            st.session_state.messages.append(item)
            st.session_state.next_index += 1
            st.error(message)
