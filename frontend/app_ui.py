import streamlit as st
import requests
import json
import time

BACKEND_URL = "http://localhost:5000"

st.set_page_config(
    page_title="RAG Audit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0e0e0e; color: #e0e0e0; }

section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

.rag-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem; font-weight: 600; color: #c8ff00;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.2rem;
}
.rag-subheader {
    font-size: 0.78rem; color: #555;
    font-family: 'IBM Plex Mono', monospace; margin-bottom: 1rem;
}

/* Sidebar nav buttons */
.nav-btn-active button {
    background: #c8ff00 !important; color: #0e0e0e !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important;
    font-size: 0.72rem !important; letter-spacing: 0.1em !important;
    border: none !important; border-radius: 3px !important;
}
.nav-btn-inactive button {
    background: #1a1a1a !important; color: #666 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 400 !important;
    font-size: 0.72rem !important; letter-spacing: 0.1em !important;
    border: 1px solid #2a2a2a !important; border-radius: 3px !important;
}
.nav-btn-inactive button:hover { color: #aaa !important; background: #222 !important; }

/* Generic buttons */
.stButton button {
    background: #c8ff00 !important; color: #0e0e0e !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important;
    font-size: 0.75rem !important; letter-spacing: 0.08em !important;
    border: none !important; border-radius: 3px !important;
}
.stButton button:hover { background: #b0e000 !important; }

.delete-btn button {
    background: #1a0505 !important; color: #ff5555 !important;
    border: 1px solid #3a1010 !important; font-size: 0.7rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.delete-btn button:hover { background: #2a0808 !important; }

.confirm-btn button {
    background: #2a0808 !important; color: #ff3333 !important;
    border: 1px solid #5a1010 !important; font-size: 0.68rem !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important;
}

/* Chat messages */
.msg-user {
    background: #1a1a1a; border-left: 3px solid #c8ff00;
    padding: 10px 14px; margin: 8px 0;
    border-radius: 0 6px 6px 0; font-size: 0.9rem;
}
.msg-assistant {
    background: #111; border-left: 3px solid #333;
    padding: 10px 14px; margin: 8px 0;
    border-radius: 0 6px 6px 0; font-size: 0.9rem; line-height: 1.6;
}
.msg-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    color: #444; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
}

/* Source box */
.source-box {
    background: #0a0a0a; border: 1px solid #1e1e1e; border-radius: 4px;
    padding: 10px 14px; margin: 4px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
}
.source-header { color: #c8ff00; font-size: 0.68rem; margin-bottom: 4px; letter-spacing: 0.08em; }
.source-section { color: #555; font-size: 0.65rem; margin-bottom: 4px; font-style: italic; }
.source-excerpt {
    color: #888; font-size: 0.7rem; line-height: 1.5;
    border-left: 2px solid #1e1e1e; padding-left: 8px; margin-top: 6px;
    white-space: pre-wrap; word-break: break-word;
}

/* Upload / misc */
.upload-hint {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #444; margin-top: 0.5rem;
}
.step-line { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #888; padding: 2px 0; }

.stTextInput input, .stChatInput textarea {
    background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
    color: #e0e0e0 !important; font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 4px !important;
}

[data-testid="stFileUploader"] {
    background: #141414; border: 1px dashed #2a2a2a; border-radius: 6px; padding: 10px;
}

[data-testid="stMetric"] {
    background: #141414; border: 1px solid #1e1e1e; border-radius: 4px; padding: 8px 12px;
}
[data-testid="stMetricValue"] {
    color: #c8ff00 !important; font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: #555 !important; font-size: 0.72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stExpander"] {
    background: #141414 !important; border: 1px solid #1e1e1e !important; border-radius: 4px !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] { background: #1a1a1a !important; }
.stMultiSelect span { font-size: 0.72rem !important; }

hr { border-color: #1e1e1e !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = {}
if "page" not in st.session_state:
    st.session_state.page = "chat"   # "chat" | "params"

# ── Helpers ────────────────────────────────────────────────────────────────────
def fetch_documents():
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        return r.json() if r.ok else []
    except:
        return []

def fetch_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except:
        return None

def delete_document(doc_id):
    try:
        r = requests.delete(f"{BACKEND_URL}/documents/{doc_id}", timeout=10)
        return r.ok
    except:
        return False

TYPE_LABELS = {
    "declaration_tva": "TVA", "declaration_is": "IS",
    "bilan": "BILAN",         "contrat": "CONTRAT",
    "rapport_audit": "AUDIT", "facture": "FACTURE",
    "code_fiscal": "CODE",    "circulaire": "CIRC",
    "note_circulaire": "CIRC","autre": "DOC",
}
TYPE_COLORS = {
    "code_fiscal": "#c8ff00",    "circulaire": "#00d4ff",
    "note_circulaire": "#00d4ff","declaration_tva": "#ff9900",
    "declaration_is": "#ff6600", "bilan": "#aa88ff",
    "contrat": "#ff88aa",        "facture": "#88ffcc",
    "autre": "#666666",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="rag-header">⬡ RAG AUDIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subheader">document intelligence system</div>', unsafe_allow_html=True)

    # Health
    health = fetch_health()
    if health:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", health.get("chunks", 0))
        col2.metric("Docs", len(st.session_state.documents or fetch_documents()))
    else:
        st.error("⚠ Backend hors ligne — lance `python app.py`")

    st.markdown("---")

    # ── Navigation ─────────────────────────────────────────────────────────
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        cls = "nav-btn-active" if st.session_state.page == "chat" else "nav-btn-inactive"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("CHAT", use_container_width=True, key="nav_chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_nav2:
        cls = "nav-btn-active" if st.session_state.page == "params" else "nav-btn-inactive"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("PARAMÈTRES", use_container_width=True, key="nav_params"):
            st.session_state.page = "params"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Contenu sidebar selon page ─────────────────────────────────────────
    if st.session_state.page == "chat":

        # Upload
        st.markdown("**UPLOAD PDF**")
        uploaded = st.file_uploader("Glisse un PDF ici", type=["pdf"], label_visibility="collapsed")
        st.markdown('<div class="upload-hint">PDF texte ou scanné · tout type</div>', unsafe_allow_html=True)

        if uploaded:
            if st.button("INGÉRER →", use_container_width=True):
                with st.spinner(""):
                    try:
                        r = requests.post(
                            f"{BACKEND_URL}/ingest",
                            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                            timeout=10
                        )
                        if r.ok:
                            doc_id = r.json()["doc_id"]
                            pp = st.empty()
                            for _ in range(60):
                                time.sleep(1)
                                sr = requests.get(f"{BACKEND_URL}/ingest/status/{doc_id}", timeout=5)
                                if sr.ok:
                                    status = sr.json()
                                    with pp.container():
                                        for s in status.get("steps", []):
                                            st.markdown(f'<div class="step-line">{s}</div>', unsafe_allow_html=True)
                                    if status["status"] == "done":
                                        st.success(f"✓ {status.get('chunk_count',0)} chunks indexés")
                                        st.session_state.documents = fetch_documents()
                                        break
                                    elif status["status"] == "error":
                                        st.error(status.get("error", "Erreur inconnue"))
                                        break
                        else:
                            st.error("Erreur upload")
                    except Exception as e:
                        st.error(f"Connexion échouée: {e}")

        st.markdown("---")

        # Sélection documents
        if not st.session_state.documents:
            st.session_state.documents = fetch_documents()
        docs = st.session_state.documents

        st.markdown("**PÉRIMÈTRE**")
        if docs:
            doc_options_map = {}
            for d in docs:
                badge = TYPE_LABELS.get(d["type"], "DOC")
                label = f"{badge} · {d['filename']}"
                if d.get("exercice"):
                    label += f" ({d['exercice']})"
                doc_options_map[label] = d["doc_id"]

            selected_labels = st.multiselect(
                "Documents à interroger",
                options=list(doc_options_map.keys()),
                default=[],
                placeholder="Tous les documents",
                label_visibility="collapsed"
            )
            st.session_state.selected_doc_ids = [doc_options_map[l] for l in selected_labels]

            if st.session_state.selected_doc_ids:
                n = len(st.session_state.selected_doc_ids)
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.67rem;color:#c8ff00;">'
                    f'⬡ {n} doc{"s" if n>1 else ""} sélectionné{"s" if n>1 else ""}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-family:IBM Plex Mono,monospace;font-size:0.67rem;color:#444;">'
                    '⬡ tous les documents</div>', unsafe_allow_html=True
                )
        else:
            st.markdown('<div class="upload-hint">Aucun document indexé</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Quick prompts
        st.markdown("**CONTRÔLES RAPIDES**")
        for prompt in [
            "Vérifier la conformité TVA",
            "Analyser les anomalies comptables",
            "Contrôler les clauses contractuelles",
            "Résumer les points clés",
            "Détecter les risques fiscaux",
        ]:
            if st.button(prompt, use_container_width=True, key=f"quick_{prompt}"):
                st.session_state._quick_prompt = prompt

        st.markdown("---")
        if st.button("🗑 Effacer conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    else:  # page == "params"
        st.markdown("**OPTIONS**")
        if st.button("↻ Rafraîchir les docs", use_container_width=True):
            st.session_state.documents = fetch_documents()
            st.session_state.delete_confirm = {}
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── PAGE CHAT ─────────────────────────────────────────────────────────────────
if st.session_state.page == "chat":
    docs = st.session_state.documents or fetch_documents()

    # Header
    if st.session_state.selected_doc_ids and docs:
        sel_docs = [d for d in docs if d["doc_id"] in st.session_state.selected_doc_ids]
        names = " · ".join(d["filename"] for d in sel_docs[:3])
        if len(sel_docs) > 3:
            names += f" +{len(sel_docs)-3}"
        st.markdown(
            f'<div class="rag-header">ANALYSE · <span style="color:#888">{names}</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="rag-header">AUDIT CHATBOT · TOUS DOCUMENTS</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:4rem 0;color:#333;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2.5rem;margin-bottom:1rem;">⬡</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;letter-spacing:0.2em;color:#3a3a3a;">
                UPLOAD UN PDF · POSE UNE QUESTION
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-label">vous</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="msg-label">assistant</div>
                    {msg["content"].replace(chr(10), '<br>')}
                </div>""", unsafe_allow_html=True)

                # Source unique (top 1 reranker)
                sources = msg.get("sources", [])
                if sources:
                    top = sources[0]  # rank 1 = meilleur après reranking
                    color = TYPE_COLORS.get(top.get("type", "autre"), "#666")
                    badge = TYPE_LABELS.get(top.get("type", "?"), "DOC")
                    section = top.get("section", "")[:80]
                    excerpt = top.get("excerpt", "")[:500]
                    with st.expander("📎 Source", expanded=False):
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-header">
                                <span style="color:{color}">{badge}</span>
                                &nbsp;·&nbsp;
                                <span style="color:#bbb">{top.get('filename','?')}</span>
                            </div>
                            {f'<div class="source-section">{section}</div>' if section else ''}
                            <div class="source-excerpt">{excerpt}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Input
    default_input = ""
    if hasattr(st.session_state, "_quick_prompt"):
        default_input = st.session_state._quick_prompt
        del st.session_state._quick_prompt

    user_input = st.chat_input("Pose une question sur tes documents...")
    query = user_input or default_input

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        doc_filter = None
        if st.session_state.selected_doc_ids:
            if len(st.session_state.selected_doc_ids) == 1:
                doc_filter = {"doc_id": st.session_state.selected_doc_ids[0]}
            else:
                doc_filter = {"doc_ids": st.session_state.selected_doc_ids}

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": query, "history": st.session_state.messages[:-1], "doc_filter": doc_filter},
                    stream=True, timeout=60
                )

                full_response = ""
                sources_data  = []
                placeholder   = st.empty()

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "sources" in chunk:
                                    sources_data = chunk["sources"]
                                    continue
                                full_response += chunk.get("content", "")
                                placeholder.markdown(f"""
                                <div class="msg-assistant">
                                    <div class="msg-label">assistant</div>
                                    {full_response.replace(chr(10), '<br>')}▌
                                </div>""", unsafe_allow_html=True)
                            except:
                                pass

                placeholder.empty()
                st.session_state.messages.append({
                    "role": "assistant", "content": full_response, "sources": sources_data,
                })
                st.rerun()

            except requests.exceptions.ConnectionError:
                st.error("⚠ Backend non accessible.")
            except Exception as e:
                st.error(f"Erreur: {e}")


# ── PAGE PARAMÈTRES ───────────────────────────────────────────────────────────
elif st.session_state.page == "params":
    st.markdown('<div class="rag-header" style="margin-bottom:1.2rem;">⬡ PARAMÈTRES</div>',
                unsafe_allow_html=True)

    # Infos système
    with st.expander("ℹ️ Infos système", expanded=False):
        h = fetch_health()
        if h:
            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks total", h.get("chunks", 0))
            c2.metric("Store", h.get("store", "?"))
            c3.metric("LLM", "online" if "online" in h.get("llm_primary", "") else "offline")
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#555;margin-top:8px;line-height:1.8;">
                PRIMARY : {h.get('llm_primary','?')}<br>
                FALLBACK : {h.get('llm_fallback','?')}
            </div>""", unsafe_allow_html=True)
        else:
            st.error("Backend hors ligne")

    st.markdown("---")
    st.markdown("**DOCUMENTS INDEXÉS**")

    docs = st.session_state.documents or fetch_documents()
    st.session_state.documents = docs

    if not docs:
        st.markdown('<div class="upload-hint">Aucun document indexé.</div>', unsafe_allow_html=True)
    else:
        total_chunks = sum(d.get("chunk_count", 0) for d in docs)
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#555;margin-bottom:12px;">'
            f'{len(docs)} document{"s" if len(docs)>1 else ""} · {total_chunks} chunks total</div>',
            unsafe_allow_html=True
        )

        for doc in docs:
            did   = doc["doc_id"]
            badge = TYPE_LABELS.get(doc["type"], "DOC")
            color = TYPE_COLORS.get(doc["type"], "#666")

            col_info, col_btn = st.columns([6, 1])

            with col_info:
                st.markdown(f"""
                <div style="background:#141414;border:1px solid #1e1e1e;border-radius:4px;
                            padding:10px 14px;margin:3px 0;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                              background:#0a0a0a;border:1px solid #2a2a2a;border-radius:3px;
                              padding:2px 6px;color:{color};">{badge}</span>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                              color:#e0e0e0;font-weight:600;">{doc['filename']}</span>
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#555;">
                        {f"exercice: {doc['exercice']} &nbsp;·&nbsp; " if doc.get('exercice') else ""}
                        {f"société: {doc['societe']} &nbsp;·&nbsp; " if doc.get('societe') else ""}
                        langue: {doc.get('langue','?')} &nbsp;·&nbsp;
                        {doc.get('chunk_count','?')} chunks
                        {f'<br><span style="color:#3a3a3a;font-style:italic;">{doc["description"]}</span>' if doc.get('description') else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_btn:
                if st.session_state.delete_confirm.get(did):
                    st.markdown('<div class="confirm-btn">', unsafe_allow_html=True)
                    if st.button("✓ OUI", key=f"confirm_{did}", use_container_width=True):
                        ok = delete_document(did)
                        if ok:
                            st.session_state.documents = fetch_documents()
                            st.session_state.delete_confirm = {}
                            if did in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.remove(did)
                            st.rerun()
                        else:
                            st.error("Erreur")
                    st.markdown('</div>', unsafe_allow_html=True)
                    if st.button("✕", key=f"cancel_{did}", use_container_width=True):
                        st.session_state.delete_confirm[did] = False
                        st.rerun()
                else:
                    st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
                    if st.button("🗑", key=f"del_{did}", use_container_width=True):
                        st.session_state.delete_confirm[did] = True
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.63rem;color:#2a2a2a;">'
            '⚠ La suppression retire définitivement tous les chunks du document de Qdrant.</div>',
            unsafe_allow_html=True
        )