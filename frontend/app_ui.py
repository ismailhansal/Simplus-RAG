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

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0e0e0e;
    color: #e0e0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

/* Main header */
.rag-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #c8ff00;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.rag-subheader {
    font-size: 0.78rem;
    color: #555;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 1.5rem;
}

/* Doc type badge */
.doc-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #c8ff00;
    margin: 2px;
}

/* Chat messages */
.msg-user {
    background: #1a1a1a;
    border-left: 3px solid #c8ff00;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
}
.msg-assistant {
    background: #111;
    border-left: 3px solid #333;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.msg-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}

/* Upload zone */
.upload-hint {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    margin-top: 0.5rem;
}

/* Status steps */
.step-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888;
    padding: 2px 0;
}

/* Input box override */
.stTextInput input, .stChatInput textarea {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 4px !important;
}

/* Buttons */
.stButton button {
    background: #c8ff00 !important;
    color: #0e0e0e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 3px !important;
}
.stButton button:hover {
    background: #b0e000 !important;
}

/* Divider */
hr { border-color: #1e1e1e !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141414;
    border: 1px dashed #2a2a2a;
    border-radius: 6px;
    padding: 10px;
}

/* Selectbox */
.stSelectbox select, [data-testid="stSelectbox"] {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a2a !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 8px 12px;
}
[data-testid="stMetricValue"] {
    color: #c8ff00 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: #555 !important;
    font-size: 0.72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

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

TYPE_LABELS = {
    "declaration_tva": "TVA",
    "declaration_is": "IS",
    "bilan": "BILAN",
    "contrat": "CONTRAT",
    "rapport_audit": "AUDIT",
    "facture": "FACTURE",
    "autre": "DOC",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="rag-header">⬡ RAG AUDIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subheader">document intelligence system</div>', unsafe_allow_html=True)

    # Health check
    health = fetch_health()
    if health:
        col1, col2 = st.columns(2)
        col1.metric("Chunks", health.get("chunks", 0))
        col2.metric("Docs", health.get("docs", 0))
    else:
        st.error("⚠ Backend hors ligne — lance `python app.py`", icon=None)

    st.markdown("---")

    # Upload section
    st.markdown("**UPLOAD PDF**")
    uploaded = st.file_uploader(
        "Glisse un PDF ici",
        type=["pdf"],
        label_visibility="collapsed"
    )
    st.markdown('<div class="upload-hint">PDF texte ou scanné · tout type de document</div>', unsafe_allow_html=True)

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
                        # Poll for completion
                        progress_placeholder = st.empty()
                        for _ in range(60):
                            time.sleep(1)
                            status_r = requests.get(f"{BACKEND_URL}/ingest/status/{doc_id}", timeout=5)
                            if status_r.ok:
                                status = status_r.json()
                                steps = status.get("steps", [])
                                with progress_placeholder.container():
                                    for s in steps:
                                        st.markdown(f'<div class="step-line">{s}</div>', unsafe_allow_html=True)
                                if status["status"] == "done":
                                    st.success(f"✓ {status.get('chunk_count', 0)} chunks indexés")
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

    # Document selector
    st.markdown("**DOCUMENTS INDEXÉS**")
    if st.button("↻ Rafraîchir", use_container_width=True):
        st.session_state.documents = fetch_documents()

    docs = st.session_state.documents or fetch_documents()
    st.session_state.documents = docs

    if docs:
        doc_options = {"Tous les documents": None}
        for d in docs:
            label = f"{TYPE_LABELS.get(d['type'], 'DOC')} · {d['filename']}"
            if d.get("exercice"):
                label += f" ({d['exercice']})"
            doc_options[label] = d["doc_id"]

        selected_label = st.selectbox(
            "Filtrer par document",
            options=list(doc_options.keys()),
            label_visibility="collapsed"
        )
        st.session_state.selected_doc = doc_options[selected_label]

        # Show doc details
        if st.session_state.selected_doc:
            selected_meta = next((d for d in docs if d["doc_id"] == st.session_state.selected_doc), None)
            if selected_meta:
                with st.expander("Détails", expanded=False):
                    if selected_meta.get("societe"):
                        st.markdown(f"**Société:** {selected_meta['societe']}")
                    if selected_meta.get("exercice"):
                        st.markdown(f"**Exercice:** {selected_meta['exercice']}")
                    if selected_meta.get("langue"):
                        st.markdown(f"**Langue:** {selected_meta['langue']}")
                    if selected_meta.get("description"):
                        st.markdown(f"**Résumé:** {selected_meta['description']}")
    else:
        st.markdown('<div class="upload-hint">Aucun document — uploadez un PDF</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick audit prompts
    st.markdown("**CONTRÔLES RAPIDES**")
    audit_prompts = [
        "Vérifier la conformité TVA",
        "Analyser les anomalies comptables",
        "Contrôler les clauses contractuelles",
        "Résumer les points clés du document",
        "Détecter les risques fiscaux",
    ]
    for prompt in audit_prompts:
        if st.button(prompt, use_container_width=True, key=f"quick_{prompt}"):
            st.session_state._quick_prompt = prompt

    st.markdown("---")
    if st.button("🗑 Effacer conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    if st.session_state.selected_doc and docs:
        sel = next((d for d in docs if d["doc_id"] == st.session_state.selected_doc), None)
        if sel:
            badge = TYPE_LABELS.get(sel["type"], "DOC")
            st.markdown(
                f'<div class="rag-header">ANALYSE · <span style="color:#888">{sel["filename"]}</span> '
                f'<span class="doc-badge">{badge}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="rag-header">AUDIT CHATBOT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-header">AUDIT CHATBOT · TOUS DOCUMENTS</div>', unsafe_allow_html=True)

st.markdown("---")

# Chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: #333;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 2rem; margin-bottom: 1rem;">⬡</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.2em; color: #444;">
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
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="msg-label">assistant</div>
                    {msg["content"].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
# Handle quick prompts
default_input = ""
if hasattr(st.session_state, "_quick_prompt"):
    default_input = st.session_state._quick_prompt
    del st.session_state._quick_prompt

user_input = st.chat_input("Pose une question sur tes documents...")

# Use quick prompt if set
query = user_input or default_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    doc_filter = None
    if st.session_state.selected_doc:
        doc_filter = {"doc_id": st.session_state.selected_doc}

    # Stream response
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "query": query,
                    "history": st.session_state.messages[:-1],
                    "doc_filter": doc_filter
                },
                stream=True,
                timeout=60
            )

            full_response = ""
            placeholder = st.empty()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            full_response += chunk.get("content", "")
                            placeholder.markdown(f"""
                            <div class="msg-assistant">
                                <div class="msg-label">assistant</div>
                                {full_response.replace(chr(10), '<br>')}▌
                            </div>
                            """, unsafe_allow_html=True)
                        except:
                            pass

            placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("⚠ Backend non accessible. Lance `python backend/app.py` d'abord.")
        except Exception as e:
            st.error(f"Erreur: {e}")
