import os
import json
import uuid
import math
import re
import threading
import tempfile
import requests as http_requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


load_dotenv(override=True)

# ── Config flags ───────────────────────────────────────────────────────────────
USE_HYDE       = False
DEBUG          = os.getenv("DEBUG", "false").lower() == "true"

# Ollama — minimax-m3 (principal)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "minimax-m3:cloud")

# Groq — fallback
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

app = Flask(__name__)
CORS(app)

ingestion_status = {}

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        r = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        import groq as groq_lib
        return groq_lib.Groq(api_key=GROQ_API_KEY)
    except ImportError:
        return None


def llm_complete(messages: list[dict], max_tokens: int = 800,
                 temperature: float = 0.1) -> str:
    if _ollama_available():
        try:
            payload = {
                "model":   OLLAMA_MODEL,
                "messages": messages,
                "stream":  False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            r = http_requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120
            )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"⚠ Ollama error ({OLLAMA_MODEL}): {e} — bascule sur Groq")

    client = _groq_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠ Groq error: {e}")

    return "Erreur: aucun LLM disponible (Ollama hors-ligne et GROQ_API_KEY absente)."


def llm_stream(messages: list[dict], max_tokens: int = 800,
               temperature: float = 0.1):
    if _ollama_available():
        try:
            payload = {
                "model":    OLLAMA_MODEL,
                "messages": messages,
                "stream":   True,
                "options":  {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            with http_requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=180
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'content': token})}\n\n"
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            print(f"⚠ Ollama stream error: {e} — bascule sur Groq")

    client = _groq_client()
    if client:
        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            print(f"⚠ Groq stream error: {e}")

    yield f"data: {json.dumps({'content': 'Erreur: aucun LLM disponible.'})}\n\n"
    yield "data: [DONE]\n\n"


# ── Qdrant ─────────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "rag_docs"
DENSE_DIM  = 384  # E5-small

qdrant = QdrantClient(url=QDRANT_URL, timeout=30)

def ensure_collection(recreate=False):
    existing = [c.name for c in qdrant.get_collections().collections]
    if recreate and COLLECTION in existing:
        qdrant.delete_collection(COLLECTION)
        print(f"🗑 Collection '{COLLECTION}' supprimée")
        existing = []
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{COLLECTION}' créée (dim={DENSE_DIM})")
    else:
        info       = qdrant.get_collection(COLLECTION)
        actual_dim = info.config.params.vectors.size
        if actual_dim != DENSE_DIM:
            print(f"⚠ Dimension incorrecte ({actual_dim} vs {DENSE_DIM}) — recréation...")
            qdrant.delete_collection(COLLECTION)
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
            )
            print(f"✅ Collection '{COLLECTION}' recréée (dim={DENSE_DIM})")
        else:
            print(f"✅ Collection '{COLLECTION}' existante (dim={actual_dim})")

try:
    ensure_collection()
except Exception as e:
    print(f"⚠ Qdrant non accessible: {e}")
    print("  Lance: docker run -d -p 6333:6333 qdrant/qdrant")

# ── E5-small ───────────────────────────────────────────────────────────────────
print("⏳ Chargement E5-small...")
from sentence_transformers import SentenceTransformer
e5_model = SentenceTransformer("intfloat/multilingual-e5-small")
print("✅ E5-small prêt")

def embed_query(text: str) -> list[float]:
    vec = e5_model.encode(f"query: {text}", normalize_embeddings=True)
    return vec.tolist()

def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    vecs = e5_model.encode(prefixed, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return vecs.tolist()

# ── Reranker ───────────────────────────────────────────────────────────────────

def rerank(query, chunks, top_k=10):
    pairs  = [(query, c["chunk_text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    if DEBUG:
        print(f"\n{'─'*60}")
        print(f"🎯 RERANKER TOP 5")
        print(f"{'─'*60}")
        for i, (score, c) in enumerate(ranked[:5]):
            print(f"  #{i+1} score={score:.4f} | section=[{c.get('section','?')[:40]}]")
            print(f"       texte: {c['chunk_text'][:150].replace(chr(10),' ')}...")
    return [c for _, c in ranked[:top_k]]

def clean_footnotes(text: str) -> str:
    text = re.sub(
        r'(?:\n\s*Article\s+\d+\s+de\s+la\s+loi\s+de\s+finances[^\n]*\n\s*\d+\s*)+',
        '\n',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    return text

# ── PDF Parsing ────────────────────────────────────────────────────────────────
def parse_pdf(filepath):
    import fitz
    doc   = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text   = page.get_text("text")
        blocks = page.get_text("blocks")
        tables = extract_table_like_blocks(blocks)
        pages.append({
            "page":       i + 1,
            "text":       text,
            "tables":     tables,
            "char_count": len(text.strip())
        })
    doc.close()
    return pages

def extract_table_like_blocks(blocks):
    tables = []
    for b in blocks:
        if b[6] == 0:
            text  = b[4]
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            numeric_lines = sum(1 for l in lines if re.search(r'\d+[\s,\.]\d*', l))
            if len(lines) >= 3 and numeric_lines >= 2:
                tables.append(text)
    return tables

def detect_pdf_quality(pages):
    total = sum(p["char_count"] for p in pages)
    return "selectable" if total > 200 else "scanned"

def is_toc_page(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return False
    dot_lines  = sum(1 for l in lines if re.search(r'\.{4,}', l))
    page_lines = sum(1 for l in lines if re.fullmatch(r'\d{1,4}', l))
    ratio = (dot_lines + page_lines) / len(lines)
    return ratio > 0.5

def filter_toc_pages(pages):
    filtered = [p for p in pages if not is_toc_page(p["text"])]
    skipped  = len(pages) - len(filtered)
    if skipped > 0:
        print(f"🗑 {skipped} page(s) TDM ignorée(s)")
    return filtered

# ── Classifier ─────────────────────────────────────────────────────────────────
KEYWORD_RULES = [
    ("code_fiscal", [
        "code général des impôts", "code des impôts", "code douanier",
        "code du travail", "code de commerce", "dahir portant loi",
        "livre premier", "livre deuxième", "titre premier", "titre ii",
        "article 1er", "article premier",
    ]),
    ("circulaire",      ["circulaire n°", "note circulaire", "note de service", "direction générale des impôts"]),
    ("declaration_tva", ["déclaration de la tva", "taxe sur la valeur ajoutée", "déclaration tva", "formulaire tva"]),
    ("declaration_is",  ["déclaration de l'impôt sur les sociétés", "déclaration is", "formulaire is"]),
    ("bilan",           ["bilan comptable", "compte de résultat", "état des soldes", "bilan de clôture", "actif immobilisé"]),
    ("contrat",         ["entre les soussignés", "il a été convenu", "contrat de", "entre les parties"]),
    ("facture",         ["facture n°", "facture pro forma", "bon de commande", "montant ttc", "montant ht"]),
]

def keyword_classify(pages) -> str | None:
    sample = " ".join(p["text"][:500] for p in pages[:5]).lower()
    for doc_type, keywords in KEYWORD_RULES:
        if any(kw in sample for kw in keywords):
            return doc_type
    return None

def classify_document(pages):
    detected = keyword_classify(pages)
    if detected:
        print(f"🔍 Classification par mots-clés: {detected}")
        return {"type": detected, "langue": "fr", "exercice": None, "societe": None, "description": ""}

    sample = ""
    for p in pages[:3]:
        sample += p["text"][:300]
        if len(sample) > 800:
            break

    prompt = (
        "Analyse ce début de document. Retourne UNIQUEMENT un JSON valide, zéro markdown.\n"
        'Champs: type ("declaration_tva"|"declaration_is"|"bilan"|"contrat"|"rapport_audit"|"facture"|"code_fiscal"|"circulaire"|"note_circulaire"|"autre"), '
        'langue ("fr"|"ar"|"en"|"mixed"), exercice (string année ou null), '
        'societe (string ou null), description (1 phrase courte).\n'
        'code_fiscal = tout code de loi structuré par articles.\n'
        'circulaire/note_circulaire = instructions fiscales structurées par articles.\n\n'
        f"Document:\n{sample[:800]}"
    )

    try:
        raw    = llm_complete([{"role": "user", "content": prompt}], max_tokens=150, temperature=0.0)
        raw    = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        print(f"🔍 Classification LLM: {result}")
        return result
    except Exception as e:
        print(f"Classifier error: {e}")
        return {"type": "autre", "langue": "fr", "exercice": None, "societe": None, "description": ""}

# ── Chunker ────────────────────────────────────────────────────────────────────
def chunk_document(pages, doc_meta):
    t = doc_meta.get("type", "autre")
    if t in ("declaration_tva", "declaration_is"):
        chunks = chunk_fiscal(pages, doc_meta)
    elif t == "bilan":
        chunks = chunk_bilan(pages, doc_meta)
    elif t in ("contrat", "code_fiscal", "circulaire", "note_circulaire"):
        chunks = chunk_by_article(pages, doc_meta)
    else:
        chunks = chunk_generic(pages, doc_meta)
    if len(chunks) < 3:
        chunks = chunk_generic(pages, doc_meta)
    return chunks

def _make_chunk(text, meta, index, section):
    return {"chunk_id": str(uuid.uuid4()), "chunk_text": text,
            "chunk_index": index, "section": section, **meta}

def chunk_by_article(pages, meta):
    full = "\n".join(p["text"] for p in pages)
    full = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', full)
    full = re.sub(r'\n{3,}', '\n\n', full)
    full = clean_footnotes(full)

    split_pattern = r'(?=(?:Article|ARTICLE|Art\.)\s*\.?\s*\d{1,4}\b)'
    raw_parts = re.split(split_pattern, full)

    clean_parts = []
    for part in raw_parts:
        part = part.strip()
        if len(part) < 20:
            continue
        lines     = [l.strip() for l in part.split("\n") if l.strip()]
        dot_ratio = sum(1 for l in lines if re.search(r'\.{4,}', l)) / max(len(lines), 1)
        if dot_ratio > 0.4:
            continue
        clean_parts.append(part)

    merged = []
    for part in clean_parts:
        is_article = bool(re.match(r'(?:Article|ARTICLE|Art\.)\s*\.?\s*\d+', part.strip()))
        if is_article:
            merged.append(part)
        elif not is_article and merged:
            merged[-1] = merged[-1] + "\n" + part
        else:
            merged.append(part)

    chunks = []
    for i, part in enumerate(merged):
        part  = part.strip()
        title = extract_section_title(part)
        words = part.split()

        if len(words) <= 1500:
            chunks.append(_make_chunk(part, meta, i, title))
        else:
            for j, sc in enumerate(sliding_window(part, 1200, 100)):
                contextualized = f"{title}\n{sc}" if j > 0 else sc
                chunks.append(_make_chunk(contextualized, meta, i * 100 + j, title))

    return chunks if len(chunks) >= 3 else chunk_generic(pages, meta)

def chunk_fiscal(pages, meta):
    patterns = [
        r"TVA\s+collect[eé]e", r"TVA\s+d[eé]ductible", r"Base\s+imposable",
        r"Chiffre\s+d.affaires", r"Article\s+\d+",
        r"(?:Section|Chapitre|Titre)\s+[IVX\d]+"
    ]
    full  = "".join(f"\n[Page {p['page']}]\n{p['text']}" for p in pages)
    parts = re.split(r'(?=(?:' + '|'.join(patterns) + r'))', full, flags=re.MULTILINE)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < 50:
            continue
        for j, sc in enumerate(sliding_window(part, 600, 100)):
            chunks.append(_make_chunk(sc, meta, i * 100 + j, extract_section_title(part)))
    return chunks or chunk_generic(pages, meta)

def chunk_bilan(pages, meta):
    kw    = ["ACTIF","PASSIF","CAPITAUX PROPRES","DETTES","IMMOBILISATIONS",
             "STOCKS","CRÉANCES","TRÉSORERIE","RÉSULTAT"]
    full  = "\n".join(p["text"] for p in pages)
    parts = re.split(r'(?=(?:' + '|'.join(kw) + r'))', full)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < 30:
            continue
        for j, sc in enumerate(sliding_window(part, 500, 80)):
            chunks.append(_make_chunk(sc, meta, i * 100 + j, part[:60]))
    return chunks or chunk_generic(pages, meta)

def chunk_generic(pages, meta):
    full   = "\n".join(p["text"] for p in pages)
    chunks = []
    for i, sc in enumerate(sliding_window(full, 600, 120)):
        if len(sc.strip()) < 30:
            continue
        chunks.append(_make_chunk(sc, meta, i, f"Section {i+1}"))
    return chunks

def sliding_window(text, size=600, overlap=400):
    words = text.split()
    step  = max(1, size - overlap)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step) if words[i:i+size]]

def extract_section_title(text):
    line = text.strip().split("\n")[0]
    return line[:80] if line else "Section"

# ── BM25 ───────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def bm25_score(query_tokens, doc_tokens, corpus_size, avg_dl, k1=1.5, b=0.75):
    dl   = len(doc_tokens)
    freq = {}
    for t in doc_tokens:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        f = freq.get(qt, 0)
        if f == 0:
            continue
        idf    = math.log(1 + (corpus_size - f + 0.5) / (f + 0.5))
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score

def get_all_chunks(doc_filter=None):
    qdrant_filter = None
    if doc_filter:
        conditions    = [FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                         for k, v in doc_filter.items()]
        qdrant_filter = Filter(must=conditions)
    chunks = []
    offset = None
    while True:
        result, offset = qdrant.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            scroll_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False
        )
        for point in result:
            p = point.payload
            chunks.append({
                "chunk_id":    p.get("chunk_id"),
                "chunk_text":  p.get("chunk_text", ""),
                "section":     p.get("section", ""),
                "chunk_index": p.get("chunk_index", 0),
                "metadata":    p.get("metadata", {})
            })
        if offset is None:
            break
    return chunks

# ── RRF ────────────────────────────────────────────────────────────────────────
def rrf_fusion(bm25_results, vector_results, bm25_weight=1.5, k=60):
    scores = {}
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        if cid not in scores:
            scores[cid] = {"chunk": chunk, "score": 0.0}
        scores[cid]["score"] += bm25_weight * (1.0 / (rank + k))
    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        if cid not in scores:
            scores[cid] = {"chunk": chunk, "score": 0.0}
        scores[cid]["score"] += 1.0 * (1.0 / (rank + k))
    return [item["chunk"] for item in sorted(scores.values(), key=lambda x: x["score"], reverse=True)]

# ── HyDE ───────────────────────────────────────────────────────────────────────
def hyde_rewrite(query: str) -> str:
    prompt = (
        "Tu es un expert en fiscalité marocaine (CGI). "
        "Donne 6 à 8 mots-clés juridiques et fiscaux exacts du CGI marocain "
        "correspondant à cette question. "
        "Réponds UNIQUEMENT avec les mots-clés séparés par des espaces, sans ponctuation.\n\n"
        f"Question : {query}"
    )
    try:
        rewritten = llm_complete([{"role": "user", "content": prompt}], max_tokens=80, temperature=0.0)
        rewritten = rewritten.strip()
        print(f"🔄 HyDE: '{query}' → '{rewritten}'")
        return rewritten if rewritten else query
    except Exception as e:
        print(f"HyDE error: {e}")
        return query

# ── Retrieval hybride ──────────────────────────────────────────────────────────
def retrieve_chunks(query, doc_filter=None, top_k=15):
    search_query = hyde_rewrite(query) if USE_HYDE else query

    if DEBUG:
        print(f"\n{'='*60}")
        print(f"🔎 QUERY          : {query}")
        if USE_HYDE:
            print(f"🔄 HYDE REWRITTEN : {search_query}")
        print(f"{'='*60}")

    all_chunks = []
    try:
        all_chunks = get_all_chunks(doc_filter)
    except Exception as e:
        print(f"Scroll error: {e}")

    if DEBUG:
        print(f"📚 Corpus total   : {len(all_chunks)} chunks")

    bm25_top20 = []
    if all_chunks:
        query_tokens = tokenize(search_query)
        avg_dl       = sum(len(tokenize(c["chunk_text"])) for c in all_chunks) / len(all_chunks)
        corpus_size  = len(all_chunks)

        scored = []
        for chunk in all_chunks:
            doc_tokens = tokenize(chunk["chunk_text"])
            score      = bm25_score(query_tokens, doc_tokens, corpus_size, avg_dl)
            if search_query.lower() in chunk["chunk_text"].lower():
                score *= 2.0
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        bm25_top20 = [c for _, c in scored[:20]]

    vector_top20 = []
    try:
        qdrant_filter = None
        if doc_filter:
            conditions    = [FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                             for k, v in doc_filter.items()]
            qdrant_filter = Filter(must=conditions)
        query_vec = embed_query(search_query)
        results   = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=20,
            with_payload=True
        )
        for r in results.points:
            p = r.payload
            vector_top20.append({
                "chunk_id":   p.get("chunk_id"),
                "chunk_text": p.get("chunk_text", ""),
                "section":    p.get("section", ""),
                "metadata":   p.get("metadata", {}),
                "_score":     r.score
            })
    except Exception as e:
        print(f"Vector search error: {e}")

    if not bm25_top20 and not vector_top20:
        return []
    if not vector_top20:
        return bm25_top20[:top_k]
    if not bm25_top20:
        return vector_top20[:top_k]

    fused = rrf_fusion(bm25_top20, vector_top20, bm25_weight=1.5)
    return rerank(query, fused[:30], top_k=top_k)


# ── Ingestion ──────────────────────────────────────────────────────────────────
@app.route("/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file     = request.files["file"]
    doc_id   = str(uuid.uuid4())
    tmp_dir  = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"{doc_id}.pdf")
    file.save(tmp_path)
    filename = file.filename

    ingestion_status[doc_id] = {"status": "processing", "steps": [], "doc_id": doc_id}

    def process():
        try:
            ingestion_status[doc_id]["steps"].append("📄 Parsing du PDF...")
            pages   = parse_pdf(tmp_path)
            quality = detect_pdf_quality(pages)
            ingestion_status[doc_id]["steps"].append(
                f"✅ PDF parsé — qualité: {quality} ({len(pages)} pages)")

            pages = filter_toc_pages(pages)
            ingestion_status[doc_id]["steps"].append(
                f"✅ Pages utiles après filtre TDM: {len(pages)}")

            ingestion_status[doc_id]["steps"].append("🔍 Classification...")
            doc_meta             = classify_document(pages)
            doc_meta["doc_id"]   = doc_id
            doc_meta["filename"] = filename
            ingestion_status[doc_id]["steps"].append(
                f"✅ Type: {doc_meta.get('type','?')} | Langue: {doc_meta.get('langue','?')} | Exercice: {doc_meta.get('exercice','N/A')}")

            ingestion_status[doc_id]["steps"].append("✂️ Chunking adaptatif...")
            chunks = chunk_document(pages, doc_meta)
            ingestion_status[doc_id]["steps"].append(
                f"✅ {len(chunks)} chunks (stratégie: {doc_meta.get('type','générique')})")

            ingestion_status[doc_id]["steps"].append(
                f"🧠 Embedding E5-small ({len(chunks)} chunks)...")
            texts = [c["chunk_text"] for c in chunks]
            vecs  = embed_passages(texts)
            ingestion_status[doc_id]["steps"].append("✅ Embeddings calculés")

            ingestion_status[doc_id]["steps"].append("📦 Indexation Qdrant...")
            points = [
                PointStruct(
                    id      = chunk["chunk_id"],
                    vector  = vecs[i],
                    payload = {
                        "chunk_id":    chunk["chunk_id"],
                        "chunk_text":  chunk["chunk_text"],
                        "section":     chunk.get("section", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "metadata":    doc_meta
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            for i in range(0, len(points), 64):
                qdrant.upsert(collection_name=COLLECTION, points=points[i:i+64])

            total = qdrant.count(collection_name=COLLECTION).count
            ingestion_status[doc_id]["steps"].append(
                f"✅ Indexé dans Qdrant ({len(chunks)} chunks · {total} total)")

            ingestion_status[doc_id]["status"]      = "done"
            ingestion_status[doc_id]["doc_meta"]    = doc_meta
            ingestion_status[doc_id]["chunk_count"] = len(chunks)

        except Exception as e:
            ingestion_status[doc_id]["status"] = "error"
            ingestion_status[doc_id]["error"]  = str(e)
            print(f"Ingestion error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    threading.Thread(target=process).start()
    return jsonify({"doc_id": doc_id, "status": "processing"})

@app.route("/ingest/status/<doc_id>", methods=["GET"])
def ingest_status(doc_id):
    s = ingestion_status.get(doc_id)
    if not s:
        return jsonify({"error": "Not found"}), 404
    return jsonify(s)


# ── Chat ───────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant fiscal marocain. Tu réponds UNIQUEMENT à partir du contexte fourni.

FORMAT OBLIGATOIRE : réponse directe en 1-2 phrases (chiffre, taux, oui/non + article). Si besoin, 1 seule précision, pas plus. Si absent du contexte : "Information non présente dans les documents."

INTERDIT : markdown (**gras**, *italique*, ##titres), listes à puces, reformulation, conclusion, connaissances externes. Jamais plus de 3 phrases. Texte brut uniquement."""

@app.route("/chat", methods=["POST"])
def chat():
    data        = request.json
    query       = data.get("query", "").strip()
    history     = data.get("history", [])
    doc_filter  = data.get("doc_filter")   # peut être None, dict, ou list de doc_ids

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # ── Gestion multi-doc filter ───────────────────────────────────────────
    # doc_filter peut être:
    #   None               → tous les docs
    #   {"doc_id": "xxx"}  → un seul doc (legacy)
    #   {"doc_ids": [...]} → plusieurs docs sélectionnés
    qdrant_filter_obj = None
    bm25_doc_filter   = None

    if isinstance(doc_filter, dict) and "doc_ids" in doc_filter:
        doc_ids = doc_filter["doc_ids"]
        if len(doc_ids) == 1:
            bm25_doc_filter = {"doc_id": doc_ids[0]}
        elif len(doc_ids) > 1:
            # Pour BM25 : on filtre après scroll (plus simple)
            bm25_doc_filter = {"doc_ids": doc_ids}
    elif isinstance(doc_filter, dict) and "doc_id" in doc_filter:
        bm25_doc_filter = doc_filter

    chunks = retrieve_chunks_multi(query, bm25_doc_filter, top_k=10)

    if not chunks:
        context = "Aucun document indexé ou aucun résultat pertinent trouvé."
    else:
        context = "\n\n---\n\n".join([
            f"[{c['metadata'].get('type','?')} | {c['metadata'].get('filename','?')} | {c.get('section','')}]\n{c['chunk_text']}"
            for c in chunks
        ])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-3:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({
        "role": "user",
        "content": f"CONTEXTE DOCUMENTAIRE :\n{context}\n\nQUESTION : {query}"
    })

    # Prépare les sources à envoyer dans le premier event SSE
    sources = []
    for i, c in enumerate(chunks):
        sources.append({
            "rank":     i + 1,
            "filename": c["metadata"].get("filename", "?"),
            "type":     c["metadata"].get("type", "?"),
            "section":  c.get("section", ""),
            "excerpt":  c["chunk_text"][:400],
        })

    def stream_with_sources():
        # Envoie les sources en premier event
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        # Puis stream la réponse LLM
        yield from llm_stream(messages, max_tokens=800, temperature=0.1)

    return Response(
        stream_with_context(stream_with_sources()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )


def retrieve_chunks_multi(query, doc_filter=None, top_k=15):
    """Comme retrieve_chunks mais supporte doc_filter avec doc_ids (liste)."""
    search_query = hyde_rewrite(query) if USE_HYDE else query

    all_chunks = []
    try:
        # Pour multi-doc : scroll sans filtre puis filtre en Python
        if isinstance(doc_filter, dict) and "doc_ids" in doc_filter:
            doc_ids_set = set(doc_filter["doc_ids"])
            raw_chunks  = get_all_chunks(None)
            all_chunks  = [c for c in raw_chunks if c["metadata"].get("doc_id") in doc_ids_set]
        else:
            all_chunks = get_all_chunks(doc_filter)
    except Exception as e:
        print(f"Scroll error: {e}")

    bm25_top20 = []
    if all_chunks:
        query_tokens = tokenize(search_query)
        avg_dl       = sum(len(tokenize(c["chunk_text"])) for c in all_chunks) / len(all_chunks)
        corpus_size  = len(all_chunks)
        scored = []
        for chunk in all_chunks:
            doc_tokens = tokenize(chunk["chunk_text"])
            score      = bm25_score(query_tokens, doc_tokens, corpus_size, avg_dl)
            if search_query.lower() in chunk["chunk_text"].lower():
                score *= 2.0
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        bm25_top20 = [c for _, c in scored[:20]]

    vector_top20 = []
    try:
        # Pour vector search multi-doc : on ne peut pas faire OR natif facilement dans Qdrant
        # → on récupère top 20 sans filtre puis on filtre
        query_vec = embed_query(search_query)

        if isinstance(doc_filter, dict) and "doc_ids" in doc_filter:
            doc_ids_set = set(doc_filter["doc_ids"])
            results = qdrant.query_points(
                collection_name=COLLECTION,
                query=query_vec,
                limit=50,
                with_payload=True
            )
            for r in results.points:
                p = r.payload
                if p.get("metadata", {}).get("doc_id") in doc_ids_set:
                    vector_top20.append({
                        "chunk_id":   p.get("chunk_id"),
                        "chunk_text": p.get("chunk_text", ""),
                        "section":    p.get("section", ""),
                        "metadata":   p.get("metadata", {}),
                        "_score":     r.score
                    })
                    if len(vector_top20) >= 20:
                        break
        else:
            qdrant_filter = None
            if doc_filter and "doc_id" in doc_filter:
                qdrant_filter = Filter(must=[
                    FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_filter["doc_id"]))
                ])
            results = qdrant.query_points(
                collection_name=COLLECTION,
                query=query_vec,
                query_filter=qdrant_filter,
                limit=20,
                with_payload=True
            )
            for r in results.points:
                p = r.payload
                vector_top20.append({
                    "chunk_id":   p.get("chunk_id"),
                    "chunk_text": p.get("chunk_text", ""),
                    "section":    p.get("section", ""),
                    "metadata":   p.get("metadata", {}),
                    "_score":     r.score
                })
    except Exception as e:
        print(f"Vector search error: {e}")

    if not bm25_top20 and not vector_top20:
        return []
    if not vector_top20:
        return bm25_top20[:top_k]
    if not bm25_top20:
        return vector_top20[:top_k]

    fused = rrf_fusion(bm25_top20, vector_top20, bm25_weight=1.5)
    return rerank(query, fused[:30], top_k=top_k)


# ── Documents list ─────────────────────────────────────────────────────────────
@app.route("/documents", methods=["GET"])
def list_documents():
    try:
        ensure_collection()
        seen   = {}
        offset = None
        while True:
            result, offset = qdrant.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for point in result:
                meta = point.payload.get("metadata", {})
                did  = meta.get("doc_id")
                if did and did not in seen:
                    seen[did] = {
                        "doc_id":      did,
                        "filename":    meta.get("filename", "?"),
                        "type":        meta.get("type", "?"),
                        "langue":      meta.get("langue", "?"),
                        "exercice":    meta.get("exercice"),
                        "societe":     meta.get("societe"),
                        "description": meta.get("description", ""),
                        "chunk_count": 0,
                    }
                if did and did in seen:
                    seen[did]["chunk_count"] += 1
            if offset is None:
                break
        return jsonify(list(seen.values()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Delete document ────────────────────────────────────────────────────────────
@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Supprime tous les chunks d'un document de Qdrant."""
    try:
        from qdrant_client.models import FilterSelector
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))
                ])
            )
        )
        # Retire aussi de ingestion_status si présent
        if doc_id in ingestion_status:
            del ingestion_status[doc_id]
        count_after = qdrant.count(collection_name=COLLECTION).count
        print(f"🗑 Document {doc_id} supprimé. Total restant: {count_after} chunks")
        return jsonify({"success": True, "doc_id": doc_id, "chunks_total": count_after})
    except Exception as e:
        print(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Health ─────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    ollama_ok = _ollama_available()
    groq_ok   = bool(GROQ_API_KEY)
    try:
        count = qdrant.count(collection_name=COLLECTION).count
        return jsonify({
            "status":      "ok",
            "chunks":      count,
            "docs":        len(ingestion_status),
            "store":       "qdrant+e5",
            "llm_primary": f"ollama/{OLLAMA_MODEL} ({'online' if ollama_ok else 'offline'})",
            "llm_fallback": f"groq/llama-3.3-70b ({'configured' if groq_ok else 'missing key'})",
        })
    except Exception as e:
        return jsonify({"status": "qdrant_offline", "error": str(e)}), 200



# ── Route /retrieve — à ajouter dans app.py ───────────────────────────────────
# Colle ce bloc juste avant :  if __name__ == "__main__":

@app.route("/retrieve", methods=["POST"])
def retrieve():
    """
    Endpoint de retrieval pur — utilisé par ragas_eval.py et eval.py.
    Retourne les chunks sans passer par le LLM.

    Body JSON :
        query      : str
        top_k      : int (défaut 10)
        doc_filter : dict ou None  ex: {"doc_id": "xxx"}
    """
    data       = request.json
    query      = data.get("query", "").strip()
    top_k      = data.get("top_k", 10)
    doc_filter = data.get("doc_filter")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        chunks = retrieve_chunks_multi(query, doc_filter, top_k=top_k)
        return jsonify(chunks)
    except Exception as e:
        print(f"Retrieve error: {e}")
        return jsonify([]), 500



if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)