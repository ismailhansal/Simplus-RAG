"""
ragas_eval.py
═══════════════════════════════════════════════════════════════
Module RAGAS pour le pipeline RAG CGI.

LLM évaluateur   : Ollama minimax-m3:cloud (même modèle que app.py)
Embeddings       : intfloat/multilingual-e5-small (même modèle que app.py)
                   → via SentenceTransformerEmbeddings (pas besoin d'Ollama
                     pour les embeddings)

Métriques implémentées :
  - faithfulness          : réponse fidèle aux chunks ?
  - answer_relevancy      : réponse pertinente à la question ?
  - context_precision     : chunks récupérés utiles ?
  - context_recall        : chunks suffisants pour répondre ?

Usage autonome :
    python ragas_eval.py                   # toutes les questions
    python ragas_eval.py --tier 1          # seulement tier 1
    python ragas_eval.py --ids q01,q04     # questions précises
    python ragas_eval.py --analyze         # analyse dernier résultat RAGAS

Intégration dans eval.py :
    from ragas_eval import score_with_ragas, build_ragas_llm, build_ragas_embeddings
═══════════════════════════════════════════════════════════════
"""

import argparse
import json
import time
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── RAGAS imports ──────────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── LangChain : LLM Ollama + embeddings SentenceTransformer ───────────────────
# LLM  → Ollama minimax-m3:cloud  (même modèle que app.py)
# Emb  → SentenceTransformer E5-small (même modèle que app.py, déjà chargé)
#         On évite OllamaEmbeddings car E5-small est local et déjà disponible.
from langchain_community.chat_models import ChatOllama
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Import depuis ton eval.py existant ────────────────────────────────────────
# On réutilise la config, le logger et les fonctions de call
try:
    from eval import (
        BACKEND_URL,
        LOG_DIR,
        RESULT_DIR,
        RUN_ID,
        PipelineLogger,
        call_chat,
        call_retrieve,
        _base,
        _avg,
        _rate,
        _pct,
        EVAL_SET,
    )
    _STANDALONE = False
except ImportError:
    # Mode autonome : on recrée le minimum nécessaire
    _STANDALONE = True
    import re
    import requests
    from eval_dataset import EVAL_SET

    BACKEND_URL = "http://localhost:5000"
    LOG_DIR     = Path(__file__).parent / "logs"
    RESULT_DIR  = Path(__file__).parent / "results"
    LOG_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _avg(lst):
        vals = [v for v in lst if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def _rate(bools):
        vals = [v for v in bools if v is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

    def _pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"

    def call_chat(question, doc_id):
        payload = {
            "query":      question,
            "history":    [],
            "doc_filter": {"doc_id": doc_id} if doc_id else None,
        }
        t0 = time.time()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=90
            )
            full = ""
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            full += json.loads(data).get("content", "")
                        except Exception:
                            pass
            return full.strip(), round(time.time() - t0, 3)
        except Exception as e:
            return f"[ERREUR BACKEND] {e}", round(time.time() - t0, 3)

    def call_retrieve(question, doc_id):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/retrieve",
                json={
                    "query":      question,
                    "doc_filter": {"doc_id": doc_id} if doc_id else None,
                    "top_k":      10,
                },
                timeout=30,
            )
            return resp.json() if resp.ok else []
        except Exception:
            return []

    def _base(item):
        return {
            "id":               item["id"],
            "tier":             item["tier"],
            "cat":              item.get("cat"),
            "type":             item["type"],
            "question":         item["question"],
            "expected":         item["expected_answer"],
            "expected_article": item.get("expected_article"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG RAGAS
# ══════════════════════════════════════════════════════════════════════════════

# LLM évaluateur : même modèle que ton app.py
OLLAMA_MODEL    = "minimax-m3:cloud"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embeddings : même modèle E5-small que ton app.py
# HuggingFaceEmbeddings le charge directement depuis HuggingFace/cache local
E5_MODEL_NAME   = "intfloat/multilingual-e5-small"

RAGAS_LOG_DIR    = LOG_DIR
RAGAS_RESULT_DIR = RESULT_DIR

RAGAS_LOGFILE    = RAGAS_LOG_DIR / f"ragas_{RUN_ID}.jsonl"
RAGAS_TXTLOG     = RAGAS_LOG_DIR / f"ragas_{RUN_ID}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    handlers=[
        logging.FileHandler(RAGAS_TXTLOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ragas_eval")


def _jl(record: dict):
    with open(RAGAS_LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# INIT LLM + EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def build_ragas_llm() -> ChatOllama:
    """
    LLM Ollama pour RAGAS (minimax-m3:cloud).
    temperature=0 pour des évaluations déterministes.
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def build_ragas_embeddings() -> HuggingFaceEmbeddings:
    """
    Embeddings E5-small via HuggingFaceEmbeddings.
    Réutilise le même modèle que app.py — déjà en cache local.
    Utilisé par RAGAS pour Answer Relevancy (cosine sim question/réponse).
    Le préfixe "query: " est appliqué automatiquement via encode_kwargs.
    """
    return HuggingFaceEmbeddings(
        model_name=E5_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


# Alias pour compatibilité avec le patch eval.py
build_ollama_llm        = build_ragas_llm
build_ollama_embeddings = build_ragas_embeddings


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE : score_with_ragas
# ══════════════════════════════════════════════════════════════════════════════

def score_with_ragas(
    question:         str,
    response:         str,
    retrieved_chunks: list,
    ground_truth:     str,
    llm=None,
    embeddings=None,
) -> dict:
    """
    Calcule les 4 métriques RAGAS pour une question/réponse donnée.

    Args:
        question         : la question posée
        response         : la réponse générée par le RAG
        retrieved_chunks : liste de dicts avec clé "chunk_text"
        ground_truth     : la réponse attendue (expected_answer dans eval_dataset)
        llm              : instance ChatOllama (réutilisée pour éviter de recréer)
        embeddings       : instance OllamaEmbeddings

    Returns:
        dict avec les 4 scores + métadonnées
    """
    # Extraire le texte brut des chunks
    contexts = [
        c.get("chunk_text", c) if isinstance(c, dict) else str(c)
        for c in retrieved_chunks
    ]

    # Si pas de chunks → scores à 0 directement, pas besoin d'appeler RAGAS
    if not contexts:
        log.warning("  RAGAS → aucun chunk récupéré, scores = 0")
        return {
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_precision": 0.0,
            "context_recall":    0.0,
            "ragas_error":       "no_chunks",
        }

    # Construire le dataset HuggingFace (format attendu par RAGAS)
    dataset = Dataset.from_dict({
        "question":    [question],
        "answer":      [response],
        "contexts":    [contexts],
        "ground_truth": [ground_truth],
    })

    # Métriques à calculer
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,   # on loggue les erreurs sans planter
        )

        scores = {
            "faithfulness":      _safe_score(result, "faithfulness"),
            "answer_relevancy":  _safe_score(result, "answer_relevancy"),
            "context_precision": _safe_score(result, "context_precision"),
            "context_recall":    _safe_score(result, "context_recall"),
            "ragas_error":       None,
        }

    except Exception as e:
        log.error(f"  RAGAS → erreur évaluation : {e}")
        scores = {
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_precision": None,
            "context_recall":    None,
            "ragas_error":       str(e),
        }

    return scores


def _safe_score(result, key: str) -> Optional[float]:
    """Extrait un score RAGAS de manière sécurisée."""
    try:
        val = result[key]
        if val is None:
            return None
        return round(float(val), 4)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER RAGAS COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_ragas(items: list, doc_id: Optional[str] = None) -> dict:
    """
    Lance l'évaluation RAGAS sur toutes les questions.
    Appelle /chat et /retrieve pour chaque question, puis score_with_ragas.
    """
    log.info("=" * 55)
    log.info(f"RAGAS RUN {RUN_ID} | {len(items)} questions")
    log.info(f"  LLM évaluateur : Ollama {OLLAMA_MODEL}")
    log.info(f"  Embeddings     : {E5_MODEL_NAME}")
    log.info("=" * 55)

    # Initialiser LLM et embeddings une seule fois (coûteux à créer)
    log.info("  Chargement modèles RAGAS...")
    try:
        llm        = build_ragas_llm()
        embeddings = build_ragas_embeddings()
        log.info("  Modèles RAGAS prêts ✅")
    except Exception as e:
        log.error(f"  Impossible d'initialiser RAGAS : {e}")
        return {}

    results = []

    for item in items:
        qid = item["id"]
        log.info(f"▶ [{qid}] T{item['tier']} | {item['question'][:70]}")

        t0 = time.time()

        # 1. Retrieval
        retrieved = call_retrieve(item["question"], doc_id)
        log.info(f"  Retrieve → {len(retrieved)} chunks")

        # 2. Génération
        response, latency = call_chat(item["question"], doc_id)
        log.info(f"  LLM     → {latency}s | {len(response)} chars")

        # 3. Calcul RAGAS
        log.info(f"  RAGAS   → calcul en cours...")
        ragas_t0 = time.time()
        scores   = score_with_ragas(
            question=item["question"],
            response=response,
            retrieved_chunks=retrieved,
            ground_truth=item["expected_answer"],
            llm=llm,
            embeddings=embeddings,
        )
        ragas_latency = round(time.time() - ragas_t0, 3)

        # 4. Log
        log.info(
            f"  SCORES  → "
            f"Faith={scores['faithfulness']} "
            f"Relev={scores['answer_relevancy']} "
            f"Prec={scores['context_precision']} "
            f"Recall={scores['context_recall']} "
            f"({ragas_latency}s)"
        )

        record = {
            **_base(item),
            "response":          response,
            "latency_s":         latency,
            "ragas_latency_s":   ragas_latency,
            "retrieved_n":       len(retrieved),
            **scores,
        }
        results.append(record)

        _jl({
            "event":       "ragas_result",
            "run_id":      RUN_ID,
            "question_id": qid,
            **scores,
            "latency_s":         latency,
            "ragas_latency_s":   ragas_latency,
            "ts":                datetime.now().isoformat(),
        })

        log.info(f"■ [{qid}] total={round(time.time()-t0, 2)}s\n{'─'*55}")
        time.sleep(0.3)  # éviter de saturer Ollama

    # Summary
    summary = _build_ragas_summary(results)
    output  = {"run_id": RUN_ID, "summary": summary, "results": results}

    out = RAGAS_RESULT_DIR / f"ragas_{RUN_ID}.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print_ragas_summary(summary)
    log.info(f"💾 Résultats RAGAS → {out}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _build_ragas_summary(results: list) -> dict:
    """Construit le résumé agrégé des scores RAGAS."""

    def _extract(key):
        return [r.get(key) for r in results if r.get(key) is not None]

    # Global
    global_scores = {
        "faithfulness":      _avg(_extract("faithfulness")),
        "answer_relevancy":  _avg(_extract("answer_relevancy")),
        "context_precision": _avg(_extract("context_precision")),
        "context_recall":    _avg(_extract("context_recall")),
        "avg_latency_s":     _avg(_extract("latency_s")),
        "avg_ragas_latency": _avg(_extract("ragas_latency_s")),
        "errors":            sum(1 for r in results if r.get("ragas_error")),
    }

    # Par tier
    by_tier = {}
    for t in [1, 2, 3, 4]:
        sub = [r for r in results if r.get("tier") == t]
        if not sub:
            continue
        by_tier[f"tier_{t}"] = {
            "n":                 len(sub),
            "faithfulness":      _avg([r.get("faithfulness") for r in sub]),
            "answer_relevancy":  _avg([r.get("answer_relevancy") for r in sub]),
            "context_precision": _avg([r.get("context_precision") for r in sub]),
            "context_recall":    _avg([r.get("context_recall") for r in sub]),
        }

    # Par catégorie
    by_cat = {}
    for r in results:
        c = r.get("cat")
        if c is None:
            continue
        key = f"cat_{c}"
        if key not in by_cat:
            by_cat[key] = {"n": 0, "faith": [], "relev": [], "prec": [], "rec": []}
        by_cat[key]["n"]     += 1
        by_cat[key]["faith"].append(r.get("faithfulness"))
        by_cat[key]["relev"].append(r.get("answer_relevancy"))
        by_cat[key]["prec"].append(r.get("context_precision"))
        by_cat[key]["rec"].append(r.get("context_recall"))

    by_cat = {
        k: {
            "n":                 v["n"],
            "faithfulness":      _avg([x for x in v["faith"] if x is not None]),
            "answer_relevancy":  _avg([x for x in v["relev"] if x is not None]),
            "context_precision": _avg([x for x in v["prec"]  if x is not None]),
            "context_recall":    _avg([x for x in v["rec"]   if x is not None]),
        }
        for k, v in by_cat.items()
    }

    # Cas problématiques (faithfulness < 0.5 = hallucination probable)
    low_faith = [
        {
            "id":          r["id"],
            "q":           r["question"][:60],
            "faithfulness": r.get("faithfulness"),
            "resp":        (r.get("response") or "")[:80],
        }
        for r in results
        if (r.get("faithfulness") or 1.0) < 0.5
    ]

    # Cas où context_recall < 0.5 = retrieval insuffisant
    low_recall = [
        {
            "id":             r["id"],
            "q":              r["question"][:60],
            "context_recall": r.get("context_recall"),
            "retrieved_n":    r.get("retrieved_n"),
        }
        for r in results
        if (r.get("context_recall") or 1.0) < 0.5
    ]

    return {
        "run_id":        RUN_ID,
        "total":         len(results),
        **global_scores,
        "by_tier":       by_tier,
        "by_cat":        by_cat,
        "low_faith":     low_faith,
        "low_recall":    low_recall,
    }


def print_ragas_summary(s: dict):
    """Affiche le résumé RAGAS dans le terminal."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RAGAS SUMMARY  run {s['run_id']}")
    print(sep)
    print(f"  Questions           : {s['total']}")
    print(f"  Faithfulness        : {_pct(s['faithfulness'])}       (cible >80%)")
    print(f"  Answer Relevancy    : {_pct(s['answer_relevancy'])}   (cible >75%)")
    print(f"  Context Precision   : {_pct(s['context_precision'])}  (cible >70%)")
    print(f"  Context Recall      : {_pct(s['context_recall'])}     (cible >70%)")
    print(f"  Latence LLM moy.    : {s['avg_latency_s']}s")
    print(f"  Latence RAGAS moy.  : {s['avg_ragas_latency']}s")
    print(f"  Erreurs RAGAS       : {s['errors']}")

    print(f"\n── Par tier {'─'*35}")
    for tk, v in s.get("by_tier", {}).items():
        print(
            f"  {tk}  n={v['n']}"
            f"  Faith={_pct(v['faithfulness'])}"
            f"  Relev={_pct(v['answer_relevancy'])}"
            f"  Prec={_pct(v['context_precision'])}"
            f"  Recall={_pct(v['context_recall'])}"
        )

    print(f"\n── Par catégorie {'─'*30}")
    for ck, v in sorted(s.get("by_cat", {}).items()):
        print(
            f"  {ck}  n={v['n']}"
            f"  Faith={_pct(v['faithfulness'])}"
            f"  Prec={_pct(v['context_precision'])}"
            f"  Recall={_pct(v['context_recall'])}"
        )

    if s.get("low_faith"):
        print(f"\n── ⚠  Faible Faithfulness ({len(s['low_faith'])}) — hallucination probable")
        for f in s["low_faith"]:
            print(f"  [{f['id']}] faith={f['faithfulness']} | {f['q']}")
            print(f"   →  {f['resp']}")

    if s.get("low_recall"):
        print(f"\n── ⚠  Faible Context Recall ({len(s['low_recall'])}) — retrieval insuffisant")
        for r in s["low_recall"]:
            print(
                f"  [{r['id']}] recall={r['context_recall']}"
                f" | chunks={r['retrieved_n']} | {r['q']}"
            )

    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE : relire un résultat RAGAS sauvegardé
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ragas(path: Optional[Path] = None):
    """Relit et affiche un fichier résultat RAGAS JSON."""
    if path is None:
        files = sorted(RAGAS_RESULT_DIR.glob("ragas_*.json"))
        if not files:
            print(f"Aucun résultat RAGAS trouvé dans {RAGAS_RESULT_DIR}")
            return
        path = files[-1]

    print(f"\nAnalyse : {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    print_ragas_summary(data["summary"])

    # Tableau détaillé par question
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {'ID':<6} {'Tier':<5} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7}  Question")
    print(sep)
    for r in data["results"]:
        print(
            f"  {r['id']:<6} T{r['tier']:<4}"
            f" {_pct(r.get('faithfulness')):>7}"
            f" {_pct(r.get('answer_relevancy')):>7}"
            f" {_pct(r.get('context_precision')):>7}"
            f" {_pct(r.get('context_recall')):>7}"
            f"  {r['question'][:45]}"
        )
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation RAGAS — RAG CGI Maroc")
    parser.add_argument("--doc-id",  default=None)
    parser.add_argument("--tier",    type=int, default=None)
    parser.add_argument("--ids",     default=None, help="ex: q01,q04")
    parser.add_argument("--analyze", action="store_true",
                        help="Relit le dernier résultat RAGAS sans relancer")
    parser.add_argument("--result",  default=None,
                        help="Chemin vers un ragas_*.json spécifique pour --analyze")
    args = parser.parse_args()

    if args.analyze:
        path = Path(args.result) if args.result else None
        analyze_ragas(path)
    else:
        items = EVAL_SET[:]
        if args.tier:
            items = [i for i in items if i["tier"] == args.tier]
        if args.ids:
            wanted = set(args.ids.split(","))
            items  = [i for i in items if i["id"] in wanted]
        if not items:
            print("Aucune question sélectionnée.")
            exit(1)
        run_ragas(items, doc_id=args.doc_id)