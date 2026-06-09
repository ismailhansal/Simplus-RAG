"""
eval.py
═══════════════════════════════════════════════════════════════
Logger + Runner + Analyzer fusionnés pour le pipeline RAG CGI.

Usage :
    python eval.py                        # toutes les questions
    python eval.py --tier 1               # seulement tier 1
    python eval.py --ids q01,q04,q05      # questions précises
    python eval.py --retrieve-only        # test retrieval sans LLM
    python eval.py --analyze              # analyse le dernier log sans relancer
    python eval.py --doc-id <uuid>        # filtrer sur un doc précis
═══════════════════════════════════════════════════════════════
"""

import argparse
import json
import math
import re
import time
import logging
import requests
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from eval_dataset import EVAL_SET

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BACKEND_URL = "http://localhost:5000"

LOG_DIR    = Path(__file__).parent / "logs"
RESULT_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGFILE = LOG_DIR / f"pipeline_{RUN_ID}.jsonl"
TXTLOG  = LOG_DIR / f"pipeline_{RUN_ID}.log"

def extract_articles(text: str) -> list:
    return re.findall(
        r"(?:article|art\.?)\s+(\d+)",
        text.lower()
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOGGER
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    handlers=[
        logging.FileHandler(TXTLOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("rag_eval")


def _jl(record: dict):
    """Écrit une ligne JSON dans le fichier JSONL."""
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class PipelineLogger:
    """
    Logge chaque étape du pipeline pour une question donnée.
    Toutes les méthodes écrivent dans le JSONL ET dans le terminal.
    """

    def __init__(self):
        self.q: Optional[dict] = None
        self._t0: float = 0.0
        self._steps: list = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, item: dict):
        self.q      = item
        self._t0    = time.time()
        self._steps = []
        log.info(f"▶ [{item['id']}] T{item['tier']} | {item['question'][:75]}")
        _jl({"event": "question_start", "run_id": RUN_ID,
             "question_id": item["id"], "tier": item["tier"],
             "type": item["type"], "question": item["question"],
             "ts": datetime.now().isoformat()})

    def end(self):
        elapsed = round(time.time() - self._t0, 3)
        log.info(f"■ [{self.q['id']}] done in {elapsed}s\n{'─'*55}")
        _jl({"event": "question_end", "run_id": RUN_ID,
             "question_id": self.q["id"], "elapsed_s": elapsed,
             "steps": self._steps, "ts": datetime.now().isoformat()})

    def error(self, stage: str, err: str):
        log.error(f"  ERROR @ {stage} → {err}")
        _jl({"event": "error", "run_id": RUN_ID,
             "question_id": self.q["id"] if self.q else "?",
             "stage": stage, "error": err, "ts": datetime.now().isoformat()})

    # ── Étapes pipeline ────────────────────────────────────────────────────────

    def hyde(self, original: str, rewritten: str):
        changed = original.strip().lower() != rewritten.strip().lower()
        s = {"step": "hyde", "original_query": original,
             "rewritten_query": rewritten, "changed": changed,
             "expansion_tokens": len(rewritten.split())}
        self._steps.append(s)
        suffix = " [CHANGED]" if changed else " [UNCHANGED]"
        log.info(f"  HyDE   → {rewritten[:90]}{suffix}")
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def bm25(self, chunks: list, scores: Optional[list] = None):
        sections = [c.get("section", "?") for c in chunks[:5]]
        previews = [c["chunk_text"][:80] for c in chunks[:5]]
        s = {"step": "bm25", "retrieved_n": len(chunks),
             "top5_sections": sections, "top5_previews": previews}
        if scores:
            s["top5_scores"] = [round(x, 4) for x in scores[:5]]
        self._steps.append(s)
        log.info(f"  BM25   → {len(chunks)} chunks | top: {sections[0] if sections else '∅'}")
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def vector(self, chunks: list, scores: Optional[list] = None):
        sections = [c.get("section", "?") for c in chunks[:5]]
        previews = [c["chunk_text"][:80] for c in chunks[:5]]
        s = {"step": "vector", "retrieved_n": len(chunks),
             "top5_sections": sections, "top5_previews": previews}
        if scores:
            s["top5_scores"] = [round(x, 4) for x in scores[:5]]
        self._steps.append(s)
        log.info(f"  VECTOR → {len(chunks)} chunks | top: {sections[0] if sections else '∅'}")
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def rrf(self, fused: list):
        sections = [c.get("section", "?") for c in fused[:5]]
        previews = [c["chunk_text"][:80] for c in fused[:5]]
        s = {"step": "rrf_fusion", "fused_n": len(fused),
             "top5_sections": sections, "top5_previews": previews}
        self._steps.append(s)
        log.info(f"  RRF    → {len(fused)} fused | top3: {sections[:3]}")
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def rerank(self, chunks: list, scores: Optional[list] = None):
        sections = [c.get("section", "?") for c in chunks[:5]]
        s = {"step": "cross_encoder_rerank", "reranked_n": len(chunks),
             "top5_sections": sections}
        if scores:
            s["top5_scores"] = [round(float(x), 4) for x in scores[:5]]
        self._steps.append(s)
        log.info(f"  RERANK → top3: {sections[:3]}"
                 + (f" | scores: {s.get('top5_scores', [])[:3]}" if scores else ""))
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def validation(self, chunks: list, confident: bool, reason: str = ""):
        s = {"step": "validation", "is_confident": confident,
             "n_chunks_kept": len(chunks), "reason": reason}
        self._steps.append(s)
        flag = "✅ CONFIDENT" if confident else "⚠  LOW CONF"
        log.info(f"  VALID  → {flag} | {len(chunks)} chunks | {reason}")
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def generation(self, response: str, latency: float):
        s = {"step": "llm_generation", "latency_s": round(latency, 3),
             "response_len": len(response), "response_preview": response[:200]}
        self._steps.append(s)
        log.info(f"  LLM    → {latency:.1f}s | {len(response)} chars")
        log.info(f"  RESP   → {response[:130]}{'…' if len(response)>130 else ''}")
        _jl({"event": "step", "run_id": RUN_ID,
             "question_id": self.q["id"], **s, "ts": datetime.now().isoformat()})

    def metrics(self, m: dict):
        log.info(
            f"  SCORE  → EM={m.get('exact_match')} "
            f"Recall={m.get('token_recall')} "
            f"Art={m.get('article_cited')} "
            f"Hall={m.get('hallucination_flag')} "
            f"KW={m.get('keyword_hit_rate')}"
        )
        _jl({"event": "metrics", "run_id": RUN_ID,
             "question_id": self.q["id"], **m, "ts": datetime.now().isoformat()})

    @staticmethod
    def ingestion(doc_meta: dict, chunks: list, pages: int):
        """À appeler depuis app.py après le chunking."""
        sizes = [len(c["chunk_text"].split()) for c in chunks]
        dist: dict = {}
        for c in chunks:
            sec = c.get("section", "unknown")[:40]
            dist[sec] = dist.get(sec, 0) + 1
        record = {
            "event": "ingestion", "run_id": RUN_ID,
            "doc_id":       doc_meta.get("doc_id"),
            "filename":     doc_meta.get("filename"),
            "doc_type":     doc_meta.get("type"),
            "langue":       doc_meta.get("langue"),
            "pages":        pages,
            "chunks_total": len(chunks),
            "chunk_avg_words": round(sum(sizes)/max(len(sizes),1), 1),
            "chunk_min_words": min(sizes) if sizes else 0,
            "chunk_max_words": max(sizes) if sizes else 0,
            "section_distribution": dist,
            "ts": datetime.now().isoformat(),
        }
        _jl(record)
        log.info(
            f"INGESTION │ {doc_meta.get('filename')} │ "
            f"{pages}p │ {len(chunks)} chunks │ "
            f"avg={record['chunk_avg_words']}w "
            f"min={record['chunk_min_words']}w "
            f"max={record['chunk_max_words']}w"
        )
        return record


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

_REFUSALS = [
    "non présent",
    "non présente",
    "information non présente",
    "information non présente dans les documents",
    "pas dans les documents",
    "information non disponible",
    "ne figure pas",
    "introuvable",
    "absent du document",
    "non mentionné",
    "aucune information",
    "je ne trouve pas",
]


def _is_refusal(response: str) -> bool:
    """Le LLM a-t-il répondu qu'il ne trouve pas l'info dans les documents ?"""
    return any(p in response.lower() for p in _REFUSALS)



def normalize_for_match(text: str) -> str:
    text = text.lower()

    # Supprime "(6)" dans "Six (6) mois"
    text = re.sub(r"\(\d+\)", "", text)

    # Supprime espaces + ponctuation
    text = re.sub(r"[.,;:()\-\s]", "", text)

    return text

def exact_match(response: str, expected: str) -> bool:
    r = response.lower()

    # Tier 4 : on attend un refus propre
    if expected == "non présent":
        return _is_refusal(r)
    if expected == "dépend":
        return any(w in r for w in ["selon", "dépend", "cas", "résident", "personne"])

    # Si le LLM a refusé → EM=False peu importe la valeur attendue
    # (évite EM=True accidentel parce que "non" est dans "information non présente")
    if _is_refusal(r):
        return False

    # "oui" / "non" : chercher le mot entier pour éviter faux positifs
    # ("non" dans "mentionnent", "oui" dans "ouvrir"...)
    if expected in ("oui", "non"):
        return bool(re.search(rf'\b{expected}\b', r))

    # Cas général : inclusion de chaîne (insensible aux espaces)
    return normalize_for_match(expected) in normalize_for_match(response)

def token_recall(response: str, expected: str) -> float:
    if expected in ("non présent", "dépend"):
        return 1.0 if exact_match(response, expected) else 0.0
    exp  = set(re.findall(r'\b\w+\b', expected.lower()))
    resp = set(re.findall(r'\b\w+\b', response.lower()))
    return round(len(exp & resp) / len(exp), 3) if exp else 0.0


def article_cited(response: str, expected: Optional[str]) -> bool:
    if not expected:
        return True
    nums = re.findall(r'\d+', expected)
    if not nums:
        return False
    return bool(re.search(rf'(?:article|art\.?)\s*{nums[0]}\b', response.lower()))


def hallucination(response: str) -> dict:
    r = response.lower()
    cited = [int(n) for n in re.findall(r'(?:article|art\.?)\s+(\d+)', r)]
    invented = [n for n in cited if n > 300]
    doubt = any(m in r for m in ["peut-être", "probablement", "il semble",
                                  "sous réserve", "non présent", "je ne trouve pas"])
    has_src = bool(re.search(r'article\s+\d+', r))
    return {
        "invented_articles":   invented,
        "hallucination_flag":  len(invented) > 0,
        "confident_no_source": not has_src and not doubt and len(response) > 80,
    }


def keyword_hit_rate(chunks: list, keywords: list) -> float:
    if not keywords or not chunks:
        return 0.0
    combined = " ".join(c.get("chunk_text", "") for c in chunks).lower()
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return round(hits / len(keywords), 3)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def call_chat(question: str, doc_id: Optional[str]) -> tuple[str, float]:
    payload = {
        "query":      question,
        "history":    [],
        "doc_filter": {"doc_id": doc_id} if doc_id else None,
    }
    t0 = time.time()
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=90)
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


def call_retrieve(question: str, doc_id: Optional[str]) -> list:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/retrieve",
            json={"query": question,
                  "doc_filter": {"doc_id": doc_id} if doc_id else None,
                  "top_k": 10},
            timeout=30,
        )
        return resp.json() if resp.ok else []
    except Exception:
        return []


def run(items: list, doc_id: Optional[str], retrieve_only: bool) -> dict:
    pl      = PipelineLogger()
    results = []

    log.info("=" * 55)
    log.info(f"RUN {RUN_ID} | {len(items)} questions | doc_id={doc_id or 'ALL'}")
    log.info("=" * 55)

    for item in items:
        pl.start(item)

        # Retrieval (appel /retrieve si dispo)
        retrieved = call_retrieve(item["question"], doc_id)
        kw_hit    = keyword_hit_rate(retrieved, item.get("keywords_hint", []))

        if retrieve_only:
            pl.bm25(retrieved)
            m = {"keyword_hit_rate": kw_hit, "retrieved_n": len(retrieved)}
            pl.metrics(m)
            results.append({
                **_base(item),
                "response": response,
                "retrieved_chunks": retrieved,
                **m
            })            
            pl.end()
            time.sleep(0.2)
            continue

        # Génération
        response, latency = call_chat(item["question"], doc_id)
        pl.generation(response, latency)

        # Scoring
        em   = exact_match(response, item["expected_answer"])
        rec  = token_recall(response, item["expected_answer"])
        art  = article_cited(response, item.get("expected_article"))
        hall = hallucination(response)
        articles_detected = extract_articles(response)

        # Tier 4 : refus propre = succès
        if item["tier"] == 4:
            em  = any(p in response.lower() for p in _REFUSALS)
            rec = 1.0 if em else 0.0

        m = {
            "exact_match":         em,
            "token_recall":        rec,
            "article_cited":       art,
            "hallucination_flag":  hall["hallucination_flag"],
            "confident_no_source": hall["confident_no_source"],
            "invented_articles":   hall["invented_articles"],
            "keyword_hit_rate":    kw_hit,
            "latency_s":           latency,
            "articles_detected": articles_detected,
        }
        pl.metrics(m)
        results.append({**_base(item), "response": response, **m})
        pl.end()
        time.sleep(0.5)

    summary = _build_summary(results)
    output  = {"run_id": RUN_ID, "summary": summary, "results": results}

    out = RESULT_DIR / f"eval_{RUN_ID}.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(summary)
    log.info(f"💾 Résultats → {out}")
    return output


def _base(item: dict) -> dict:
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
# SECTION 4 — SUMMARY + AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

def _avg(lst):
    vals = [v for v in lst if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None

def _rate(bools):
    vals = [v for v in bools if v is not None]
    return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

def _pct(v):
    return f"{v*100:.1f}%" if v is not None else "—"


def _build_summary(results: list) -> dict:
    gen = [r for r in results if r.get("response") is not None]

    by_tier = {}
    for t in [1, 2, 3, 4]:
        sub = [r for r in gen if r.get("tier") == t]
        if not sub:
            continue
        by_tier[f"tier_{t}"] = {
            "n":             len(sub),
            "exact_match":   _rate([r.get("exact_match") for r in sub]),
            "token_recall":  _avg([r.get("token_recall") for r in sub]),
            "article_cited": _rate([r.get("article_cited") for r in sub]),
            "hallucination": _rate([r.get("hallucination_flag") for r in sub]),
            "avg_latency_s": _avg([r.get("latency_s") for r in sub]),
        }

    by_type = {}
    for r in gen:
        t = r.get("type", "?")
        if t not in by_type:
            by_type[t] = {"n": 0, "em": [], "rec": []}
        by_type[t]["n"]   += 1
        by_type[t]["em"].append(r.get("exact_match", False))
        by_type[t]["rec"].append(r.get("token_recall", 0.0))
    by_type = {
        t: {"n": v["n"],
            "exact_match":  _rate(v["em"]),
            "token_recall": _avg(v["rec"])}
        for t, v in by_type.items()
    }

    _CAT_LABELS = {
        1: "Keyword/BM25 pur",
        2: "Sémantique/Embedding",
        3: "Numérique/Tableaux",
        4: "Négatif/Exclusion",
        5: "Multi-conditions",
        6: "Temporel/Transitoire",
        7: "Piège/Ambiguïté",
        8: "Procédure/Sanction",
    }
    by_cat = {}
    for r in gen:
        c = r.get("cat")
        if c is None:
            continue
        key = f"cat_{c}"
        if key not in by_cat:
            by_cat[key] = {"n": 0, "em": [], "rec": [], "label": _CAT_LABELS.get(c, str(c))}
        by_cat[key]["n"]   += 1
        by_cat[key]["em"].append(r.get("exact_match", False))
        by_cat[key]["rec"].append(r.get("token_recall", 0.0))
    by_cat = {
        k: {"n": v["n"], "label": v["label"],
            "exact_match":  _rate(v["em"]),
            "token_recall": _avg(v["rec"])}
        for k, v in by_cat.items()
    }

    failures = [
        {"id": r["id"], "q": r["question"][:60],
         "expected": r.get("expected",""),
         "resp": (r.get("response") or "")[:80]}
        for r in gen if not r.get("exact_match")
    ]

    return {
        "run_id":               RUN_ID,
        "total":                len(results),
        "generated":            len(gen),
        "exact_match":          _rate([r.get("exact_match") for r in gen]),
        "token_recall":         _avg([r.get("token_recall") for r in gen]),
        "article_cited":        _rate([r.get("article_cited") for r in gen]),
        "hallucination_rate":   _rate([r.get("hallucination_flag") for r in gen]),
        "confident_no_source":  _rate([r.get("confident_no_source") for r in gen]),
        "keyword_hit_rate":     _avg([r.get("keyword_hit_rate") for r in gen]),
        "avg_latency_s":        _avg([r.get("latency_s") for r in gen]),
        "by_tier":              by_tier,
        "by_cat":               by_cat,
        "by_type":              by_type,
        "failures":             failures,
    }


def _print_summary(s: dict):
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RÉSUMÉ  run {s['run_id']}")
    print(sep)
    print(f"  Questions       : {s['total']} (générées: {s['generated']})")
    print(f"  Exact Match     : {_pct(s['exact_match'])}")
    print(f"  Token Recall    : {_pct(s['token_recall'])}")
    print(f"  Article cité    : {_pct(s['article_cited'])}")
    print(f"  Hallucination   : {_pct(s['hallucination_rate'])}  (cible <10%)")
    print(f"  Confiant/source : {_pct(s['confident_no_source'])}")
    print(f"  Keyword hit     : {_pct(s['keyword_hit_rate'])}")
    print(f"  Latence moy.    : {s['avg_latency_s']}s")

    objs = {"tier_1": .90, "tier_2": .70, "tier_3": .50, "tier_4": 1.0}
    print(f"\n── Par tier {'─'*35}")
    for tk, v in s.get("by_tier", {}).items():
        obj  = objs.get(tk, 0)
        flag = "✅" if (v["exact_match"] or 0) >= obj else "❌"
        print(f"  {flag} {tk} (obj {_pct(obj)})  n={v['n']}"
              f"  EM={_pct(v['exact_match'])}"
              f"  Recall={_pct(v['token_recall'])}"
              f"  Art={_pct(v['article_cited'])}"
              f"  Hall={_pct(v['hallucination'])}"
              f"  {v['avg_latency_s']}s")

    print(f"\n── Par catégorie diagnostique {'─'*20}")
    for ck, v in sorted(s.get("by_cat", {}).items()):
        flag = "✅" if (v["exact_match"] or 0) >= 0.6 else "❌"
        print(f"  {flag} {ck} {v['label']:<26} n={v['n']}"
              f"  EM={_pct(v['exact_match'])}"
              f"  Recall={_pct(v['token_recall'])}")

    print(f"\n── Par type {'─'*35}")
    for t, v in sorted(s.get("by_type", {}).items(), key=lambda x: -(x[1]["exact_match"] or 0)):
        print(f"  {t:<32} n={v['n']}  EM={_pct(v['exact_match'])}  Recall={_pct(v['token_recall'])}")

    if s.get("failures"):
        print(f"\n── Échecs ({len(s['failures'])}) {'─'*30}")
        for f in s["failures"]:
            print(f"  [{f['id']}] attendu: {f['expected']}")
            print(f"   q: {f['q']}")
            print(f"   →  {f['resp']}")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ANALYZER (logs JSONL)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_logs(path: Optional[Path] = None):
    """Analyse un fichier JSONL de pipeline et affiche les stats par étape."""
    if path is None:
        files = sorted(LOG_DIR.glob("pipeline_*.jsonl"))
        if not files:
            print(f"Aucun log trouvé dans {LOG_DIR}")
            return
        path = files[-1]

    print(f"\n{'='*55}")
    print(f"  LOG ANALYSIS  │  {path.name}")
    print(f"{'='*55}\n")

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    by_step   = defaultdict(list)
    metrics   = []
    ingestions = []
    errors    = []

    for r in records:
        ev = r.get("event")
        if ev == "step":
            by_step[r.get("step", "?")].append(r)
        elif ev == "metrics":
            metrics.append(r)
        elif ev == "ingestion":
            ingestions.append(r)
        elif ev == "error":
            errors.append(r)

    SEP = "─" * 45

    # Ingestion
    if ingestions:
        print("  📄 INGESTION / CHUNKING")
        print(f"  {SEP}")
        for ing in ingestions:
            print(f"  Fichier        : {ing.get('filename')}")
            print(f"  Type détecté   : {ing.get('doc_type')}")
            print(f"  Pages          : {ing.get('pages')}")
            print(f"  Chunks total   : {ing.get('chunks_total')}")
            print(f"  Mots moy/chunk : {ing.get('chunk_avg_words')}")
            print(f"  Min / Max      : {ing.get('chunk_min_words')} / {ing.get('chunk_max_words')} mots")
            dist = ing.get("section_distribution", {})
            if dist:
                print(f"  Top sections   :")
                for sec, cnt in list(dist.items())[:6]:
                    print(f"    • {sec[:48]:<50} {cnt}")
        print()

    # HyDE
    hyde = by_step.get("hyde", [])
    if hyde:
        changed = [s for s in hyde if s.get("changed")]
        tokens  = [s.get("expansion_tokens", 0) for s in hyde]
        print("  🔄 HyDE")
        print(f"  {SEP}")
        print(f"  Appels         : {len(hyde)}")
        print(f"  Queries modif. : {len(changed)} ({_pct(len(changed)/len(hyde))})")
        print(f"  Tokens moy.    : {_avg(tokens)}")
        for s in changed[:2]:
            print(f"  ex orig  → {s.get('original_query','')[:60]}")
            print(f"     rewr  → {s.get('rewritten_query','')[:70]}")
        print()

    # BM25
    bm25 = by_step.get("bm25", [])
    if bm25:
        ns = [s.get("retrieved_n", 0) for s in bm25]
        print("  🔍 BM25")
        print(f"  {SEP}")
        print(f"  Appels         : {len(bm25)}")
        print(f"  Rés. moyen     : {_avg(ns)}")
        print(f"  Min / Max      : {min(ns)} / {max(ns)}")
        print(f"  Zéro résultats : {sum(1 for n in ns if n==0)}")
        print()

    # Vector
    vec = by_step.get("vector", [])
    if vec:
        ns = [s.get("retrieved_n", 0) for s in vec]
        print("  🧠 VECTOR")
        print(f"  {SEP}")
        print(f"  Appels         : {len(vec)}")
        print(f"  Rés. moyen     : {_avg(ns)}")
        print(f"  Zéro résultats : {sum(1 for n in ns if n==0)}")
        print()

    # RRF
    rrf = by_step.get("rrf_fusion", [])
    if rrf:
        ns = [s.get("fused_n", 0) for s in rrf]
        print("  ⚡ RRF FUSION")
        print(f"  {SEP}")
        print(f"  Appels         : {len(rrf)}")
        print(f"  Fused moyen    : {_avg(ns)}")
        print(f"  Zéro résultats : {sum(1 for n in ns if n==0)}")
        print()

    # Rerank
    rr = by_step.get("cross_encoder_rerank", [])
    if rr:
        ns     = [s.get("reranked_n", 0) for s in rr]
        scores = [sc for s in rr for sc in s.get("top5_scores", [])]
        print("  🎯 CROSS-ENCODER RERANK")
        print(f"  {SEP}")
        print(f"  Appels         : {len(rr)}")
        print(f"  Reranked moy.  : {_avg(ns)}")
        print(f"  Score top moy. : {_avg(scores)}")
        print(f"  Score top min. : {min(scores) if scores else '—'}")
        print()

    # Validation
    val = by_step.get("validation", [])
    if val:
        conf = [s for s in val if s.get("is_confident")]
        print("  ✅ VALIDATION")
        print(f"  {SEP}")
        print(f"  Appels         : {len(val)}")
        print(f"  Confiant       : {len(conf)} ({_pct(len(conf)/len(val))})")
        lc = [s.get("reason","") for s in val if not s.get("is_confident") and s.get("reason")]
        for r in lc[:3]:
            print(f"  low conf → {r}")
        print()

    # LLM
    gen = by_step.get("llm_generation", [])
    if gen:
        lats = [s.get("latency_s", 0) for s in gen]
        lens = [s.get("response_len", 0) for s in gen]
        p90  = sorted(lats)[int(len(lats)*0.9)] if lats else None
        print("  💬 LLM GÉNÉRATION")
        print(f"  {SEP}")
        print(f"  Appels         : {len(gen)}")
        print(f"  Latence moy.   : {_avg(lats)}s")
        print(f"  Latence p90    : {p90}s")
        print(f"  Latence max    : {max(lats) if lats else '—'}s")
        print(f"  Longueur moy.  : {_avg(lens)} chars")
        print(f"  Appels >10s    : {sum(1 for l in lats if l>10)}")
        print()

    # Métriques agrégées
    if metrics:
        print("  📊 MÉTRIQUES AGRÉGÉES")
        print(f"  {SEP}")
        em   = [m.get("exact_match")      for m in metrics if "exact_match" in m]
        rec  = [m.get("token_recall")     for m in metrics if "token_recall" in m]
        art  = [m.get("article_cited")    for m in metrics if "article_cited" in m]
        hall = [m.get("hallucination_flag") for m in metrics if "hallucination_flag" in m]
        kw   = [m.get("keyword_hit_rate") for m in metrics if "keyword_hit_rate" in m]
        print(f"  Questions      : {len(metrics)}")
        print(f"  Exact Match    : {_pct(_rate([bool(v) for v in em]))}")
        print(f"  Token Recall   : {_pct(_avg(rec))}")
        print(f"  Article cité   : {_pct(_rate([bool(v) for v in art]))}")
        print(f"  Hallucination  : {_pct(_rate([bool(v) for v in hall]))}")
        print(f"  Keyword hit    : {_pct(_avg(kw))}")
        print()

    # Erreurs
    if errors:
        print(f"  🚨 ERREURS ({len(errors)})")
        print(f"  {SEP}")
        for e in errors:
            print(f"  [{e.get('question_id','?')}] {e.get('stage','?')} → {e.get('error','?')}")
        print()

    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation RAG CGI Maroc")
    parser.add_argument("--doc-id",        default=None)
    parser.add_argument("--tier",          type=int, default=None)
    parser.add_argument("--ids",           default=None, help="ex: q01,q04")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--analyze",       action="store_true",
                        help="Analyse le dernier log JSONL sans relancer l'éval")
    parser.add_argument("--log",           default=None,
                        help="Chemin vers un log JSONL spécifique pour --analyze")
    args = parser.parse_args()

    if args.analyze:
        path = Path(args.log) if args.log else None
        analyze_logs(path)
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
        run(items, doc_id=args.doc_id, retrieve_only=args.retrieve_only)