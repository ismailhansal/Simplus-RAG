"""
report_generator.py
Génère un rapport HTML visuel depuis un fichier eval_*.json.
Si un fichier ragas_*.json du même run existe, les métriques RAGAS
sont automatiquement intégrées au rapport.

Usage :
    python report_generator.py                         # dernier run
    python report_generator.py results/eval_XYZ.json  # run spécifique
    python report_generator.py --compare a.json b.json # comparaison
"""

import json
import math
import argparse
from pathlib import Path
from datetime import datetime

RESULT_DIR = Path(__file__).parent / "results"
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement
# ══════════════════════════════════════════════════════════════════════════════

def load_latest() -> dict:
    files = sorted(RESULT_DIR.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError(f"Aucun résultat dans {RESULT_DIR}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ragas(run_id: str) -> dict | None:
    """
    Charge le fichier ragas_<run_id>.json correspondant au même run.
    Retourne None si absent (RAGAS n'a pas tourné sur ce run).
    """
    path = RESULT_DIR / f"ragas_{run_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pct(v, decimals=1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%"


def badge(value: bool | None, true_txt="✓", false_txt="✗") -> str:
    if value is None:
        return '<span class="badge neutral">—</span>'
    if value:
        return f'<span class="badge ok">{true_txt}</span>'
    return f'<span class="badge fail">{false_txt}</span>'


def tier_label(t: int) -> str:
    return {
        1: "Factuel",
        2: "Négation/Cond.",
        3: "Multi/Barème",
        4: "Hors périmètre",
    }.get(t, str(t))


def bar(value: float | None, target: float = 0.7) -> str:
    if value is None:
        return '<div class="bar-track"><div class="bar-fill" style="width:0%;background:#333"></div></div>'
    pv    = value * 100
    color = "#4ade80" if value >= target else ("#facc15" if value >= target * 0.7 else "#f87171")
    return (f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pv:.1f}%;background:{color}"></div>'
            f'</div>')


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

HTML_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root {
  --bg:#0e0e0e; --surface:#141414; --border:#242424;
  --accent:#c8ff00; --text:#e0e0e0; --muted:#555;
  --ok:#4ade80; --warn:#facc15; --fail:#f87171;
  --ragas:#818cf8;
  --font-mono:'IBM Plex Mono',monospace; --font-sans:'IBM Plex Sans',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:14px;line-height:1.6}
a{color:var(--accent);text-decoration:none}
h1{font-family:var(--font-mono);font-size:1.1rem;letter-spacing:.15em;color:var(--accent);text-transform:uppercase;padding:2rem 2.5rem .5rem}
h2{font-family:var(--font-mono);font-size:.8rem;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin:2rem 2.5rem .8rem}
.meta{font-family:var(--font-mono);font-size:.72rem;color:var(--muted);padding:0 2.5rem 1.5rem}
.cards{display:flex;flex-wrap:wrap;gap:1rem;padding:0 2.5rem 1rem}
.card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1.2rem 1.4rem;min-width:160px;flex:1}
.card-label{font-family:var(--font-mono);font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem}
.card-val{font-family:var(--font-mono);font-size:1.5rem;font-weight:600;color:var(--accent)}
.card-sub{font-size:.72rem;color:var(--muted);margin-top:.2rem}
.card.warn .card-val{color:var(--warn)}
.card.danger .card-val{color:var(--fail)}
.card.ok .card-val{color:var(--ok)}
.card.ragas .card-val{color:var(--ragas)}
.card.ragas{border-color:#2e2d5a}
.ragas-section-title{
  font-family:var(--font-mono);font-size:.65rem;letter-spacing:.1em;
  color:var(--ragas);text-transform:uppercase;
  padding:.3rem .7rem;background:#1a1a2e;border-radius:3px;
  display:inline-block;margin-bottom:.5rem
}
.tier-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem;padding:0 2.5rem 1rem}
.tier-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1rem 1.2rem}
.tier-head{font-family:var(--font-mono);font-size:.75rem;letter-spacing:.08em;margin-bottom:.8rem;display:flex;justify-content:space-between;align-items:center}
.tier-name{color:var(--text)}
.tier-obj{color:var(--muted);font-size:.65rem}
.bar-row{display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-size:.72rem}
.bar-label{color:var(--muted);width:110px;flex-shrink:0}
.bar-track{flex:1;background:#1e1e1e;border-radius:99px;height:6px;overflow:hidden}
.bar-fill{height:100%;border-radius:99px}
.bar-val{color:var(--text);width:40px;text-align:right;font-family:var(--font-mono);font-size:.65rem}
.bar-ragas .bar-label{color:#5b5b8a}
.divider-ragas{border:none;border-top:1px dashed #2e2d5a;margin:.5rem 0}
table{width:calc(100% - 5rem);margin:0 2.5rem 2rem;border-collapse:collapse;font-size:.8rem}
th{font-family:var(--font-mono);font-size:.65rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);padding:.6rem .8rem;text-align:left}
td{padding:.55rem .8rem;border-bottom:1px solid #1a1a1a;vertical-align:top}
tr:hover td{background:#111}
.badge{display:inline-block;border-radius:3px;padding:1px 6px;font-family:var(--font-mono);font-size:.65rem}
.badge.ok{background:#0f2e1b;color:var(--ok)}
.badge.fail{background:#2e0f0f;color:var(--fail)}
.badge.warn{background:#2e2507;color:var(--warn)}
.badge.neutral{background:#1e1e1e;color:var(--muted)}
.tier-badge{font-size:.62rem;font-family:var(--font-mono);padding:1px 5px;border-radius:2px}
.t1{background:#0d2d0d;color:#4ade80}.t2{background:#1a200d;color:#a3e635}
.t3{background:#2d200d;color:#fbbf24}.t4{background:#2d0d2d;color:#c084fc}
.response-cell{max-width:280px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:var(--muted);font-size:.72rem}
.section-sep{border:none;border-top:1px solid var(--border);margin:1.5rem 2.5rem}
.hall-flag{color:var(--fail);font-family:var(--font-mono);font-size:.65rem}
.alert-box{margin:0 2.5rem 1.2rem;background:#1a1218;border:1px solid #3a1a2a;border-radius:6px;padding:1rem 1.2rem}
.alert-title{font-family:var(--font-mono);font-size:.65rem;color:var(--fail);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem}
.alert-item{font-size:.75rem;color:var(--muted);margin:.3rem 0;display:flex;gap:.8rem}
.alert-id{font-family:var(--font-mono);color:var(--fail);width:50px;flex-shrink:0}
.alert-score{font-family:var(--font-mono);color:var(--warn);width:60px;flex-shrink:0}
.footer{font-family:var(--font-mono);font-size:.65rem;color:var(--muted);padding:1rem 2.5rem 2rem;border-top:1px solid var(--border);margin-top:2rem}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Composants HTML
# ══════════════════════════════════════════════════════════════════════════════

def kpi_cards_html(s: dict) -> str:
    em_cls   = "ok"     if (s.get("exact_match") or 0) >= 0.70 else ("warn" if (s.get("exact_match") or 0) >= 0.50 else "danger")
    hall_cls = "ok"     if (s.get("hallucination_rate") or 0) <= 0.10 else "danger"
    return f"""
    <div class="cards">
      <div class="card {em_cls}">
        <div class="card-label">Exact Match</div>
        <div class="card-val">{pct(s.get('exact_match'))}</div>
        <div class="card-sub">cible globale ≥ 70%</div>
      </div>
      <div class="card">
        <div class="card-label">Token Recall</div>
        <div class="card-val">{pct(s.get('token_recall'))}</div>
      </div>
      <div class="card">
        <div class="card-label">Article cité</div>
        <div class="card-val">{pct(s.get('article_cited'))}</div>
        <div class="card-sub">cible ≥ 60%</div>
      </div>
      <div class="card {hall_cls}">
        <div class="card-label">Hallucination</div>
        <div class="card-val">{pct(s.get('hallucination_rate'))}</div>
        <div class="card-sub">cible &lt; 10%</div>
      </div>
      <div class="card">
        <div class="card-label">Keyword Hit</div>
        <div class="card-val">{pct(s.get('keyword_hit_rate'))}</div>
        <div class="card-sub">retrieval quality</div>
      </div>
      <div class="card">
        <div class="card-label">Latence moy.</div>
        <div class="card-val">{s.get('avg_latency_s','?')}s</div>
      </div>
      <div class="card">
        <div class="card-label">Questions</div>
        <div class="card-val">{s.get('total','?')}</div>
        <div class="card-sub">générées: {s.get('generated','?')}</div>
      </div>
    </div>"""


def ragas_cards_html(rs: dict) -> str:
    """4 KPI cards RAGAS en violet."""
    def rcard(label, val, target, hint):
        cls = "ok" if (val or 0) >= target else ("warn" if (val or 0) >= target * 0.7 else "danger")
        return f"""
        <div class="card ragas {cls}">
          <div class="card-label">{label}</div>
          <div class="card-val">{pct(val)}</div>
          <div class="card-sub">{hint}</div>
        </div>"""

    err = rs.get("errors", 0)
    err_html = f'<div class="card ragas"><div class="card-label">Erreurs RAGAS</div><div class="card-val" style="font-size:1rem;color:var(--fail)">{err}</div></div>' if err else ""

    return f"""
    <div style="padding:0 2.5rem .5rem">
      <span class="ragas-section-title">⬡ RAGAS — évaluation sémantique</span>
    </div>
    <div class="cards">
      {rcard("Faithfulness",      rs.get("faithfulness"),      0.80, "réponse fidèle aux chunks · cible &gt; 80%")}
      {rcard("Answer Relevancy",  rs.get("answer_relevancy"),  0.75, "répond à la question · cible &gt; 75%")}
      {rcard("Context Precision", rs.get("context_precision"), 0.70, "chunks utiles · cible &gt; 70%")}
      {rcard("Context Recall",    rs.get("context_recall"),    0.70, "chunks suffisants · cible &gt; 70%")}
      {err_html}
    </div>"""


def ragas_alerts_html(rs: dict) -> str:
    """Blocs d'alertes low_faith et low_recall."""
    out = ""

    low_faith = rs.get("low_faith", [])
    if low_faith:
        items = "".join(
            f'<div class="alert-item">'
            f'<span class="alert-id">[{f["id"]}]</span>'
            f'<span class="alert-score">faith={f["faithfulness"]}</span>'
            f'<span>{f["q"]}</span>'
            f'</div>'
            for f in low_faith
        )
        out += f"""
        <div class="alert-box">
          <div class="alert-title">⚠ Faible Faithfulness ({len(low_faith)}) — hallucination probable</div>
          {items}
        </div>"""

    low_recall = rs.get("low_recall", [])
    if low_recall:
        items = "".join(
            f'<div class="alert-item">'
            f'<span class="alert-id">[{r["id"]}]</span>'
            f'<span class="alert-score">recall={r["context_recall"]}</span>'
            f'<span>{r["q"]} &nbsp;<span style="color:#444">({r["retrieved_n"]} chunks)</span></span>'
            f'</div>'
            for r in low_recall
        )
        out += f"""
        <div class="alert-box" style="border-color:#1a2d3a;background:#111a20">
          <div class="alert-title" style="color:var(--warn)">⚠ Faible Context Recall ({len(low_recall)}) — retrieval insuffisant</div>
          {items}
        </div>"""

    return out


def tier_card_html(tier_key: str, v: dict, ragas_tier: dict | None = None) -> str:
    targets = {"tier_1": 0.90, "tier_2": 0.70, "tier_3": 0.50, "tier_4": 1.00}
    t_num   = int(tier_key.split("_")[1])
    obj     = targets.get(tier_key, 0.7)
    label   = tier_label(t_num)

    # Métriques eval.py standard
    rows = ""
    for metric, label_m, tgt in [
        ("exact_match",   "Exact Match",   obj),
        ("token_recall",  "Token Recall",  obj),
        ("article_cited", "Article cité",  0.60),
        ("hallucination", "Hallucination", 0.00),
    ]:
        val         = v.get(metric)
        display_val = (1.0 - val) if (metric == "hallucination" and val is not None) else val
        rows += (f'<div class="bar-row">'
                 f'<span class="bar-label">{label_m}</span>'
                 f'{bar(display_val, tgt if metric != "hallucination" else 0.9)}'
                 f'<span class="bar-val">{pct(val, 0)}</span>'
                 f'</div>')

    # Métriques RAGAS si disponibles pour ce tier
    ragas_rows = ""
    if ragas_tier:
        ragas_rows = '<hr class="divider-ragas">'
        for metric, label_m, tgt in [
            ("faithfulness",      "Faithfulness",   0.80),
            ("answer_relevancy",  "Ans. Relevancy", 0.75),
            ("context_precision", "Ctx Precision",  0.70),
            ("context_recall",    "Ctx Recall",     0.70),
        ]:
            val = ragas_tier.get(metric)
            ragas_rows += (
                f'<div class="bar-row bar-ragas">'
                f'<span class="bar-label">{label_m}</span>'
                f'{bar(val, tgt)}'
                f'<span class="bar-val" style="color:#818cf8">{pct(val, 0)}</span>'
                f'</div>'
            )

    return f"""
    <div class="tier-card">
      <div class="tier-head">
        <span class="tier-name">Tier {t_num} — {label}</span>
        <span class="tier-obj">obj {pct(obj, 0)} · n={v['n']}</span>
      </div>
      {rows}
      {ragas_rows}
      <div style="margin-top:.6rem;font-size:.65rem;font-family:var(--font-mono);color:var(--muted)">
        latence moy. {v.get('avg_latency_s', '?')}s
      </div>
    </div>"""


def type_table_html(by_type: dict) -> str:
    rows = ""
    for t, v in sorted(by_type.items(), key=lambda x: -(x[1].get("exact_match") or 0)):
        rows += (f'<tr>'
                 f'<td style="font-family:var(--font-mono);font-size:.72rem">{t}</td>'
                 f'<td>{v["n"]}</td>'
                 f'<td>{bar(v.get("exact_match"), 0.7)}</td>'
                 f'<td style="font-family:var(--font-mono);font-size:.7rem">{pct(v.get("exact_match"))}</td>'
                 f'<td style="font-family:var(--font-mono);font-size:.7rem">{pct(v.get("token_recall"))}</td>'
                 f'</tr>')
    return f"""
    <table>
      <thead><tr><th>Type</th><th>n</th><th>EM (barre)</th><th>EM %</th><th>Recall</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def results_table_html(results: list, ragas_by_id: dict) -> str:
    """
    Table détaillée par question.
    ragas_by_id : dict {question_id: ragas_result} pour afficher
                  Faithfulness et Context Recall par ligne.
    """
    rows = ""
    has_ragas = bool(ragas_by_id)

    for r in results:
        t  = r.get("tier", 0)
        tc = f"t{t}"

        hall_info = ""
        if r.get("invented_articles"):
            hall_info = f'<div class="hall-flag">⚠ art. inventés: {r["invented_articles"]}</div>'

        full_response = (r.get("response") or "N/A").replace("<", "&lt;")

        # Colonnes RAGAS par ligne
        ragas_cols = ""
        if has_ragas:
            rr = ragas_by_id.get(r["id"], {})
            faith = rr.get("faithfulness")
            recall = rr.get("context_recall")
            faith_cls  = "ok" if (faith  or 0) >= 0.8 else ("warn" if (faith  or 0) >= 0.5 else "fail")
            recall_cls = "ok" if (recall or 0) >= 0.7 else ("warn" if (recall or 0) >= 0.5 else "fail")
            ragas_cols = (
                f'<td><span class="badge {faith_cls}">{pct(faith, 0)}</span></td>'
                f'<td><span class="badge {recall_cls}">{pct(recall, 0)}</span></td>'
            )

        rows += f"""
        <tr>
          <td><span class="tier-badge {tc}">T{t}</span></td>
          <td style="font-family:var(--font-mono);font-size:.65rem;color:var(--accent)">{r['id']}</td>
          <td style="max-width:200px">{r['question'][:85]}{'…' if len(r['question'])>85 else ''}</td>
          <td style="font-family:var(--font-mono);font-size:.7rem">{r.get('expected','')}</td>
          <td>{badge(r.get('exact_match'))}</td>
          <td style="font-family:var(--font-mono);font-size:.65rem">{pct(r.get('token_recall'))}</td>
          <td>{badge(r.get('article_cited'), '📎 oui', '✗ non')}</td>
          <td>{badge(not r.get('hallucination_flag'), '✓ propre', '⚠ suspect')}{hall_info}</td>
          {ragas_cols}
          <td style="font-family:var(--font-mono);font-size:.65rem">{r.get('latency_s','?')}s</td>
          <td class="response-cell" title="{full_response}">{full_response}</td>
        </tr>"""

    # En-têtes avec colonnes RAGAS conditionnelles
    ragas_headers = ""
    if has_ragas:
        ragas_headers = '<th style="color:#818cf8">Faith.</th><th style="color:#818cf8">Ctx Rec.</th>'

    return f"""
    <table>
      <thead><tr>
        <th>Tier</th><th>ID</th><th>Question</th><th>Attendu</th>
        <th>EM</th><th>Recall</th><th>Article</th><th>Hallucin.</th>
        {ragas_headers}
        <th>Latence</th><th>Réponse</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ══════════════════════════════════════════════════════════════════════════════
# Génération HTML principale
# ══════════════════════════════════════════════════════════════════════════════

def generate_html(data: dict, compare: dict | None = None) -> str:
    s       = data["summary"]
    results = data.get("results", [])
    run_id  = s.get("run_id", "?")
    ts      = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Charger RAGAS si disponible
    ragas_data  = load_ragas(run_id)
    ragas_s     = ragas_data["summary"] if ragas_data else None
    # Index RAGAS par question_id pour la table détaillée
    ragas_by_id = {}
    if ragas_data:
        for r in ragas_data.get("results", []):
            ragas_by_id[r["id"]] = r

    # ── KPI cards standard ────────────────────────────────────────────────────
    kpis = kpi_cards_html(s)

    # ── KPI cards RAGAS (si dispo) ────────────────────────────────────────────
    ragas_kpis = ragas_cards_html(ragas_s) if ragas_s else ""

    # ── Alertes RAGAS (si dispo) ──────────────────────────────────────────────
    ragas_alerts = ragas_alerts_html(ragas_s) if ragas_s else ""

    # ── Tier cards enrichies ──────────────────────────────────────────────────
    tier_cards = ""
    for tk, tv in s.get("by_tier", {}).items():
        ragas_tier = (ragas_s or {}).get("by_tier", {}).get(tk) if ragas_s else None
        tier_cards += tier_card_html(tk, tv, ragas_tier)

    # ── Type table ────────────────────────────────────────────────────────────
    type_table = type_table_html(s.get("by_type", {}))

    # ── Table détaillée ───────────────────────────────────────────────────────
    detail_table = results_table_html(results, ragas_by_id)

    # ── Badge RAGAS dans le header ────────────────────────────────────────────
    ragas_badge = (
        '<span style="font-family:var(--font-mono);font-size:.65rem;'
        'color:var(--ragas);background:#1a1a2e;padding:2px 8px;'
        'border-radius:3px;margin-left:1rem">+ RAGAS</span>'
        if ragas_s else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RAG Eval Report — {run_id}</title>
  {HTML_STYLE}
</head>
<body>
  <h1>⬡ RAG Audit — Rapport d'évaluation {ragas_badge}</h1>
  <div class="meta">Run ID: {run_id} &nbsp;·&nbsp; Généré le {ts}</div>

  <h2>KPIs globaux</h2>
  {kpis}

  {'<hr class="section-sep">' + ragas_kpis if ragas_kpis else ''}

  {ragas_alerts}

  <hr class="section-sep">
  <h2>Résultats par tier</h2>
  <div class="tier-grid">{tier_cards}</div>

  <hr class="section-sep">
  <h2>Résultats par type de question</h2>
  {type_table}

  <hr class="section-sep">
  <h2>Détail des questions</h2>
  {detail_table}

  <div class="footer">
    RAG Eval · run {run_id} · {ts}
    &nbsp;·&nbsp; eval: results/eval_{run_id}.json
    {'&nbsp;·&nbsp; ragas: results/ragas_' + run_id + '.json' if ragas_s else '&nbsp;·&nbsp; RAGAS non disponible pour ce run'}
  </div>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file",      nargs="?",  default=None, help="Fichier JSON eval (défaut: dernier)")
    parser.add_argument("--compare", nargs=2, metavar="FILE",  help="Comparer deux runs")
    args = parser.parse_args()

    if args.compare:
        data = load_file(args.compare[0])
        html = generate_html(data, compare=load_file(args.compare[1]))
    elif args.file:
        data = load_file(args.file)
        html = generate_html(data)
    else:
        data = load_latest()
        html = generate_html(data)

    run_id   = data["summary"].get("run_id", "latest")
    out_path = REPORT_DIR / f"report_{run_id}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Rapport généré → {out_path}")