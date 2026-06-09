"""
report_generator.py
Génère un rapport HTML visuel depuis un fichier eval_*.json.

Usage :
    python report_generator.py                         # dernier run
    python report_generator.py results/eval_XYZ.json  # run spécifique
    python report_generator.py --compare a.json b.json # comparaison
"""

import json
import sys
import re
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
    return {1: "Factuel", 2: "Négation/Cond.", 3: "Multi/Barème", 4: "Hors périmètre"}.get(t, str(t))


def gauge_svg(value: float, target: float, size: int = 72) -> str:
    """Mini SVG gauge (demi-cercle)."""
    if value is None:
        value = 0.0
    r     = size // 2 - 6
    cx    = size // 2
    cy    = size // 2
    angle = value * 180          # 0→0°  1→180°
    rad   = (angle - 180) * 3.14159 / 180
    import math
    ex = cx + r * math.cos(rad)
    ey = cy + r * math.sin(rad)
    color = "#4ade80" if value >= target else ("#facc15" if value >= target * 0.7 else "#f87171")
    bg_color = "#1e1e1e"
    return f"""
    <svg width="{size}" height="{size//2+8}" viewBox="0 0 {size} {size//2+8}">
      <path d="M {cx-r},{cy} A {r},{r} 0 0,1 {cx+r},{cy}"
            fill="none" stroke="{bg_color}" stroke-width="8" stroke-linecap="round"/>
      <path d="M {cx-r},{cy} A {r},{r} 0 0,1 {ex:.1f},{ey:.1f}"
            fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>
      <text x="{cx}" y="{cy+4}" text-anchor="middle"
            font-size="11" font-family="IBM Plex Mono,monospace" fill="{color}">
        {pct(value, 0)}
      </text>
    </svg>"""


# ══════════════════════════════════════════════════════════════════════════════
# Génération HTML
# ══════════════════════════════════════════════════════════════════════════════

HTML_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root {
  --bg:#0e0e0e; --surface:#141414; --border:#242424;
  --accent:#c8ff00; --text:#e0e0e0; --muted:#555;
  --ok:#4ade80; --warn:#facc15; --fail:#f87171;
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
.tier-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1rem;padding:0 2.5rem 1rem}
.tier-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1rem 1.2rem}
.tier-head{font-family:var(--font-mono);font-size:.75rem;letter-spacing:.08em;margin-bottom:.8rem;display:flex;justify-content:space-between;align-items:center}
.tier-name{color:var(--text)}
.tier-obj{color:var(--muted);font-size:.65rem}
.bar-row{display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-size:.72rem}
.bar-label{color:var(--muted);width:90px;flex-shrink:0}
.bar-track{flex:1;background:#1e1e1e;border-radius:99px;height:6px;overflow:hidden}
.bar-fill{height:100%;border-radius:99px;transition:width .3s}
.bar-val{color:var(--text);width:40px;text-align:right;font-family:var(--font-mono);font-size:.65rem}
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
.response-cell{max-width:320px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;color:var(--muted);font-size:.72rem}
.section-sep{border:none;border-top:1px solid var(--border);margin:1.5rem 2.5rem}
.hall-flag{color:var(--fail);font-family:var(--font-mono);font-size:.65rem}
.footer{font-family:var(--font-mono);font-size:.65rem;color:var(--muted);padding:1rem 2.5rem 2rem;border-top:1px solid var(--border);margin-top:2rem}
</style>
"""


def bar(value: float | None, target: float = 0.7) -> str:
    if value is None:
        return '<div class="bar-track"><div class="bar-fill" style="width:0%;background:#333"></div></div>'
    pv     = value * 100
    color  = "#4ade80" if value >= target else ("#facc15" if value >= target * 0.7 else "#f87171")
    return (f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pv:.1f}%;background:{color}"></div>'
            f'</div>')


def tier_card_html(tier_key: str, v: dict) -> str:
    targets  = {"tier_1": 0.90, "tier_2": 0.70, "tier_3": 0.50, "tier_4": 1.00}
    t_num    = int(tier_key.split("_")[1])
    obj      = targets.get(tier_key, 0.7)
    label    = tier_label(t_num)
    rows = ""
    for metric, label_m, tgt in [
        ("exact_match",   "Exact Match",   obj),
        ("token_recall",  "Token Recall",  obj),
        ("article_cited", "Article cité",  0.60),
        ("hallucination", "Hallucination", 0.00),  # inversé
    ]:
        val = v.get(metric)
        display_val = 1.0 - val if (metric == "hallucination" and val is not None) else val
        rows += (f'<div class="bar-row">'
                 f'<span class="bar-label">{label_m}</span>'
                 f'{bar(display_val, tgt if metric != "hallucination" else 0.9)}'
                 f'<span class="bar-val">{pct(val, 0)}</span>'
                 f'</div>')
    return f"""
    <div class="tier-card">
      <div class="tier-head">
        <span class="tier-name">Tier {t_num} — {label}</span>
        <span class="tier-obj">obj {pct(obj, 0)} · n={v['n']}</span>
      </div>
      {rows}
      <div style="margin-top:.6rem;font-size:.65rem;font-family:var(--font-mono);color:var(--muted)">
        latence moy. {v.get('avg_latency_s', '?')}s
      </div>
    </div>"""


def results_table_html(results: list) -> str:
    rows = ""
    for r in results:
        t = r.get("tier", 0)
        tc = f"t{t}"
        hall_info = ""
        if r.get("invented_articles"):
            hall_info = f'<div class="hall-flag">⚠ art. inventés: {r["invented_articles"]}</div>'
        full_response = (r.get("response") or "N/A").replace("<", "&lt;")
        rows += f"""
        <tr>
          <td><span class="tier-badge {tc}">T{t}</span></td>
          <td style="font-family:var(--font-mono);font-size:.65rem;color:var(--accent)">{r['id']}</td>
          <td style="max-width:220px">{r['question'][:90]}{'…' if len(r['question'])>90 else ''}</td>
          <td style="font-family:var(--font-mono);font-size:.7rem">{r.get('expected','')}</td>
          <td>{badge(r.get('exact_match'))}</td>
          <td style="font-family:var(--font-mono);font-size:.65rem">{pct(r.get('token_recall'))}</td>
          <td>{badge(r.get('article_cited'), '📎 oui', '✗ non')}</td>
          <td>{badge(not r.get('hallucination_flag'), '✓ propre', '⚠ suspect')}{hall_info}</td>
          <td style="font-family:var(--font-mono);font-size:.65rem">{r.get('latency_s','?')}s</td>
          <td class="response-cell" title="{full_response}">{full_response}</td>
        </tr>"""
    return f"""
    <table>
      <thead><tr>
        <th>Tier</th><th>ID</th><th>Question</th><th>Attendu</th>
        <th>EM</th><th>Recall</th><th>Article</th><th>Hallucin.</th>
        <th>Latence</th><th>Réponse</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def generate_html(data: dict, compare: dict | None = None) -> str:
    s  = data["summary"]
    rs = data.get("results", [])

    run_id = s.get("run_id", "?")
    ts     = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    em_cls    = "ok" if (s.get("exact_match") or 0) >= 0.70 else ("warn" if (s.get("exact_match") or 0) >= 0.50 else "danger")
    hall_cls  = "ok" if (s.get("hallucination_rate") or 0) <= 0.10 else "danger"
    kpi_cards = f"""
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
        <div class="card-val">{s.get('total_questions')}</div>
        <div class="card-sub">générées: {s.get('generated')}</div>
      </div>
    </div>"""

    # ── Tier cards ────────────────────────────────────────────────────────────
    tier_cards = "".join(
        tier_card_html(k, v) for k, v in s.get("by_tier", {}).items()
    )

    # ── Type table ────────────────────────────────────────────────────────────
    type_rows = ""
    for t, v in sorted(s.get("by_type", {}).items(), key=lambda x: -x[1]["exact_match"]):
        type_rows += (f'<tr><td style="font-family:var(--font-mono);font-size:.72rem">{t}</td>'
                      f'<td>{v["n"]}</td>'
                      f'<td>{bar(v["exact_match"], 0.7)}</td>'
                      f'<td style="font-family:var(--font-mono);font-size:.7rem">{pct(v["exact_match"])}</td>'
                      f'<td style="font-family:var(--font-mono);font-size:.7rem">{pct(v["token_recall"])}</td></tr>')
    type_table = f"""
    <table>
      <thead><tr><th>Type</th><th>n</th><th>EM (barre)</th><th>EM %</th><th>Recall</th></tr></thead>
      <tbody>{type_rows}</tbody>
    </table>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RAG Eval Report — {run_id}</title>
  {HTML_STYLE}
</head>
<body>
  <h1>⬡ RAG Audit — Rapport d'évaluation</h1>
  <div class="meta">Run ID: {run_id} &nbsp;·&nbsp; Généré le {ts}</div>

  <h2>KPIs globaux</h2>
  {kpi_cards}

  <hr class="section-sep">
  <h2>Résultats par tier</h2>
  <div class="tier-grid">{tier_cards}</div>

  <hr class="section-sep">
  <h2>Résultats par type de question</h2>
  {type_table}

  <hr class="section-sep">
  <h2>Détail des questions</h2>
  {results_table_html(rs)}

  <div class="footer">
    RAG Eval · run {run_id} · {ts}
    &nbsp;·&nbsp; Fichier source : results/eval_{run_id}.json
  </div>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default=None, help="Fichier JSON eval (défaut: dernier)")
    parser.add_argument("--compare", nargs=2, metavar="FILE", help="Comparer deux runs")
    args = parser.parse_args()

    if args.compare:
        data_a = load_file(args.compare[0])
        data_b = load_file(args.compare[1])
        # Rapport simple sur les deux (extensible)
        html = generate_html(data_a, compare=data_b)
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
