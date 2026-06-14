# Guide — Évaluation RAGAS

## Installation

```bash
pip install ragas datasets langchain-community langchain-core
```

Rien d'autre : `sentence-transformers` et `multilingual-e5-small` sont déjà dans ton env.

---

## Fichiers

```
ragas_eval.py          # module principal
eval_ragas_patch.py    # 4 blocs à coller dans eval.py
```

---

## Utilisation

### Mode autonome (sans toucher à eval.py)

```bash
# Toutes les questions
python ragas_eval.py

# Un seul tier
python ragas_eval.py --tier 1

# Questions précises
python ragas_eval.py --ids q01,q04,q07

# Filtrer sur un document
python ragas_eval.py --doc-id <uuid>

# Relire le dernier résultat sans relancer
python ragas_eval.py --analyze

# Relire un résultat spécifique
python ragas_eval.py --analyze --result results/ragas_20250610_143022.json
```

### Mode intégré dans eval.py

Colle les 4 blocs de `eval_ragas_patch.py` dans `eval.py`, puis :

```bash
# eval normal + RAGAS automatiquement
python eval.py

# Les scores RAGAS apparaissent dans le terminal et dans eval_XXXXXXXX.json
```

---

## Outputs

| Fichier | Contenu |
|---|---|
| `logs/ragas_XXXXXXXX.jsonl` | Un enregistrement par question |
| `logs/ragas_XXXXXXXX.log` | Log texte lisible |
| `results/ragas_XXXXXXXX.json` | Résultats complets + summary |

---

## Métriques

| Métrique | Question | Cible |
|---|---|---|
| **Faithfulness** | La réponse est-elle fidèle aux chunks ? | > 80% |
| **Answer Relevancy** | La réponse répond-elle à la question ? | > 75% |
| **Context Precision** | Les chunks récupérés sont-ils utiles ? | > 70% |
| **Context Recall** | A-t-on récupéré tous les chunks nécessaires ? | > 70% |

### Comment lire les alertes

- `⚠ Faible Faithfulness (< 0.5)` → le LLM a probablement halluciné
- `⚠ Faible Context Recall (< 0.5)` → le retrieval manque des chunks importants

---

## Config (haut de ragas_eval.py)

```python
OLLAMA_MODEL  = "minimax-m3:cloud"               # LLM évaluateur
OLLAMA_BASE_URL = "http://localhost:11434"
E5_MODEL_NAME = "intfloat/multilingual-e5-small" # embeddings
```

Même modèles que `app.py` — rien à changer.
