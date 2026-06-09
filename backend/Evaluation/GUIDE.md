# RAG Eval — Guide rapide

## Lancer les tests

```bash
# Toutes les questions (33) — ~5 min
python eval.py

# Un seul tier
python eval.py --tier 1        # 4 questions faciles, ~30s
python eval.py --tier 2        # 15 questions moyennes
python eval.py --tier 3        # 11 questions difficiles
python eval.py --tier 4        # 3 questions hors-périmètre

# Questions précises (debug ciblé)
python eval.py --ids q01,q07,q14

# Sur un doc spécifique (récupère le doc_id depuis /documents)
python eval.py --doc-id 3f7a2b1c-...

# Tester le retrieval sans passer par le LLM
python eval.py --retrieve-only
```

## Analyser les logs

```bash
# Analyser le dernier run (stats par étape du pipeline)
python eval.py --analyze

# Analyser un run spécifique
python eval.py --analyze --log logs/pipeline_20250607_143022.jsonl
```

## Rapport HTML

```bash
# Générer depuis le dernier run
python report_generator.py

# Depuis un run spécifique
python report_generator.py results/eval_20250607_143022.json
# → ouvre reports/report_*.html dans ton navigateur
```

---

## Ce que tu lis dans le terminal

```
▶ [q04] T2 | Une entreprise peut-elle déduire une amende...
  HyDE   → amende pénalité non déductible résultat fiscal [CHANGED]
  VECTOR → 18 chunks | top: Article 11 - Charges non déductibles
  BM25   → 20 chunks | top: Article 11-I
  RRF    → 20 fused | top3: ['Article 11', 'Article 10', 'Article 13']
  LLM    → 3.2s | 210 chars
  RESP   → Non, les amendes et pénalités de toute nature ne sont pas...
  SCORE  → EM=True Recall=1.0 Art=True Hall=False KW=0.8
■ [q04] done in 4.1s
```

| Ligne   | Ce que ça dit                                      |
|---------|----------------------------------------------------|
| HyDE    | La query a-t-elle été réécrite ? En quoi ?         |
| VECTOR  | Combien de chunks remontés, lequel est en tête     |
| BM25    | Idem pour la recherche par mots-clés               |
| RRF     | Résultat de la fusion des deux canaux              |
| LLM     | Latence + longueur de la réponse                   |
| RESP    | Début de la réponse générée                        |
| SCORE   | Métriques calculées automatiquement (voir ci-dessous) |

### Décoder la ligne SCORE

```
SCORE → EM=True Recall=1.0 Art=False Hall=False KW=0.8
```

**EM** — Exact Match  
Est-ce que la valeur clé attendue est présente dans la réponse ?  
`True` = la réponse contient "10%" (ou "non", ou "quatrième année"...)  
`False` = la valeur attendue est absente ou mal formulée

**Recall** — Token Recall (entre 0.0 et 1.0)  
Fraction des mots importants de la réponse attendue qui apparaissent dans la réponse.  
`1.0` = tous les mots attendus sont là  
`0.5` = la moitié des mots attendus seulement  
`0.0` = aucun mot attendu trouvé  
→ Un Recall élevé avec EM=False signifie que le sens y est mais pas la valeur exacte.

**Art** — Article cité  
Est-ce que la réponse mentionne le bon article du CGI ?  
`True` = l'article attendu (ex: "article 99") est cité dans la réponse  
`False` = aucun article cité, ou le mauvais article  
→ Art=False avec EM=True = bonne réponse mais sans sourcer. Risque en audit.

**Hall** — Hallucination  
Est-ce que le LLM a inventé un article qui n'existe pas ?  
`False` = aucun article suspect détecté ✅  
`True` = la réponse cite un article avec un numéro > 300 (le CGI s'arrête ~270) ⚠️  
→ Hall=True est le signal le plus grave : le modèle invente des références légales.

**KW** — Keyword Hit Rate (entre 0.0 et 1.0)  
Fraction des mots-clés attendus qui sont présents dans les chunks remontés par le retrieval.  
`1.0` = tous les bons mots-clés sont dans les chunks récupérés  
`0.5` = la moitié seulement  
`0.0` = aucun mot-clé attendu dans les chunks → le retrieval a complètement raté  
→ KW=0.0 avec EM=False = le problème vient du retrieval, pas du LLM.

### Exemple de lecture combinée

| EM    | Recall | Art   | Hall  | KW  | Diagnostic                                              |
|-------|--------|-------|-------|-----|---------------------------------------------------------|
| True  | 1.0    | True  | False | 0.8 | ✅ Parfait                                              |
| True  | 1.0    | False | False | 0.8 | ⚠️ Bonne réponse mais article non cité                 |
| False | 0.7    | True  | False | 0.8 | Retrieval OK, LLM a mal formulé la réponse              |
| False | 0.3    | False | False | 0.2 | Retrieval raté → mauvais chunks → mauvaise réponse      |
| False | 0.0    | True  | True  | 0.0 | 🚨 Hallucination : LLM invente avec contexte vide       |
| True  | 1.0    | False | False | 0.0 | Retrieval raté mais LLM a répondu juste par chance      |

---

## Ce que tu lis dans le résumé final

```
  Exact Match     : 61.2%    ← valeur clé présente dans la réponse
  Token Recall    : 74.3%    ← fraction des mots attendus trouvés
  Article cité    : 58.4%    ← bon article du CGI mentionné
  Hallucination   : 9.1%     ← articles inventés (cible < 10%)
  Keyword hit     : 67.2%    ← bons chunks remontés par le retrieval
```

```
── Par catégorie diagnostique ──
  ✅ cat_1 Keyword/BM25 pur       EM=100%  → indexation OK
  ✅ cat_2 Sémantique/Embedding   EM=75%   → embedding OK
  ❌ cat_4 Négatif/Exclusion      EM=25%   → retrieval remonte mauvais chunks
  ❌ cat_7 Piège/Ambiguïté        EM=33%   → LLM suit son intuition
```

---

## Diagnostic rapide par catégorie

| Catégorie qui rate | Cause probable            | Où chercher              |
|--------------------|---------------------------|--------------------------|
| cat_1 Keyword      | Indexation cassée         | `ensure_collection()`    |
| cat_2 Sémantique   | Embedding trop faible     | modèle E5, HyDE          |
| cat_3 Tableaux     | Chunking fragmente        | `chunk_by_article()`     |
| cat_4 Négatif      | Retrieval trop générique  | BM25 scores, RRF weights |
| cat_5 Multi-cond.  | top_k trop petit          | `top_k` dans retrieve    |
| cat_6 Temporel     | Livre II sous-indexé      | vérifier chunks Livre II |
| cat_7 Piège        | LLM ignore le contexte    | system prompt, temp      |
| cat_8 Procédure    | Livre II sous-indexé      | vérifier chunks Livre II |

---

## Fichiers produits après chaque run

```
logs/
  pipeline_20250607_143022.jsonl   ← toutes les étapes en JSON ligne par ligne
  pipeline_20250607_143022.log     ← même chose, lisible humain

results/
  eval_20250607_143022.json        ← résultats + métriques complets

reports/
  report_20250607_143022.html      ← rapport visuel (si report_generator.py lancé)
```

---

## Workflow typique de debug

```bash
# 1. Lancer les factuels pour valider que la base fonctionne
python eval.py --tier 1

# 2. Si tout est vert, lancer tout
python eval.py

# 3. Voir quelles catégories ratent dans le résumé

# 4. Cibler les questions qui échouent
python eval.py --ids q14,q15,q16

# 5. Analyser ce que le retrieval remonte
python eval.py --ids q14 --retrieve-only

# 6. Corriger (chunking, prompt, top_k...)

# 7. Relancer pour mesurer le gain
python eval.py --tier 2
```
