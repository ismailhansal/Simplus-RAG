"""
eval_dataset.py
Jeu de 30 questions de test — CGI Maroc 2026
8 catégories diagnostiques + 3 questions hors-périmètre

Structure par catégorie (ce qu'elle diagnostique) :
  CAT 1 — keyword/BM25 pur       : si ça rate ici → problème d'indexation
  CAT 2 — sémantique/embedding   : si ça rate ici → embedding trop faible
  CAT 3 — numérique/tableaux     : si ça rate ici → chunking fragmente les tableaux
  CAT 4 — négatif/exclusion      : si ça rate ici → retrieval remonte les règles générales
  CAT 5 — multi-conditions       : si ça rate ici → RAG agrège mal plusieurs chunks
  CAT 6 — temporel/transitoire   : si ça rate ici → Livre II sous-représenté
  CAT 7 — piège/ambiguïté        : si ça rate ici → LLM suit son intuition
  CAT 8 — procédure/sanction     : si ça rate ici → Livre II sous-représenté

Champs :
  id              : identifiant unique
  tier            : niveau de difficulté (1=facile … 4=hors périmètre)
  cat             : catégorie diagnostique (1-8)
  type            : sous-type fin
  question        : question posée au RAG
  expected_answer : valeur clé attendue dans la réponse (exact match)
  expected_article: article CGI source (vérifie si bien cité)
  keywords_hint   : mots qui DOIVENT apparaître dans les chunks remontés
"""

EVAL_SET = [

    # ═══════════════════════════════════════════════════════════════
    # CAT 1 — KEYWORD / BM25 PUR  (tier 1 — objectif >90%)
    # Mots exacts présents dans le texte, retrieval trivial.
    # Si le RAG échoue ici → problème fondamental d'indexation.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q01", "tier": 1, "cat": 1, "type": "taux_keyword",
        "question": "Quel est le taux de TVA applicable aux opérations d'hébergement et de restauration ?",
        "expected_answer": "10%",
        "expected_article": "article 99",
        "keywords_hint": ["hébergement", "restauration", "10", "TVA", "taux réduit"],
    },
    {
        "id": "q02", "tier": 1, "cat": 1, "type": "taux_keyword",
        "question": "Quel est le taux de l'IS applicable aux établissements de crédit et à Bank Al-Maghrib ?",
        "expected_answer": "40%",
        "expected_article": "article 19",
        "keywords_hint": ["établissements de crédit", "Bank Al-Maghrib", "40", "IS"],
    },
    {
        "id": "q03", "tier": 1, "cat": 1, "type": "règle_générale",
        "question": "Les agents de l'administration fiscale sont-ils tenus au secret professionnel ?",
        "expected_answer": "oui",
        "expected_article": "article 246",
        "keywords_hint": ["secret professionnel", "agents", "administration fiscale", "lois pénales"],
    },
    {
        "id": "q04", "tier": 1, "cat": 1, "type": "taux_spécifique",
        "question": "Quel est le taux de la cotisation minimale pour les ventes de produits pétroliers, gaz, beurre, huile, sucre, farine, eau, électricité et médicaments ?",
        "expected_answer": "0,15%",
        "expected_article": "article 144",
        "keywords_hint": ["cotisation minimale", "0,15", "produits pétroliers", "médicaments", "farine"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 2 — SÉMANTIQUE / EMBEDDING  (tier 2 — objectif >70%)
    # Le mot-clé exact est absent, le sens doit guider le retrieval.
    # Si le RAG échoue ici → embedding trop faible ou BM25 trop dominant.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q05", "tier": 2, "cat": 2, "type": "synonymie_concept",
        "question": "Une startup marocaine nouvellement créée est-elle obligée de payer des impôts dès la première année d'activité ?",
        "expected_answer": "pas nécessairement",
        "expected_article": "article 6",
        "keywords_hint": ["exonération", "première exploitation", "société nouvellement créée", "IS", "temporaire"],
    },
    {
        "id": "q06", "tier": 2, "cat": 2, "type": "synonymie_concept",
        "question": "Mon entreprise vend uniquement à des clients étrangers. Est-ce qu'elle bénéficie d'avantages fiscaux au Maroc ?",
        "expected_answer": "oui",
        "expected_article": "article 6",
        "keywords_hint": ["exportation", "exportatrices", "exonération", "chiffre d'affaires", "IS"],
    },
    {
        "id": "q07", "tier": 2, "cat": 2, "type": "synonymie_négation",
        "question": "Une entreprise peut-elle déduire la TVA sur l'achat d'un véhicule de tourisme destiné à son directeur général ?",
        "expected_answer": "non",
        "expected_article": "article 106",
        "keywords_hint": ["véhicules de transport de personnes", "déduction", "TVA", "exclusion"],
    },
    {
        "id": "q08", "tier": 2, "cat": 2, "type": "synonymie_concept",
        "question": "Une société étrangère qui a une succursale au Maroc doit-elle payer l'IS sur les bénéfices générés par cette succursale ?",
        "expected_answer": "oui",
        "expected_article": "article 5",
        "keywords_hint": ["établissement stable", "succursale", "source marocaine", "IS", "non résidente"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 3 — NUMÉRIQUE EXACT / TABLEAUX  (tier 2 — objectif >70%)
    # Seuils précis, taux dans des barèmes, valeurs dans des tableaux.
    # Si le RAG échoue ici → chunking qui fragmente les tableaux.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q09", "tier": 2, "cat": 3, "type": "seuil_précis",
        "question": "À partir de quel montant de bénéfice net une société est-elle soumise au taux IS de 35% ?",
        "expected_answer": "100 000 000",
        "expected_article": "article 19",
        "keywords_hint": ["35%", "100 000 000", "bénéfice net", "IS", "taux"],
    },
    {
        "id": "q10", "tier": 2, "cat": 3, "type": "borne_barème",
        "question": "Quelle est la tranche du revenu exonérée dans le barème de l'impôt sur le revenu 2026 ?",
        "expected_answer": "40 000",
        "expected_article": "article 73",
        "keywords_hint": ["40 000", "exonéré", "barème", "IR", "tranche"],
    },
    {
        "id": "q11", "tier": 2, "cat": 3, "type": "barème_double_entrée",
        "question": "Quel est l'abattement forfaitaire applicable aux pensions de retraite pour la part annuelle ne dépassant pas 168 000 dirhams ?",
        "expected_answer": "70%",
        "expected_article": "article 60",
        "keywords_hint": ["abattement", "pension", "retraite", "168 000", "70"],
    },
    {
        "id": "q12", "tier": 3, "cat": 3, "type": "seuil_rare",
        "question": "Quel est le seuil de chiffre d'affaires consolidé à partir duquel une entreprise doit déposer une déclaration pays par pays ?",
        "expected_answer": "8 122 500 000",
        "expected_article": "article 154 ter",
        "keywords_hint": ["déclaration pays par pays", "8 122 500 000", "chiffre d'affaires consolidé"],
    },
    {
        "id": "q13", "tier": 2, "cat": 3, "type": "distinction_barème",
        "question": "Quel est le taux d'imposition des profits nets de cession d'actions cotées en bourse par une personne physique ?",
        "expected_answer": "15%",
        "expected_article": "article 73",
        "keywords_hint": ["actions cotées", "bourse", "15%", "personne physique", "profit net"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 4 — NÉGATIF / EXCLUSION  (tier 2-3 — objectif >60%)
    # Ce que la loi interdit, exclut ou limite.
    # Si le RAG échoue ici → retrieval remonte les règles générales
    # et rate les dispositions d'exclusion.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q14", "tier": 2, "cat": 4, "type": "négation_déduction",
        "question": "Une entreprise peut-elle déduire de son résultat fiscal une amende pour infraction à la législation du travail ?",
        "expected_answer": "non",
        "expected_article": "article 11",
        "keywords_hint": ["amende", "non déductibles", "pénalités", "législation du travail"],
    },
    {
        "id": "q15", "tier": 2, "cat": 4, "type": "négation_durée",
        "question": "Une société peut-elle reporter indéfiniment un déficit fiscal sur les exercices suivants ?",
        "expected_answer": "non",
        "expected_article": "article 12",
        "keywords_hint": ["report", "déficit", "quatrième exercice", "limité"],
    },
    {
        "id": "q16", "tier": 3, "cat": 4, "type": "exception_cachée",
        "question": "Une société installée dans une Zone d'Accélération Industrielle est-elle soumise au taux IS de 35% si son bénéfice dépasse 100 millions de dirhams ?",
        "expected_answer": "non",
        "expected_article": "article 19",
        "keywords_hint": ["Zone d'Accélération Industrielle", "ZAI", "35%", "exception", "exclusion"],
    },
    {
        "id": "q17", "tier": 3, "cat": 4, "type": "exception_régime",
        "question": "Les dividendes versés entre sociétés marocaines sont-ils toujours soumis à la retenue à la source IS de 10% ?",
        "expected_answer": "non",
        "expected_article": "article 6",
        "keywords_hint": ["dividendes", "retenue à la source", "mère-fille", "abattement", "exonération"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 5 — MULTI-CONDITIONS / MULTI-ARTICLES  (tier 3 — objectif >50%)
    # Réponse complète dispersée sur plusieurs chunks ou articles.
    # Si le RAG échoue ici → agrégation insuffisante.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q18", "tier": 3, "cat": 5, "type": "conditions_multiples",
        "question": "Un promoteur immobilier peut-il bénéficier d'une exonération de TVA sur les logements sociaux ? Si oui, quelles sont toutes les conditions ?",
        "expected_answer": "50 et 80 m²",
        "expected_article": "article 92",
        "keywords_hint": ["logement social", "superficie", "250 000", "habitation principale", "4 ans", "TVA"],
    },
    {
        "id": "q19", "tier": 3, "cat": 5, "type": "conditions_cumulatives",
        "question": "Quelles sont les conditions cumulatives pour qu'une société bénéficie de l'exclusion du taux IS de 35% via un investissement de 1,5 milliard de dirhams ?",
        "expected_answer": "1er janvier 2023",
        "expected_article": "article 19",
        "keywords_hint": ["1,5 milliard", "convention", "immobilisations corporelles", "10 ans", "2023"],
    },
    {
        "id": "q20", "tier": 3, "cat": 5, "type": "double_condition",
        "question": "Une coopérative agricole qui transforme les matières premières de ses adhérents et réalise un CA annuel de 8 millions DH HT est-elle exonérée d'IS ?",
        "expected_answer": "oui",
        "expected_article": "article 7",
        "keywords_hint": ["coopérative", "agricole", "10 millions", "transformation", "exonération", "IS"],
    },
    {
        "id": "q21", "tier": 3, "cat": 5, "type": "triple_condition",
        "question": "Les dividendes versés par une société en Zone d'Accélération Industrielle à un non-résident, provenant de bénéfices de source étrangère, sont-ils exonérés de retenue à la source ?",
        "expected_answer": "oui",
        "expected_article": "article 6",
        "keywords_hint": ["ZAI", "non-résident", "source étrangère", "dividendes", "exonération", "retenue"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 6 — TEMPOREL / TRANSITOIRE  (tier 2-3 — objectif >60%)
    # Délais, prescriptions, régimes transitoires 2026.
    # Si le RAG échoue ici → Livre II sous-représenté dans le retrieval.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q22", "tier": 2, "cat": 6, "type": "délai_prescription",
        "question": "Dans quel délai l'administration fiscale peut-elle rectifier une omission dans une déclaration IS ?",
        "expected_answer": "quatrième année",
        "expected_article": "article 232",
        "keywords_hint": ["délai", "prescription", "rectifier", "quatrième année", "clôture"],
    },
    {
        "id": "q23", "tier": 2, "cat": 6, "type": "délai_réclamation",
        "question": "Quel est le délai dont dispose un contribuable pour contester un impôt mis en recouvrement par voie de rôle ?",
        "expected_answer": "six mois",
        "expected_article": "article 235",
        "keywords_hint": ["délai", "réclamation", "rôle", "six mois", "recouvrement"],
    },
    {
        "id": "q24", "tier": 3, "cat": 6, "type": "transitoire_2026",
        "question": "Quel est le taux IS applicable à une société dont le bénéfice net est inférieur ou égal à 300 000 dirhams pour l'exercice ouvert à compter du 1er janvier 2026 ?",
        "expected_answer": "20%",
        "expected_article": "article 247",
        "keywords_hint": ["300 000", "2026", "transitoire", "20%", "convergence"],
    },
    {
        "id": "q25", "tier": 2, "cat": 6, "type": "délai_déclaratif",
        "question": "Quel est le délai de dépôt de la déclaration de résultat fiscal IS après la clôture de l'exercice ?",
        "expected_answer": "3 mois",
        "expected_article": "article 20",
        "keywords_hint": ["déclaration", "résultat fiscal", "IS", "3 mois", "clôture"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 7 — PIÈGE / AMBIGUÏTÉ  (tier 3 — objectif >50%)
    # Bornes exactes, exceptions cachées, intuition incorrecte.
    # Si le RAG échoue ici → LLM suit son intuition au lieu du texte.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q26", "tier": 3, "cat": 7, "type": "borne_inclusive",
        "question": "Un contribuable dont le revenu annuel est exactement de 40 000 dirhams paie-t-il de l'IR ?",
        "expected_answer": "non",
        "expected_article": "article 73",
        "keywords_hint": ["40 000", "exonéré", "jusqu'à", "tranche", "IR"],
    },
    {
        "id": "q27", "tier": 3, "cat": 7, "type": "condition_cachée",
        "question": "Le taux IS de 35% s'applique-t-il automatiquement dès qu'une société dépasse 100 millions DH de bénéfice net ?",
        "expected_answer": "non",
        "expected_article": "article 19",
        "keywords_hint": ["3 exercices consécutifs", "35%", "100 millions", "condition", "20%"],
    },
    {
        "id": "q28", "tier": 3, "cat": 7, "type": "confusion_régime",
        "question": "Les panneaux photovoltaïques sont-ils exonérés de TVA ?",
        "expected_answer": "non",
        "expected_article": "article 99",
        "keywords_hint": ["photovoltaïques", "taux réduit", "10%", "TVA", "exonération"],
    },

    # ═══════════════════════════════════════════════════════════════
    # CAT 8 — PROCÉDURE / SANCTION  (tier 2-3 — objectif >60%)
    # Recours, pénalités, obligations déclaratives, Livre II.
    # Si le RAG échoue ici → Livre II sous-indexé.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q29", "tier": 2, "cat": 8, "type": "barème_pénalité",
        "question": "En cas de paiement spontané d'un impôt avec un retard de 20 jours, quelle est la pénalité applicable ?",
        "expected_answer": "5%",
        "expected_article": "article 208",
        "keywords_hint": ["pénalité", "retard", "30 jours", "5%", "paiement spontané"],
    },
    {
        "id": "q30", "tier": 2, "cat": 8, "type": "sanction_employeur",
        "question": "Un employeur qui n'opère pas la retenue à la source sur les salaires de ses employés, quelles sont les conséquences fiscales pour lui ?",
        "expected_answer": "amendes",
        "expected_article": "article 208",
        "keywords_hint": ["retenue à la source", "salaires", "amende", "majoration", "intérêts de retard"],
    },

    # ═══════════════════════════════════════════════════════════════
    # HORS PÉRIMÈTRE  (tier 4 — objectif 100% refus propre)
    # Le RAG doit dire "je ne sais pas", pas inventer.
    # ═══════════════════════════════════════════════════════════════
    {
        "id": "q_hors_01", "tier": 4, "cat": None, "type": "hors_périmètre",
        "question": "Quel est le régime fiscal applicable aux revenus générés par les cryptomonnaies au Maroc ?",
        "expected_answer": "non présent",
        "expected_article": None,
        "keywords_hint": [],
    },
    {
        "id": "q_hors_02", "tier": 4, "cat": None, "type": "hors_périmètre",
        "question": "Quelle est la TVA applicable aux NFT vendus par des artistes marocains ?",
        "expected_answer": "non présent",
        "expected_article": None,
        "keywords_hint": [],
    },
    {
        "id": "q_hors_03", "tier": 4, "cat": None, "type": "ambiguë",
        "question": "Quel est le taux applicable aux dividendes ?",
        "expected_answer": "dépend",
        "expected_article": None,
        "keywords_hint": ["dividendes", "retenue à la source"],
    },
]
