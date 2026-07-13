# Détection de fraude — Ensemble Stacking + Leak-Free PageRank

Pipeline de détection de fraude sur transactions mobile money.
Combine 3 modèles de gradient boosting via stacking, avec des features
temporelles, relationnelles et de graphe — toutes causales (aucun leakage).

## Score

2ᵉ place — hackathon DataTour (DataAfrique Hub), 1.29M transactions, ~10% de fraude.

## Installation

```bash
pip install pandas numpy xgboost lightgbm catboost scikit-learn scipy networkx
```

## Utilisation

Place `train.csv` et `test.csv` dans le même dossier que le script, puis :

```bash
python solution.py
```

Sortie : `soumission.csv` (colonnes `id`, `target`).

## Comment ça marche

**1. Feature engineering** — construit ~60 caractéristiques :
- montants/soldes (ratios, différences, indicateurs binaires)
- dynamiques d'arêtes (temps entre transactions d'une même paire de comptes)
- PageRank sur le graphe des transactions (calculé sur le **train seul**)
- ancienneté des comptes, rang des montants, cumuls historiques
- target encoding chronologique (origine, destination, arête)

**2. Niveau 1** — XGBoost, LightGBM, CatBoost en validation croisée
stratifiée 5 plis. Génère des prédictions hors-sac (OOF).

**3. Niveau 2** — deux stratégies comparées sur la PR-AUC OOF :
- mélange linéaire optimisé (Scipy SLSQP)
- méta-modèle LightGBM (stacking sur OOF + top features)

La meilleure des deux est retenue automatiquement.

**4. Post-traitement** — seules les transactions `op_03` sont scorées
(exigence de la métrique) ; les autres sont forcées à 0.

## Point clé : pas de leakage

Tous les calculs cumulés utilisent `shift(1)` / `cumsum` → uniquement le passé.
Le PageRank est calculé sur le train seul ; les comptes inconnus au test → 0.

## Métrique

PR-AUC (average precision) — adaptée au fort déséquilibre de classes.
