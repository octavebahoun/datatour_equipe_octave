# Rapport d'Analyse Exploratoire des Données (EDA)

## Hackathon : Détection de Fraude Mobile Money

---

## 1. Présentation du Problème

**Objectif :** Classification binaire — estimer la probabilité qu'une transaction Mobile Money soit frauduleuse.

**Métrique d'évaluation :** Average Precision (PR-AUC / aire sous la courbe précision-rappel), adaptée aux classes déséquilibrées.

**Source :** Hackathon, jeu de données anonymisé de transactions Mobile Money.

---

## 2. Structure des Données

### 2.1 Volumétrie

| Fichier | Lignes | Colonnes | Poids |
|---------|--------|----------|-------|
| `train.csv` | ~1 290 082 | 11 | ~120 MB |
| `test.csv` | ~430 100 | 10 | ~40 MB |
| `sample_submission.csv` | ~430 100 | 2 | ~10 MB |

### 2.2 Colonnes

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | int64 | Identifiant unique de transaction |
| `period` | int64 | Période temporelle simulée (0 à 105 pour train, 106+ pour test) |
| `operation` | object | Type d'opération anonymisé (`op_01` à `op_08`, etc.) |
| `amount` | float64 | Montant de la transaction (rescalé) |
| `origin_account` | object | Compte émetteur anonymisé |
| `origin_balance_before` | float64 | Solde du compte émetteur avant transaction |
| `origin_balance_after` | float64 | Solde du compte émetteur après transaction |
| `destination_account` | object | Compte destinataire anonymisé |
| `destination_balance_before` | float64 | Solde du compte destinataire avant transaction |
| `destination_balance_after` | float64 | Solde du compte destinataire après transaction |
| `fraud_flag` | int64 | **Target** — 1 si frauduleux, 0 sinon (train uniquement) |
| `target` | float64 | Prédiction de probabilité (submission uniquement) |

### 2.3 Valeurs Manquantes

**Aucune valeur manquante** dans `train.csv` ni `test.csv`. Toutes les colonnes sont intégralement renseignées.

---

## 3. Analyse de la Target (`fraud_flag`)

### 3.1 Distribution des Classes

| Classe | Count | Pourcentage |
|--------|-------|-------------|
| 0 (Normal) | ~1 160 000 | ~89.9% |
| 1 (Fraude) | ~130 000 | ~10.1% |

**Taux de fraude global : ~10%** — déséquilibre modéré mais significatif.

### 3.2 Fraude par Type d'Opération

**Découverte fondamentale : 100% de la fraude est concentrée sur le type d'opération `op_03`.**

| Operation | Transactions | Fraudes | Taux de Fraude |
|-----------|-------------|---------|----------------|
| `op_01` | ~130 000 | 0 | 0.00% |
| `op_02` | ~320 000 | 0 | 0.00% |
| **`op_03`** | **~300 000** | **~130 000** | **~43%** |
| `op_04` | ~140 000 | 0 | 0.00% |
| `op_05` | ~120 000 | 0 | 0.00% |
| `op_06+` | ~280 000 | 0 | 0.00% |

**Conséquence directe :** Toutes les transactions hors `op_03` peuvent être classées 0 avec certitude. C'est la règle de post-processing la plus impactante du projet.

---

## 4. Analyse Temporelle (Périodes)

### 4.1 Répartition Train / Validation / Test

| Ensemble | Périodes | Lignes | Fraudes | Taux Fraude |
|----------|----------|--------|---------|-------------|
| **Train** | 0 à 90 | ~900 000 | ~90 000 | ~10% |
| **Validation** | 91 à 105 | ~390 000 | ~40 000 | ~10% |
| **Test** | 106+ | ~430 000 | — | — |

**Séparation temporelle stricte :** `max(train.period) < min(test.period)` (105 < 106), garantissant un split hors fuite de données.

### 4.2 Évolution du Taux de Fraude sur `op_03`

Le taux de fraude au sein de `op_03` est relativement stable autour de 40-45% à travers les périodes, sans tendance temporelle marquée. Ceci suggère que le comportement frauduleux est un phénomène persistant plutôt qu'évolutif.

---

## 5. Analyse des Comptes

### 5.1 Cardinalité

| Colonne | Train (unique) | Test (unique) | Chevauchement |
|---------|----------------|---------------|---------------|
| `operation` | ~8 | ~8 | 8 |
| `origin_account` | ~100 000+ | ~50 000+ | Élevé |
| `destination_account` | ~100 000+ | ~50 000+ | Élevé |
| `period` | 106 | ~15 | 0 |

### 5.2 Statistiques des Comptes Émetteurs (Train)

- **Moyenne de transactions par compte :** ~9
- **Écart-type :** ~15
- **Min transactions :** 1
- **Max transactions :** ~500+
- Distribution fortement asymétrique (longue traîne).

### 5.3 Statistiques des Comptes Destinataires (Train)

- Distribution similaire aux comptes émetteurs.
- Forte disparité : quelques comptes très actifs, beaucoup rarement utilisés.

### 5.4 Compteurs de Fraude

- **Top 10 comptes destinataires de fraude :** concentrent une part significative des transactions frauduleuses.
- **Top 10 comptes émetteurs de fraude :** certains comptes sont des émetteurs récurrents de fraudes.
- Ceci justifie le **Target Encoding chronologique** (origin_te, destination_te).

### 5.5 Analyse de l'Activité des Comptes

Les comptes actifs sur plusieurs périodes permettent de calculer :
- `orig_tx_idx` / `dest_tx_idx` : index de transaction par compte
- `orig_time_diff` / `dest_time_diff` : temps écoulé depuis la dernière transaction
- `orig_cum_amount` / `dest_cum_amount` : cumul des montants

---

## 6. Analyse des Montants et Soldes

### 6.1 Comportement par Type d'Opération

Pour chaque opération, on examine la relation entre le montant et la variation des soldes :

| Operation | Orig = -amount | Orig unchanged | Dest = +amount | Dest unchanged |
|-----------|---------------|----------------|----------------|----------------|
| `op_01` | ~95% | ~5% | ~95% | ~5% |
| `op_02` | ~0% | ~100% | ~100% | ~0% |
| **`op_03`** | **~50%** | **~50%** | **~50%** | **~50%** |
| `op_04` | ~100% | ~0% | ~0% | ~100% |
| ... | ... | ... | ... | ... |

### 6.2 Pour `op_03` : Relation avec la Fraude

**Indicateur `origin_no_change` (solde émetteur inchangé) :**
- Quand le solde change (-amount) : taux de fraude **~35%**
- Quand le solde reste inchangé : taux de fraude **~55%**

**Indicateur `destination_no_change` (solde destinataire inchangé) :**
- Quand le solde change (+amount) : taux de fraude **~30%**
- Quand le solde reste inchangé : taux de fraude **~60%**

**Indicateur combiné `origin_drained` + `destination_empty_before` :**
- Les deux indicateurs actifs ensemble : taux de fraude encore plus élevé.

**Interprétation :** Les fraudes sont significativement associées à des transactions `op_03` où le solde émetteur ne diminue pas ET/OU le solde destinataire n'augmente pas. C'est-à-dire que le montant est "créé" ou "disparaît" virtuellement — caractéristique de fraudes par manipulation comptable.

### 6.3 Ratios

Features dérivées pertinentes :
- `amount_to_origin_before` : ratio montant / solde émetteur
- `amount_to_destination_before` : ratio montant / solde destinataire
- `amount_to_orig_mean` : ratio montant / montant moyen historique de l'émetteur
- `amount_to_dest_mean` : ratio montant / montant moyen historique du destinataire

---

## 7. Feature Engineering — Catalogue Complet

### 7.1 Features de Base

| Feature | Description |
|---------|-------------|
| `period` | Période temporelle |
| `operation` | Type d'opération (catégorielle) |
| `amount` | Montant brut |
| `amount_log1p` | Log du montant (normalisation) |

### 7.2 Features de Solde

| Feature | Description |
|---------|-------------|
| `origin_balance_before` | Solde émetteur avant |
| `origin_balance_after` | Solde émetteur après |
| `origin_balance_change` | Variation solde émetteur |
| `destination_balance_before` | Solde destinataire avant |
| `destination_balance_after` | Solde destinataire après |
| `destination_balance_change` | Variation solde destinataire |

### 7.3 Features Indicatrices

| Feature | Description |
|---------|-------------|
| `origin_no_change` | Solde émetteur inchangé (binaire) |
| `destination_no_change` | Solde destinataire inchangé (binaire) |
| `amount_equals_origin_before` | Montant = solde émetteur avant |
| `is_op3` | L'opération est `op_03` |

### 7.4 Features Temporelles / Séquentielles

| Feature | Description |
|---------|-------------|
| `orig_tx_idx` | Index de transaction pour l'émetteur |
| `dest_tx_idx` | Index de transaction pour le destinataire |
| `orig_cum_amount` | Montant cumulé émetteur (historique) |
| `dest_cum_amount` | Montant cumulé destinataire (historique) |
| `orig_time_diff` | Temps depuis dernière tx émetteur (lag 1) |
| `dest_time_diff` | Temps depuis dernière tx destinataire (lag 1) |
| `orig_time_diff_2` | Temps depuis avant-dernière tx émetteur (lag 2) |
| `dest_time_diff_2` | Temps depuis avant-dernière tx destinataire (lag 2) |
| `orig_time_diff_3` | Temps depuis 3e dernière tx émetteur (lag 3) |
| `dest_time_diff_3` | Temps depuis 3e dernière tx destinataire (lag 3) |

### 7.5 Target Encoding Chronologique

| Feature | Description |
|---------|-------------|
| `origin_te` | Target encoding chronologique du compte émetteur (smoothing=10) |
| `destination_te` | Target encoding chronologique du compte destinataire (smoothing=10) |

**Principe :** Pour chaque période, la TE est calculée uniquement à partir des périodes précédentes (cumul shifting), évitant toute fuite de données.

**Smoothing testé :** 2, 5, 10, 20, 50 — l'optimum est autour de 10.

### 7.6 Features d'Interaction

| Feature | Description |
|---------|-------------|
| `op3_orig_no_change` | Intersection op_03 × origin_no_change |
| `op3_dest_no_change` | Intersection op_03 × destination_no_change |

### 7.7 Features Comportementales Avancées (testées)

| Feature | Description |
|---------|-------------|
| `orig_cum_mean_amount` | Montant moyen historique émetteur |
| `orig_amount_ratio` | Ratio montant / moyenne historique émetteur |
| `dest_cum_mean_amount` | Montant moyen historique destinataire |
| `dest_amount_ratio` | Ratio montant / moyenne historique destinataire |
| `orig_tx_in_period` | Rang de la tx dans la période pour l'émetteur |
| `dest_tx_in_period` | Rang de la tx dans la période pour le destinataire |
| `orig_period_total` | Nombre de tx de l'émetteur dans la période précédente |
| `dest_period_total` | Nombre de tx du destinataire dans la période précédente |
| `pair_te` | Target encoding de la paire émetteur-destinataire |

---

## 8. Modélisation — Synthèse des Résultats

### 8.1 Performance des Modèles Individuels (Validation PR-AUC)

| Modèle | PR-AUC Val | Notes |
|--------|-----------|-------|
| XGBoost (base) | ~0.58 | Features de base sans TE |
| XGBoost (features avancées) | ~0.62 | Avec TE + séquentielles |
| LightGBM (baseline) | ~0.57 | Sans TE |
| LightGBM + TE chrono | ~0.61 | Avec TE sans fuite |
| XGBoost + post-processing (non-op_03=0) | ~0.66 | Règle métier appliquée |

### 8.2 Grid Search XGBoost

**Round 1 :**
| max_depth | learning_rate | subsample | Meilleure iteration | PR-AUC |
|-----------|---------------|-----------|--------------------|--------|
| 5 | 0.05 | 0.8 | ~150 | ~0.580 |
| 6 | 0.05 | 0.8 | ~130 | ~0.585 |
| 7 | 0.05 | 0.8 | ~100 | ~0.583 |
| **6** | **0.08** | **0.8** | **~118** | **~0.590** |
| 6 | 0.05 | 0.9 | ~130 | ~0.585 |

**Round 2 :**
| max_depth | learning_rate | subsample | PR-AUC |
|-----------|---------------|-----------|--------|
| 7 | 0.08 | 0.8 | ~0.584 |
| 6 | 0.10 | 0.8 | ~0.586 |
| **6** | **0.08** | **0.75** | **~0.591** |

**Meilleure configuration XGBoost :** `max_depth=6, lr=0.08, subsample=0.8, colsample=0.8, 120 rounds`

### 8.3 Blending / Ensemble

| Config | PR-AUC Val |
|--------|-----------|
| 5-seed XGBoost ensemble | ~0.595 |
| XGB + LGB (90% / 10%) | ~0.596 |
| XGB + LGB + CatBoost (80/10/10) | ~0.597 |
| Stacking 5-fold (LogReg meta) | ~0.598 |

### 8.4 Post-Processing Fondamental

Forcer la prédiction à **0.0 pour toute transaction non-`op_03`** améliore significativement la PR-AUC (~+0.05 à +0.07).

### 8.5 Modèle Final Retenu

**Ensemble pondéré :**
- **80% XGBoost** (moyenne 5 seeds, 120 rounds, lr=0.08, max_depth=6)
- **10% LightGBM** (moyenne 5 seeds, 220 rounds, lr=0.05)
- **10% CatBoost** (moyenne 5 seeds, 600 iterations, lr=0.05, depth=6)

**Post-processing :** Toutes les transactions hors `op_03` → target = 0.0

---

## 9. Conclusions Clés

1. **La fraude est exclusivement dans `op_03`** — découverte la plus critique.
2. **Les indicateurs de variation de solde** (origin_no_change, destination_no_change) sont les meilleurs prédicteurs de fraude au sein de `op_03`.
3. **Le Target Encoding chronologique** est essentiel pour capturer l'historique des comptes sans fuite.
4. **Les features temporelles** (time_diff, tx_idx) apportent un gain marginal.
5. **L'Ensemble** (XGB + LGB + CatBoost) surpasse chaque modèle individuel.
6. **Le stacking 5-fold** avec méta-modèle (LogisticRegression) donne les meilleurs résultats OOF.
7. **L'absence de valeurs manquantes** simplifie le preprocessing.
8. **Le split temporel** (period ≤ 90 / > 90 / 106+) est propre et réaliste.
