# DataTour 2026 — Dossier de Solution Finale

**Équipe :** Octave  
**Épreuve :** Détection de Fraude Mobile Money  
**Date :** Juillet 2026

---

## Table des matières

1. [Présentation du problème](#1-présentation-du-problème)
2. [Architecture de la solution](#2-architecture-de-la-solution)
3. [Feature engineering](#3-feature-engineering)
4. [Modèles de niveau 1](#4-modèles-de-niveau-1)
5. [Ensemble de niveau 2](#5-ensemble-de-niveau-2)
6. [Validation et reproductibilité](#6-validation-et-reproductibilité)
7. [Résultats et performance](#7-résultats-et-performance)
8. [Instructions d'exécution](#8-instructions-dexécution)

---

## 1. Présentation du problème

**Objectif :** Prédire une probabilité de fraude (entre 0 et 1) pour chaque transaction du jeu de test d'un système de Mobile Money africain.

**Métrique :** Average Precision (PR-AUC).

**Contraintes :**
- 1 290 081 transactions d'entraînement, 430 100 de test
- Déséquilibre : 10 % de fraude, 90 % de légitimes
- Split temporel : test dans le futur du train (périodes 106-143 vs 0-105)
- 100 % des fraudes concentrées sur le type d'opération `op_03`

---

## 2. Architecture de la solution

```
train.csv ──┐
            ├──► Feature Engineering (65+ features)
test.csv ───┘         │
                      ├── 5-Fold Stratified CV
                      │     ├── XGBoost (scale_pos_weight)
                      │     ├── LightGBM (is_unbalance)
                      │     ├── CatBoost (Balanced)
                      │     └── Logistic Regression (balanced)
                      │
                      └── Level 2 Ensemble
                            ├── Scipy blending optimisé
                            └── LightGBM meta-model stacking

                    ──► submission.csv
```

**Choix d'architecture :**
- 4 modèles diversifiés (2 boosting + 1 catboost + 1 linéaire) pour capturer différents patterns
- 2 stratégies d'ensemble concurrentes, sélection automatique de la meilleure
- Target Encoding chronologique (pas de leakage temporel)
- PageRank calculé uniquement sur les données d'entraînement

---

## 3. Feature engineering

### 3.1 Features de base (groupes)

| Groupe | Features | Description |
|--------|----------|-------------|
| Montant | `amount_log1p`, `is_round_1000`, `is_round_5000` | Transformation log, montants ronds |
| Balance origine | `origin_balance_change`, `origin_balance_ratio`, `amount_to_origin_before`, `origin_no_change`, `balance_anomaly_orig` | Variation, ratio, anomalie de solde |
| Balance destination | `destination_balance_change`, `dest_balance_ratio`, `amount_to_destination_before`, `dest_no_change`, `balance_anomaly_dest` | Même logique côté destinataire |
| Edge (relation) | `edge_time_diff`, `is_repeated_amount_on_edge`, `edge_cum_tx_count`, `edge_cum_amount_sum` | Dynamique de la paire (envoyeur, receveur) |
| Temporel compte | `orig_account_age`, `dest_account_age`, `orig_time_diff_{1,2,3}`, `dest_time_diff_{1,2,3}` | Ancienneté et rythme des comptes |

### 3.2 Features historiques par compte

| Feature | Formule | Utilité |
|---------|---------|---------|
| `orig_tx_idx` | Cumcount par origine | Nombre de transactions déjà effectuées |
| `orig_cum_amount` | Somme cumulée des montants (hors courante) | Volume total transféré |
| `orig_avg_amount` | Montant moyen historique | Profil de compte |
| `amount_vs_orig_avg` | ratio montant / moyenne | Déviation par rapport à la normale |
| `orig_cum_unique_dests` | Nombre de destinataires uniques | Comportement de dispersion |
| `orig_amount_rank` | Montant / max historique | Position relative |

### 3.3 Target Encoding chronologique

Nous utilisons un Target Encoding (TE) temporel qui calcule la proportion de fraude passée pour chaque groupe, avec **lissage bayésien** :

```python
te = (somme_antérieure + global_mean * smoothing) / (count_antérieur + smoothing)
```

| Feature | Groupe | Smoothing | Rôle |
|---------|--------|-----------|------|
| `origin_te` | Compte origine | 10 | Taux de fraude historique de l'envoyeur |
| `destination_te` | Compte destinataire | 10 | Taux de fraude historique du receveur |
| `edge_te` | Paire (orig, dest) | 5 | Taux de fraude de cette relation |
| `origin_te_smooth_20` | Compte origine | 20 | TE plus lissé (signal long-terme) |
| `dest_te_smooth_20` | Compte destination | 20 | Idem destinataire |
| `edge_te_smooth_3` | Paire | 3 | TE plus réactif (signal court-terme) |

### 3.4 Features de graphe (PageRank)

Le PageRank est calculé sur le graphe orienté des transactions d'entraînement (poids = nombre de transactions entre deux comptes). Permet d'identifier les comptes "centraux" dans le réseau transactionnel.

### 3.5 Features de rolling window

| Feature | Fenêtre | Description |
|---------|---------|-------------|
| `orig_amount_roll_mean_3` | 3 tx | Montant moyen des 3 dernières transactions |
| `orig_amount_roll_std_3` | 3 tx | Volatilité du montant sur 3 transactions |
| `orig_amount_zscore` | — | Écart du montant à la moyenne historique (en écarts-types) |
| `amount_velocity_accel_orig` | 1 tx | Accélération = vitesse(t) - vitesse(t-1) |

### 3.6 Features d'interaction

`amount_log1p_op3` : interaction entre le montant (log) et le type d'opération op_03, pour capturer l'effet spécifique des montants dans les transactions à risque.

---

## 4. Modèles de niveau 1

### 4.1 XGBoost

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| learning_rate | 0.08 | Compromis vitesse / précision |
| max_depth | 6 | Profondeur standard pour éviter l'overfitting |
| subsample | 0.8 | Échantillonnage lignes pour robustesse |
| colsample_bytree | 0.8 | Échantillonnage colonnes |
| scale_pos_weight | ~9 | Ratio légitimes / fraudes pour le déséquilibre |
| tree_method | hist | Optimisé pour grands volumes |
| early_stopping | 50 rounds | Arrêt automatique |
| n_estimators | 600 (max) | Suffisant avec early stopping |

### 4.2 LightGBM

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| learning_rate | 0.05 | Apprentissage plus fin que XGBoost |
| num_leaves | 31 | Complexité contrôlée |
| is_unbalance | True | Gestion automatique du déséquilibre |
| metric | average_precision | Aligné sur la métrique de la compétition |
| early_stopping | 50 rounds | Anti-overfitting |
| n_estimators | 800 (max) | Plus d'itérations possibles (plus rapide) |

### 4.3 CatBoost

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| iterations | 2000 | Beaucoup d'itérations avec early stopping |
| learning_rate | 0.08 | Standard |
| depth | 6 | Profondeur modérée |
| auto_class_weights | Balanced | Gestion du déséquilibre |
| eval_metric | AUC | Métrique interne |
| early_stopping_rounds | 80 | Patience un peu plus grande |

### 4.4 Logistic Regression

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| class_weight | balanced | Gestion du déséquilibre |
| C | 0.5 | Régularisation L2 modérée |
| solver | saga | Robuste pour grands volumes |
| StandardScaler | Oui | Normalisation préalable obligatoire |

**Rôle :** Apporte de la diversité à l'ensemble. Les modèles linéaires capturent des patterns différents des modèles à base d'arbres.

---

## 5. Ensemble de niveau 2

### 5.1 Scipy Blending

Optimisation des poids par minimisation SLSQP de la PR-AUC OOF :

```python
final = w1 * xgb + w2 * lgb + w3 * cb + w4 * lr
```

### 5.2 LightGBM Stacking

Un méta-modèle LightGBM entraîné sur les prédictions OOF des 4 modèles, enrichi de 20 features clés de niveau 1 (rolling stats, z-scores, TE, PageRank, etc.).

**Sélection automatique :** Le meilleur des deux méthodes est retenu pour la soumission finale.

---

## 6. Validation et reproductibilité

### 6.1 Stratégie de validation

- **5-fold Stratified Cross-Validation** avec shuffle et seed fixe (42)
- Les folds sont stratifiés sur `fraud_flag` pour maintenir la proportion de fraude
- **Validation croisée imbriquée** pour le stacking (seed 123)

### 6.2 Contrôle de l'aléa

- Toutes les seeds sont fixées : Python `random`, NumPy, et seeds de chaque modèle
- `random_state=42+fold` pour chaque fold de chaque modèle
- `random_state=123+fold` pour le méta-modèle
- Le hash SHA256 du fichier `submission.csv` est calculé et affiché pour traçabilité

### 6.3 Gestion du leakage temporel

- Target Encoding calculé de manière chronologique (les statistiques passées uniquement)
- PageRank calculé sur le train uniquement
- Pas de fit sur le test à aucun moment

---

## 7. Résultats et performance

| Modèle | OOF PR-AUC |
|--------|------------|
| XGBoost | À l'exécution |
| LightGBM | À l'exécution |
| CatBoost | À l'exécution |
| Logistic Regression | À l'exécution |
| **Scipy Blending** | À l'exécution |
| **LGB Stacking** | À l'exécution |
| **Final** | Max des deux méthodes |

*Les scores exacts sont affichés dans les logs de la solution lors de l'exécution.*

---

## 8. Instructions d'exécution

### Prérequis

```bash
pip install -r requirements.txt
```

### Exécution

```bash
# Depuis la racine du projet
python solution.py

# Avec chemins personnalisés (optionnel)
DATA_DIR=./dataset OUTPUT_DIR=. python solution.py
```

### Structure attendue

```
.
├── dataset/
│   ├── train.csv
│   └── test.csv
├── solution.py
├── requirements.txt
├── METHODOLOGIE.md
└── submission.csv  (généré)
```

Le fichier `submission.csv` est généré dans le répertoire de sortie spécifié (par défaut le répertoire courant).

### Temps d'exécution estimé

- Feature engineering : ~5-10 min
- Entraînement (4 modèles × 5 folds) : ~20-40 min
- Total : ~30-50 min selon la machine
