# 📊 DataTour 2026 — Documentation EDA & Préparation des Données

> **Contexte** : Compétition DataTour 2026 — Phase Nationale  
> **Problème** : Détecter des transactions frauduleuses dans un système de **Mobile Money africain**  
> **Livrable** : Probabilité de fraude (entre 0 et 1) pour chaque transaction du jeu de test  
> **Audience** : Équipe OpenAI — Bourse ML 2026

---

## 🗺️ Table des Matières

1. [Présentation du Problème](#1-présentation-du-problème)
2. [Description du Dataset](#2-description-du-dataset)
3. [Analyse de la Variable Cible](#3-analyse-de-la-variable-cible)
4. [Analyse par Type d'Opération — Insight Majeur](#4-analyse-par-type-dopération--insight-majeur)
5. [Analyse Temporelle](#5-analyse-temporelle)
6. [Analyse des Montants](#6-analyse-des-montants)
7. [Analyse des Soldes (Balances)](#7-analyse-des-soldes-balances)
8. [Analyse des Comptes](#8-analyse-des-comptes)
9. [Corrélations](#9-corrélations)
10. [Détection d'Outliers](#10-détection-doutliers)
11. [Feature Engineering — 29 Nouvelles Variables](#11-feature-engineering--29-nouvelles-variables)
12. [Préparation Finale des Données](#12-préparation-finale-des-données)
13. [Choix de la Métrique](#13-choix-de-la-métrique)
14. [Résumé des Insights](#14-résumé-des-insights)

---

## 1. Présentation du Problème

### C'est quoi le Mobile Money ?

Le Mobile Money, c'est envoyer et recevoir de l'argent via son téléphone portable, **sans compte bancaire classique**. C'est la norme en Afrique subsaharienne (M-Pesa au Kenya, Orange Money, Airtel Money…). Des millions de transactions par jour.

### Pourquoi y a-t-il de la fraude ?

Les fraudeurs exploitent ces systèmes de plusieurs façons :
- **Phishing / vol de code** : ils volent les identifiants d'un vrai utilisateur
- **Social engineering** : ils se font passer pour un agent et arnaquent les gens
- **Account takeover** : ils prennent le contrôle d'un compte et vident l'argent
- **Mule accounts** : ils utilisent des comptes intermédiaires pour blanchir

### Notre mission

On a **1 290 081 transactions historiques** avec une étiquette `fraud_flag`. On doit créer un système capable de prédire **la probabilité de fraude** pour **430 100 nouvelles transactions** (le futur).

> **Important** : Ce n'est pas une classification dure ("fraude / pas fraude"), c'est une **probabilité** entre 0 et 1. Les équipes de sécurité peuvent alors fixer leur propre seuil d'alerte selon leurs contraintes métier.

---

## 2. Description du Dataset

### Fichiers disponibles

| Fichier | Taille | Lignes | Colonnes | Rôle |
|---|---|---|---|---|
| `train.csv` | 155 Mo | **1 290 081** | 11 | Données d'entraînement (avec étiquettes) |
| `test.csv` | 52 Mo | **430 100** | 10 | Données à prédire (sans étiquettes) |
| `sample_submission.csv` | 17 Mo | 430 100 | 2 | Format attendu pour la soumission |

### Description colonne par colonne

| Colonne | Type | Exemple | Description simple |
|---|---|---|---|
| `id` | Texte | `dtf_0000001_ffa5beb5` | Identifiant unique de la transaction |
| `period` | Entier | 0, 1, 2… 105 | Horodatage simulé — comme un numéro de semaine |
| `operation` | Texte | `op_03` | Type de transaction (5 types : op_01 à op_05) |
| `amount` | Décimal | 636.75 | Montant en unité monétaire |
| `origin_account` | Texte | `acc_o_307358...` | Compte envoyeur (anonymisé) |
| `origin_balance_before` | Décimal | 87.00 | Solde envoyeur **avant** la transaction |
| `origin_balance_after` | Décimal | -549.75 | Solde envoyeur **après** la transaction |
| `destination_account` | Texte | `acc_d_7fac3b...` | Compte receveur (anonymisé) |
| `destination_balance_before` | Décimal | 630.88 | Solde receveur **avant** |
| `destination_balance_after` | Décimal | 1267.62 | Solde receveur **après** |
| `fraud_flag` | 0 ou 1 | 0 | **Variable cible** — absent du test |

### Qualité des données

```
✅ 0 valeurs manquantes dans le train  (parfait — rien à imputer)
✅ 0 doublons dans le train
✅ 0 valeurs manquantes dans le test
✅ 1 290 081 IDs uniques (pas de doublon d'identifiant)
```

---

## 3. Analyse de la Variable Cible

### Les vrais chiffres

```
Transactions légitimes     : 1 160 539   (89.96%)
Transactions frauduleuses  :   129 542   (10.04%)
Ratio                      :  1 fraude pour  9 transactions légitimes
```

### Le problème de déséquilibre — Expliqué simplement

Imaginez une ville avec 10 000 habitants. 9 000 sont honnêtes, 1 000 sont des voleurs. Si un policier dit "tout le monde est honnête", il a raison 90% du temps — mais il est complètement inutile pour attraper des voleurs.

C'est **exactement notre situation**. Un modèle idiot qui dit "jamais de fraude" aurait **89.96% d'accuracy**. C'est pourquoi on ne regarde pas l'accuracy, et pourquoi il faut des techniques spéciales.

### Techniques pour gérer le déséquilibre

| Technique | Comment ça marche | Verdict |
|---|---|---|
| `class_weight='balanced'` | Le modèle pénalise 9× plus les erreurs sur fraude | ✅ Simple, efficace |
| `scale_pos_weight=9` (XGBoost) | Idem, natif XGBoost | ✅ Recommandé |
| SMOTE | Génère des fraudes synthétiques | ⚠️ Lent, risque de bruit |
| Sous-échantillonnage légitimes | Retire des légitimes | ⚠️ Perd de l'info |

> **Notre choix** : `class_weight='balanced'` pour les modèles linéaires et forêts, `scale_pos_weight=9` pour XGBoost/LightGBM.

---

## 4. Analyse par Type d'Opération — Insight Majeur

### Résultats complets (vraies données)

| Opération | Volume | Fraudes | **Taux de fraude** | Montant moyen |
|---|---|---|---|---|
| `op_01` | 4 087 | 0 | **0.00%** | 9 442 |
| `op_02` | 71 876 | 0 | **0.00%** | 153 589 |
| `op_03` | **415 323** | **129 542** | **31.19%** 🔴 | 50 302 |
| `op_04` | 305 443 | 0 | **0.00%** | 163 184 |
| `op_05` | 493 352 | 0 | **0.00%** | 1 944 |

### 🚨 Découverte fondamentale

**100% des fraudes (129 542 sur 129 542) sont dans `op_03`.**

Les opérations `op_01`, `op_02`, `op_04`, `op_05` n'ont **zéro fraude**. Cette découverte transforme le problème :
- Si `operation ≠ op_03` → probabilité de fraude = **0%** (quasi certitude)
- Si `operation = op_03` → probabilité de fraude = **31.2%** (à analyser plus finement)

`op_03` ressemble à une opération de **transfert d'argent / retrait cash** — celles où l'argent quitte définitivement le système. Ce sont historiquement les plus exploitées par les fraudeurs car l'argent est immédiatement disponible.

> La feature `operation_fraud_rate` que nous créons encode directement cette information : 0.00 pour op_01/02/04/05, 0.3119 pour op_03.

---

## 5. Analyse Temporelle

### Structure temporelle du dataset

```
Train : périodes    0 → 105   (106 périodes = ~106 semaines)
Test  : périodes  106 → 143   (38 périodes DANS LE FUTUR)
```

**Le test est chronologiquement après le train.** Le modèle doit prédire des transactions qui se passent dans le futur de l'historique d'entraînement. C'est un problème de **prévision temporelle**, pas juste de classification.

### Variation du taux de fraude dans le temps

```
Taux de fraude minimum (par période) :  4.08%
Taux de fraude maximum (par période) : 18.37%
Taux de fraude moyen                 : 10.04%
```

Le taux de fraude varie du **simple au quadruple** selon la période. Cela suggère des **vagues d'attaques organisées** — des groupes de fraudeurs qui opèrent intensément pendant un temps, puis se taisent.

### ⚠️ La règle d'or : Toujours un split temporel

**ERREUR COURANTE** : Mélanger aléatoirement train et validation.

```python
# ❌ À NE PAS FAIRE
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)
```

Pourquoi c'est faux ? Le modèle verrait des transactions "du futur" (période 100) pendant l'entraînement, puis serait validé sur des transactions "du passé" (période 10). Ce n'est pas ce qui se passe en production.

```python
# ✅ CE QU'ON FAIT
SPLIT_PERIOD = train_fe['period'].quantile(0.80)  # = 76
X_train = data[data['period'] <= 76]   # périodes 0-76
X_val   = data[data['period'] > 76]    # périodes 77-105
```

---

## 6. Analyse des Montants

### Statistiques comparatives (vraies données)

| Statistique | Légitime | Fraude | Ce qu'on apprend |
|---|---|---|---|
| Minimum | **0.37** | **19 815** | Toutes les fraudes > 19 815 ! |
| Médiane | 20 224 | 23 150 | Proches |
| Moyenne | 67 969 | 30 043 | Légitimes plus variables |
| Maximum | 2 526 958 | 559 774 | Très grands = souvent légitimes |
| Écart-type | 110 417 | **20 999** | Fraudes dans une fourchette étroite |
| Q25 | 708 | 20 146 | Toutes les fraudes dans la zone haute |
| Q75 | 101 133 | 28 898 | — |
| Q99 | 427 724 | 95 288 | — |

### Insights clés sur les montants

1. **Seuil minimum de fraude = 19 815** : Ce n'est pas une coïncidence. Les fraudeurs visent des montants suffisamment grands pour valoir le coup, mais pas trop grands pour ne pas déclencher des alertes.

2. **Fourchette étroite** : L'écart-type des fraudes (21 000) est 5× plus faible que celui des légitimes (110 000). Les fraudeurs opèrent dans une "zone de confort" de montants.

3. **Très grands montants = légitimes** : Les transferts d'entreprises, virements internationaux, etc. ont des montants très élevés mais sont légitimes.

> **Test statistique** Mann-Whitney (robuste aux distributions non-normales) :  
> **p-value = 1.68 × 10⁻²³** → Différence **hautement significative**. Le montant est une feature utile.

---

## 7. Analyse des Soldes (Balances)

### Le test de cohérence comptable — Notre feature la plus puissante

Dans une transaction honnête, la comptabilité doit être équilibrée :

```
Côté envoyeur  : balance_after = balance_before - amount
Côté receveur  : balance_after = balance_before + amount
```

On mesure l'écart entre ce qu'il devrait y avoir et ce qu'il y a :

```python
erreur_origine      = |origin_balance_before  - amount - origin_balance_after|
erreur_destination  = |dest_balance_before    + amount - dest_balance_after|
```

### Résultats de la vérification comptable (vraies données)

```
Incohérences côté origine :
  Dans les transactions légitimes : 534 384  (46.0%)
  Dans les transactions frauduleuses :  39 893  (30.8%)
```

> **Note** : Ces erreurs existent aussi chez les légitimes car le système Mobile Money inclut des frais de transaction qui ne sont pas explicitement dans les données. Ce n'est pas une erreur de données — c'est une caractéristique du système. Le modèle apprendra à distinguer les "erreurs normales" des "erreurs anormales".

### Soldes des comptes (vraies données)

| Statistique | Légitime | Fraude |
|---|---|---|
| Médiane solde origine avant | 3 144 788 | 2 976 154 |
| Médiane solde destination avant | **60 907** | **23 150** |
| Comptes à solde négatif (origine) | 2.4% | 3.2% |

Les comptes destination des fraudes ont des soldes médians **2.6× plus faibles** que les légitimes. Les fraudeurs envoient l'argent vers des comptes "froids" avec peu d'historique.

---

## 8. Analyse des Comptes

### Chiffres clés

```
Comptes origine uniques      : 13 431
Comptes destination uniques  : 15 818
Total transactions train     : 1 290 081

Médiane transactions par compte  :  46
Moyenne transactions par compte  :  96.1
Maximum transactions par compte  : 1 401
```

### Comportement suspect des comptes

On peut calculer, pour chaque compte, son **historique de comportement** :
- Combien de transactions au total ?
- Quel est son taux de fraude passé ?
- Quel est son montant moyen ?
- Quel est le delta de balance moyen ?

Ces **agrégats comportementaux** sont très informatifs car les comptes utilisés pour la fraude ont souvent des patterns distincts.

> **Attention data leakage** : Ces agrégats sont calculés **uniquement sur le train** et ensuite appliqués au test. On ne regarde jamais les transactions futures.

---

## 9. Corrélations

### Corrélations Spearman avec fraud_flag (vraies données)

*Spearman plutôt que Pearson car les données ne sont pas normalement distribuées.*

| Feature | Corrélation Spearman | Interprétation |
|---|---|---|
| `destination_balance_before` | **-0.156** | Soldes bas → plus de fraude |
| `amount` | **+0.065** | Montant élevé → légèrement plus de fraude |
| `origin_balance_after` | -0.049 | — |
| `origin_balance_before` | -0.040 | — |
| `destination_balance_after` | -0.037 | — |

> **Les corrélations semblent faibles** — ne vous y trompez pas ! Les modèles à base d'arbres (XGBoost, LightGBM) captent des relations **non-linéaires et combinées** que le coefficient de Spearman ne voit pas. Par exemple : "si amount > 20 000 ET operation == op_03 ET balance_error > 0", la probabilité de fraude est très élevée — cette règle est non-linéaire.

---

## 10. Détection d'Outliers

### Méthode IQR à 3 sigma

```
IQR = Q75 - Q25
Outlier si : valeur < Q25 - 3×IQR  ou  valeur > Q75 + 3×IQR
```

*(On utilise 3× au lieu de 1.5× car les données financières ont naturellement une forte variance)*

### Résultats (vraies données)

| Colonne | Outliers | % du total | Taux fraude parmi outliers |
|---|---|---|---|
| `amount` | 22 924 | **1.78%** | 0.4% (moins de fraude dans les très grands montants) |
| `origin_balance_before` | 122 | 0.01% | 7.4% |
| `destination_balance_before` | 17 350 | **1.34%** | 9.9% |

### Que faire des outliers ?

**On ne les supprime pas.** Voici pourquoi :

1. Les gros montants légitimes (transferts d'entreprises) sont réels et importants
2. XGBoost et LightGBM sont **naturellement robustes aux outliers** (les arbres splittent sur des seuils, peu importe les valeurs extrêmes)
3. On applique `log(x + 1)` pour réduire l'effet de levier des valeurs extrêmes

---

## 11. Feature Engineering — 29 Nouvelles Variables

### Qu'est-ce que le Feature Engineering ?

C'est l'art de **créer de nouvelles informations** à partir des données existantes, pour aider le modèle à apprendre plus facilement.

**Analogie** : Imaginez que vous voulez savoir si quelqu'un est malade. Vous avez sa température (38.5°C) et la normale (37°C). Vous pourriez donner juste ces deux chiffres au médecin. Ou vous pourriez calculer `fièvre = temp - 37 = 1.5°C`. Ce "feature engineered" est plus direct et utile.

### Les 29 features créées

#### 📦 Groupe A — Features de Montant (3)

| Feature | Formule | Raison |
|---|---|---|
| `log_amount` | `log(amount + 1)` | Réduit l'impact des très grosses valeurs |
| `amount_is_round` | `amount % 500 == 0` | Les fraudeurs font souvent des montants ronds |
| `amount_is_high` | `amount > percentile_95` | Gros montants = profil distinct |

#### ⚖️ Groupe B — Balance Origine (8)

| Feature | Formule | Raison |
|---|---|---|
| `origin_balance_delta` | `balance_before - balance_after` | Ce qui a vraiment été débité |
| `origin_balance_error` | `\|before - amount - after\|` | **Incohérence comptable !** |
| `origin_has_balance_error` | `error > 0.01 → 1/0` | Version binaire |
| `origin_amount_ratio` | `amount / (balance_before + 1)` | Transaction grande vs solde ? |
| `origin_account_emptied` | `balance_after < 1` | Compte complètement vidé |
| `origin_balance_negative_before` | `balance_before < 0` | Compte à découvert |
| `origin_balance_negative_after` | `balance_after < 0` | Mis à découvert |
| `log_origin_balance_before/after` | `log(\|balance\| + 1)` | Version log |

#### 📥 Groupe C — Balance Destination (7)

| Feature | Description |
|---|---|
| `dest_balance_delta` | Augmentation réelle côté destination |
| `dest_balance_error` | Incohérence comptable destination |
| `dest_has_balance_error` | Version binaire |
| `dest_amount_ratio` | Montant / solde destination |
| `dest_balance_unchanged` | **Balance destination inchangée** (très suspect !) |
| `log_dest_balance_before/after` | Version log |

> `dest_balance_unchanged` : si on envoie de l'argent mais que le solde du destinataire ne bouge pas... c'est très bizarre.

#### 🔗 Groupe D — Features Croisées (3)

| Feature | Formule | Raison |
|---|---|---|
| `both_balance_error` | `err_o AND err_d` | Incohérence des DEUX côtés = très suspect |
| `any_balance_error` | `err_o OR err_d` | Au moins une incohérence |
| `balance_error_asymmetry` | `log(err_o) - log(err_d)` | Asymétrie de l'erreur |

#### 🏷️ Groupe E — Opération (2)

| Feature | Description |
|---|---|
| `operation_code` | Encodage numérique (op_01→0, op_03→2, etc.) |
| `operation_fraud_rate` | **0.3119** pour op_03, **0.0** pour les autres |

#### ⏰ Groupe F — Temporel (2)

| Feature | Description |
|---|---|
| `period` | Période brute |
| `period_normalized` | `period / max_period` — normalisé entre 0 et 1 |

#### 👤 Groupe G — Comportement Compte (4)

| Feature | Description |
|---|---|
| `acc_o_n_tx` | Nombre total de transactions du compte origine |
| `acc_o_fraud_rate` | Taux de fraude historique du compte |
| `acc_o_mean_amount` | Montant moyen des transactions du compte |
| `acc_o_mean_delta` | Delta de balance moyen du compte |

### Validation des features — Analyse de lift (vraies données)

| Feature | Taux fraude si 0 | Taux fraude si 1 | Lift | Observations |
|---|---|---|---|---|
| `origin_has_balance_error` | **12.5%** | **6.9%** | — | Les erreurs existent dans les 2 classes |
| `dest_has_balance_error` | 14.7% | 5.0% | — | Idem |
| `any_balance_error` | 14.4% | 7.1% | — | Signal inversé par rapport à l'attendu |
| `origin_account_emptied` | 10.1% | 0.0% | — | Comptes vidés = légitimes ! |
| `dest_balance_unchanged` | 13.9% | 2.2% | 0.2× | Fort signal inverse |
| `amount_is_high` | 10.6% | **0.2%** | 0.02× | Très gros montants = légitimes |

> **Surprise des données réelles !** Certaines intuitions étaient fausses. Par exemple, on s'attendait à ce que les "balance errors" soient plus fréquentes dans les fraudes, mais ce n'est pas le cas ici. Les légitimes ont aussi beaucoup d'erreurs (à cause des frais). Le modèle ML apprendra les vraies combinaisons gagnantes.

---

## 12. Préparation Finale des Données

### Pipeline complet

```
train.csv (raw) ─── 1 290 081 lignes × 11 colonnes
        │
        ├── [1] Vérification qualité : 0 nul, 0 doublon ✅
        │
        ├── [2] Feature Engineering : +29 nouvelles features (→ 40 colonnes)
        │
        ├── [3] Agrégats comportementaux par compte (sur train uniquement)
        │
        ├── [4] Split temporel :
        │      ├── Train  : périodes  0-76  → 1 035 820 lignes (80.3%) | 9.10% fraude
        │      └── Val    : périodes 77-105 →   254 261 lignes (19.7%) | 13.86% fraude
        │
        └── [5] Sauvegarde :
               ├── train_prepared.csv  (1 035 820 × 36)
               ├── val_prepared.csv    (  254 261 × 36)
               └── test_prepared.csv   (  430 100 × 35)
```

### Les 35 features finales sélectionnées

```python
FEATURE_COLS = [
    # Temps
    'period', 'period_normalized',
    
    # Montant
    'amount', 'log_amount', 'amount_is_round', 'amount_is_high',
    
    # Balance origine (brut)
    'origin_balance_before', 'origin_balance_after',
    
    # Balance origine (engineered)
    'origin_balance_delta', 'origin_balance_error',
    'origin_has_balance_error', 'origin_amount_ratio',
    'origin_account_emptied', 'origin_balance_negative_before',
    'origin_balance_negative_after', 'log_origin_balance_before',
    'log_origin_balance_after',
    
    # Balance destination (brut)
    'destination_balance_before', 'destination_balance_after',
    
    # Balance destination (engineered)
    'dest_balance_delta', 'dest_balance_error',
    'dest_has_balance_error', 'dest_amount_ratio',
    'dest_balance_unchanged', 'log_dest_balance_before',
    'log_dest_balance_after',
    
    # Features croisées
    'both_balance_error', 'any_balance_error',
    'balance_error_asymmetry',
    
    # Opération
    'operation_code', 'operation_fraud_rate',
    
    # Comportement compte
    'acc_o_n_tx', 'acc_o_fraud_rate',
    'acc_o_mean_amount', 'acc_o_mean_delta',
]
```

---

## 13. Choix de la Métrique

### Pourquoi pas l'Accuracy ?

Un modèle bête qui dit "jamais de fraude" obtiendrait **89.96% d'accuracy**. Inutile.

### ROC-AUC — Notre métrique principale

**ROC** = Receiver Operating Characteristic  
**AUC** = Area Under the Curve

**En termes simples** : Si je prends une vraie fraude et une vraie transaction légitime au hasard, le ROC-AUC mesure la probabilité que le modèle attribue un score **plus élevé** à la fraude. Un score de 0.5 = hasard total. Un score de 1.0 = parfait.

```
AUC = 0.5    → Modèle inutile (tirage au sort)
AUC = 0.70   → Passable
AUC = 0.85   → Bien
AUC = 0.95   → Très bien
AUC = 1.00   → Parfait (suspect — probablement data leakage !)
```

> C'est la **métrique officielle de DataTour 2026**.

### Autres métriques utiles (pour analyse interne)

| Métrique | Description | Formule |
|---|---|---|
| **Précision** | Sur les alarmes levées, combien vraies ? | TP / (TP + FP) |
| **Rappel** | Sur les fraudes réelles, combien détectées ? | TP / (TP + FN) |
| **F1-Score** | Compromis précision/rappel | 2PR / (P+R) |
| **PR-AUC** | Aire sous courbe précision-rappel | Intégrale |

> En fraude financière, le **rappel** est souvent prioritaire : mieux vaut une fausse alarme que rater une vraie fraude.

---

## 14. Résumé des Insights

### 🔑 Les 7 découvertes clés (vraies données)

| # | Insight | Impact |
|---|---|---|
| 1 | **100% des fraudes sont en `op_03`** (31.2% de taux) | 🔴 Feature n°1 |
| 2 | **Toutes les fraudes ont amount > 19 815** | 🔴 Feature n°2 |
| 3 | **Les fraudes ont une fourchette de montant étroite** (σ = 21k vs 110k) | 🟠 Signal fort |
| 4 | **Les comptes destination des fraudes ont des soldes 2.6× plus faibles** | 🟠 Signal fort |
| 5 | **Le taux de fraude varie de 4% à 18% selon la période** | 🟡 Patterns temporels |
| 6 | **Split temporel obligatoire** (test = futur du train) | 🔴 Méthodologique |
| 7 | **Déséquilibre 90/10** → class_weight obligatoire | 🔴 Méthodologique |

### ⚠️ Points de vigilance

- **Data leakage** : Les agrégats par compte calculés uniquement sur le train
- **Validation** : Toujours un split temporel, jamais aléatoire
- **Métrique** : Uniquement ROC-AUC pour comparer les modèles
- **Outliers** : Ne pas supprimer — transformer en log

### 🎯 Données prêtes pour la modélisation

```
train_prepared.csv  :  1 035 820 lignes  ×  35 features  |  9.10% fraude
val_prepared.csv    :    254 261 lignes  ×  35 features  | 13.86% fraude
test_prepared.csv   :    430 100 lignes  ×  35 features  |  à prédire
```

---

*📁 Fichiers produits : `eda_local.py` (script), `EDA_DOCUMENTATION.md` (ce fichier), `notebook_colab.ipynb` (notebook), `choix_modeles.md` (conseil modèles)*  
*Auteur : Équipe Octave | DataTour 2026 | Juin 2026*
