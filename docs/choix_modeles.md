# 🤖 Choix des Modèles — DataTour 2026
## Détection de Fraude Mobile Money

> Ce document justifie chaque modèle envisagé, explique pourquoi on les choisit (ou pas),
> et détaille la stratégie de modélisation complète — **sans entraîner ni développer le modèle**.

---

## 🎯 Rappel du problème

- **Type** : Classification binaire (fraude / pas fraude)
- **Sortie** : Probabilité entre 0 et 1 (pas un label dur)
- **Métrique** : ROC-AUC (mesure la séparation des deux classes)
- **Déséquilibre** : 10% de fraude vs 90% de légitimes
- **Taille** : 1.29M transactions train, 430K test
- **Structure** : Données tabulaires (pas d'images, pas de texte)
- **Temps** : Le test est dans le futur du train → problème temporel

---

## 📊 Tableau Comparatif des Modèles

| Modèle | Vitesse | ROC-AUC attendu | Gestion déséquilibre | Interprétabilité | Verdict |
|---|---|---|---|---|---|
| Régression Logistique | ⚡⚡⚡ Très rapide | 0.75-0.82 | `class_weight='balanced'` | ⭐⭐⭐ Très bonne | 🟡 Baseline |
| Random Forest | ⚡⚡ Rapide | 0.85-0.92 | `class_weight='balanced'` | ⭐⭐ Moyenne | 🟠 Bon |
| **XGBoost** | ⚡⚡ Rapide | **0.90-0.97** | `scale_pos_weight=9` | ⭐⭐ Moyenne | ✅ **Recommandé** |
| **LightGBM** | ⚡⚡⚡ Très rapide | **0.90-0.97** | `is_unbalance=True` | ⭐⭐ Moyenne | ✅ **Recommandé** |
| CatBoost | ⚡⚡ Rapide | 0.90-0.96 | `class_weights` | ⭐⭐ Moyenne | 🟠 Alternatif |
| MLP (réseau de neurones) | ⚡ Lent | 0.85-0.94 | `class_weight` | ⭐ Faible | 🟡 Optionnel |
| SVM | ❌ Très lent (1.3M) | 0.80-0.88 | `class_weight='balanced'` | ⭐ Faible | ❌ Trop lent |

---

## 1. 📏 Régression Logistique — Le Baseline

### C'est quoi ?

La régression logistique est le modèle le plus simple pour la classification. C'est une ligne droite (ou un hyperplan en multi-dimensions) qui sépare les fraudes des légitimes. Elle produit directement une probabilité.

**Analogie** : Imaginez qu'on trace une ligne sur un graphique "montant vs solde". Tout ce qui est d'un côté de la ligne = légitime, de l'autre côté = fraude.

### Pourquoi on l'utilise ?

```
✅ Extrêmement rapide à entraîner
✅ Très interprétable (les coefficients ont un sens direct)
✅ Bonne métrique de baseline pour comprendre si les features sont bonnes
✅ Robuste — pas de risque d'overfitting avec peu de features
```

### Ses limites pour notre problème

```
❌ Capture uniquement les relations LINÉAIRES
❌ Ne peut pas apprendre "si amount > 20 000 ET op_03 → fraude"
❌ Sensible aux variables à grande échelle (nécessite normalisation)
❌ Avec 35 features non-linéaires, elle sera vite dépassée
```

### Configuration recommandée

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# IMPORTANT : normaliser les features pour la régression logistique
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(
    class_weight='balanced',    # compense le déséquilibre 90/10
    C=1.0,                      # régularisation (évite l'overfitting)
    max_iter=1000,              # assez d'itérations pour converger
    random_state=42,
    n_jobs=-1                   # utilise tous les CPU disponibles
)
```

**ROC-AUC attendu : 0.75 – 0.82**

> C'est notre point de départ. Si on n'atteint pas au moins 0.75 avec la régression logistique, il y a un problème dans les données ou les features.

---

## 2. 🌲 Random Forest — Le Polyvalent

### C'est quoi ?

Un Random Forest, c'est une **forêt d'arbres de décision**. Chaque arbre pose des questions successives :
- "Le montant est-il > 19 815 ?"
- "L'opération est-elle op_03 ?"
- "Le solde destination est-il < 30 000 ?"

Chaque arbre vote, et on prend la majorité. C'est comme demander l'avis à 500 experts différents et prendre la décision la plus souvent votée.

### Pourquoi c'est bien pour notre problème ?

```
✅ Capture les relations non-linéaires (crucial ici)
✅ Robuste aux outliers (les arbres splittent sur des seuils)
✅ Pas besoin de normaliser les features
✅ Donne une importance des features (très utile pour comprendre)
✅ Peu sensible aux hyperparamètres — difficile de vraiment rater
✅ Parallélisable facilement
```

### Ses limites

```
❌ Moins précis que XGBoost/LightGBM sur des données tabulaires
❌ Lent à prédire sur de très grands datasets (mémoire importante)
❌ Peut overfitter si les arbres sont trop profonds
```

### Configuration recommandée

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=500,           # 500 arbres (bon compromis)
    max_depth=None,             # arbres profonds pour capturer la complexité
    min_samples_leaf=20,        # évite l'overfitting
    class_weight='balanced',    # compense le déséquilibre
    max_features='sqrt',        # standard Random Forest
    random_state=42,
    n_jobs=-1
)
```

**ROC-AUC attendu : 0.85 – 0.92**

---

## 3. ⚡ XGBoost — Notre Champion n°1

### C'est quoi ?

XGBoost (eXtreme Gradient Boosting) est un algorithme de **boosting** : au lieu de construire des arbres indépendants (comme Random Forest), il les construit **séquentiellement**. Chaque nouvel arbre corrige les erreurs du précédent.

**Analogie** : Imaginez un apprenti qui apprend de ses erreurs. Premier arbre : il fait des erreurs sur les transactions avec de gros montants. Deuxième arbre : il se concentre sur ces transactions ratées. Troisième arbre : il corrige les nouvelles erreurs. Et ainsi de suite.

### Pourquoi XGBoost domine les problèmes tabulaires ?

```
✅ Meilleure performance que Random Forest sur données tabulaires (en général)
✅ Gestion native du déséquilibre via scale_pos_weight
✅ Régularisation intégrée (L1/L2) pour éviter l'overfitting
✅ Très rapide avec la version GPU
✅ Importance des features très précise
✅ Supporte les valeurs manquantes nativement (pas besoin d'imputer)
✅ Utilisé par ~70% des gagnants de compétitions Kaggle tabulaires
```

### Ses limites

```
⚠️ Plus sensible aux hyperparamètres que Random Forest
⚠️ Peut overfitter si mal configuré (learning_rate trop élevé)
⚠️ Training lent sans GPU sur 1.3M lignes (mais raisonnable)
```

### Configuration recommandée

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    # Performance
    n_estimators=1000,          # beaucoup d'arbres (on arrête tôt avec early stopping)
    learning_rate=0.05,         # petit lr = meilleure généralisation
    max_depth=6,                # profondeur standard
    
    # Déséquilibre de classes
    scale_pos_weight=9,         # 9 légitimes pour 1 fraude → weight = 9
    
    # Régularisation
    subsample=0.8,              # utilise 80% des données par arbre
    colsample_bytree=0.8,       # utilise 80% des features par arbre
    min_child_weight=10,        # évite de splitter sur trop peu d'exemples
    reg_alpha=0.1,              # régularisation L1
    reg_lambda=1.0,             # régularisation L2
    
    # Technique
    eval_metric='auc',          # optimise sur notre métrique
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',         # rapide même sans GPU
)

# Utilisation avec early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,   # arrête si pas d'amélioration sur 50 rounds
    verbose=100
)
```

**ROC-AUC attendu : 0.90 – 0.97**

---

## 4. 🚀 LightGBM — Notre Champion n°2 (souvent le meilleur)

### C'est quoi ?

LightGBM (Light Gradient Boosting Machine) est développé par **Microsoft**. C'est la même idée que XGBoost (boosting), mais avec une implémentation différente qui le rend **beaucoup plus rapide** sur de grands datasets.

La différence clé : XGBoost fait croître ses arbres "en largeur" (level-wise), LightGBM les fait croître "en profondeur" (leaf-wise). Résultat : LightGBM trouve des patterns plus précis, plus vite.

### Pourquoi préférer LightGBM à XGBoost ?

```
✅ 5 à 10× plus rapide que XGBoost sur de grands datasets (1.3M lignes ici)
✅ Utilise moins de mémoire
✅ Aussi précis, voire plus précis que XGBoost
✅ Supporte les features catégorielles nativement (pas besoin d'encoder)
✅ Paramètre is_unbalance=True pour le déséquilibre
```

### Ses limites

```
⚠️ Peut overfitter si le num_leaves est trop grand
⚠️ Moins stable que XGBoost (plus sensible aux hyperparamètres)
⚠️ Légèrement moins bien documenté
```

### Configuration recommandée

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    # Performance
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,              # max_depth équivalent ~ log2(63) ≈ 6
    
    # Déséquilibre
    is_unbalance=True,          # gère automatiquement le 90/10
    # Alternative : class_weight='balanced'
    
    # Régularisation
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    
    # Technique
    metric='auc',
    random_state=42,
    n_jobs=-1,
    verbose=-1                  # pas de logs parasites
)

# Avec early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)
```

**ROC-AUC attendu : 0.90 – 0.97**

---

## 5. 🐱 CatBoost — L'Alternatif Russe

### C'est quoi ?

CatBoost (Categorical Boosting) est développé par **Yandex**. Son point fort unique : il gère les variables catégorielles **sans encodage préalable**. Vous lui donnez une colonne texte, il sait quoi faire.

### Dans notre cas

Dans notre dataset, les variables catégorielles (`operation`, `origin_account`, `destination_account`) sont déjà encodées numériquement dans notre feature engineering. CatBoost n'apporte donc pas son avantage principal ici.

```
✅ Très robuste aux overfitting (target encoding interne sécurisé)
✅ Gère nativement les catégorielles
✅ Souvent excellent sans tuning
⚠️ Plus lent que LightGBM
⚠️ Avantage principale non exploité ici (on a déjà encodé)
```

**Configuration recommandée** (si on l'utilise) :

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    class_weights=[1, 9],       # [poids_classe_0, poids_classe_1]
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)
```

**ROC-AUC attendu : 0.90 – 0.96**

---

## 6. 🧠 MLP (Réseau de Neurones) — Le Modern

### C'est quoi ?

Un MLP (Multi-Layer Perceptron) est un réseau de neurones simple avec des couches cachées. Chaque neurone fait une combinaison linéaire des entrées, puis applique une fonction non-linéaire (ReLU, etc.).

### Pour les données tabulaires

Les réseaux de neurones sont **moins efficaces** que les méthodes de boosting sur des données tabulaires structurées. C'est contre-intuitif, mais c'est prouvé par de nombreuses études (voir "Why do tree-based models still outperform deep learning on tabular data?" — Grinsztajn et al., 2022).

```
⚠️ Moins performant que XGBoost/LightGBM sur données tabulaires
⚠️ Besoin de normaliser toutes les features
⚠️ Long à entraîner et à tuner
✅ Peut être intéressant en ensemble (stacking)
```

### Quand l'utiliser ?

- Pour créer un **ensemble** (combiner plusieurs modèles)
- Si on veut explorer TabNet ou d'autres architectures spécialisées tabulaires
- Si le volume de données est beaucoup plus grand

---

## 7. ❌ SVM — À Éviter

Le SVM (Support Vector Machine) est techniquement très bon sur des petits datasets. Mais avec **1.3 millions de lignes**, l'entraînement prendrait des heures voir des jours. Il est exclu.

---

## 🏆 Stratégie Recommandée

### Étape 1 — Baseline (30 min)
Entraîner une régression logistique pour avoir un point de comparaison. Si le ROC-AUC est < 0.75, revoir les features.

### Étape 2 — Modèle principal (2-4h)
Entraîner LightGBM avec early stopping sur le split temporel. C'est le meilleur compromis vitesse/performance pour ce volume de données.

### Étape 3 — Modèle secondaire (2-4h)
Entraîner XGBoost avec les mêmes features. Comparer les performances.

### Étape 4 — Ensemble (optionnel, +1-2h)
Combiner les prédictions LightGBM + XGBoost + Random Forest :
```python
y_pred_final = (
    0.40 * y_pred_lgbm + 
    0.40 * y_pred_xgb + 
    0.20 * y_pred_rf
)
```
L'ensemble est presque toujours meilleur que n'importe quel modèle seul.

### Étape 5 — Optimisation (optionnel)
Utiliser Optuna pour chercher les meilleurs hyperparamètres automatiquement.

---

## 🔬 Hyperparamètres à Optimiser (par ordre de priorité)

### Pour LightGBM / XGBoost

| Paramètre | Valeurs à tester | Effet |
|---|---|---|
| `learning_rate` | 0.01, 0.05, 0.1 | Vitesse d'apprentissage |
| `num_leaves` / `max_depth` | 31, 63, 127 | Complexité du modèle |
| `n_estimators` | 500-2000 (+ early stopping) | Nombre d'arbres |
| `subsample` | 0.6, 0.8, 1.0 | Fraction de données par arbre |
| `colsample_bytree` | 0.6, 0.8, 1.0 | Fraction de features par arbre |
| `min_child_samples` | 10, 20, 50 | Anti-overfitting |

### Recherche automatique avec Optuna

```python
import optuna
from sklearn.metrics import roc_auc_score

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
    }
    model = lgb.LGBMClassifier(**params, n_estimators=500, is_unbalance=True)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 📋 Résumé des Recommandations

### Notre pick : **LightGBM comme modèle principal**

**Pourquoi ?**
1. **Le plus rapide** sur 1.3M lignes (5-10× plus rapide que XGBoost)
2. **Aussi précis** que XGBoost dans la plupart des cas
3. `is_unbalance=True` gère le déséquilibre sans calcul manuel
4. Early stopping automatique évite l'overfitting
5. Importance des features bien calculée

**Complément : XGBoost comme modèle secondaire** pour l'ensemble

**Baseline : Régression Logistique** pour valider les features

### Ce qu'on n'utilise PAS et pourquoi

| Modèle | Raison d'exclusion |
|---|---|
| SVM | Trop lent (1.3M lignes) |
| k-NN | Trop lent, pas adapté aux grands volumes |
| Naive Bayes | Hypothèses trop fortes (indépendance des features) |
| Deep Learning (CNN, RNN) | Données tabulaires → pas d'avantage |

---

*📁 Document produit dans le cadre du projet DataTour 2026 — Phase Nationale*  
*Auteur : Équipe Octave | Juin 2026*
