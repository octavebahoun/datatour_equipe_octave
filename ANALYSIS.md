# Analyse du Répertoire : Détection de Fraude Mobile Money

## 1. Contexte et Objectifs du Projet

Le projet porte sur la détection de transactions frauduleuses dans les services de mobile money, un secteur vital pour l'inclusion financière en Afrique. La mission consiste à concevoir un modèle de machine learning capable d'estimer la probabilité qu'une transaction soit frauduleuse.

**Objectifs clés :**
- Prédire une probabilité (entre 0 et 1) de fraude pour chaque transaction.
- Gérer le déséquilibre des classes (les transactions frauduleuses sont minoritaires).
- Assurer la robustesse du modèle face à des données transactionnelles anonymisées.

**Métrique d'évaluation :**
La métrique officielle est l'**Average Precision (PR-AUC)**. Cette métrique est préférée à l'Accuracy car elle est particulièrement adaptée aux jeux de données fortement déséquilibrés, récompensant la capacité du modèle à classer les transactions frauduleuses en haut de l'échelle de risque.

## 2. Synthèse de l'Analyse Exploratoire (EDA)

L'analyse exploratoire a révélé des points critiques pour la modélisation :

- **Concentration de la fraude :** La découverte la plus importante est que **100 % des fraudes se concentrent sur le type d'opération `op_03`**. Toutes les autres opérations ont un taux de fraude de 0 %.
- **Déséquilibre des classes :** Le taux de fraude global est d'environ 10 %. Au sein de `op_03`, il grimpe à environ 43 %.
- **Indicateurs de solde :** Les transactions où le solde de l'émetteur ne diminue pas (`origin_no_change`) ou le solde du destinataire n'augmente pas (`destination_no_change`) présentent des taux de fraude significativement plus élevés (jusqu'à 60 %).
- **Structure temporelle :** Les données sont organisées par périodes, avec une séparation stricte entre l'entraînement (périodes 0-90), la validation (91-105) et le test (106+).
- **Activité des comptes :** Certains comptes sont des émetteurs ou destinataires récurrents de fraudes, justifiant l'utilisation de l'historique des comptes.

## 3. Ingénierie des Caractéristiques (Feature Engineering)

Le répertoire met en œuvre des techniques avancées pour capturer les comportements frauduleux :

- **Target Encoding Chronologique :** Calculé sans fuite de données (leak-free) en utilisant uniquement les informations des périodes précédentes pour encoder les comptes (`origin_account`, `destination_account`).
- **Caractéristiques de Vélocité :** Statistiques glissantes sur les comptes, telles que le nombre de transactions, la somme des montants et le montant moyen au cours des périodes précédentes.
- **Différences Temporelles (Time Lags) :** Temps écoulé depuis la dernière, l'avant-dernière et la troisième dernière transaction pour chaque compte.
- **Ratios de Montant :** Comparaison du montant de la transaction actuelle avec le solde disponible ou avec la moyenne historique du compte.
- **Indicateurs Métier :** Variables binaires signalant l'absence de changement de solde, le vidage complet d'un compte, ou l'utilisation de l'opération `op_03`.
- **Interactions :** Croisement de variables, notamment l'interaction entre `op_03` et les indicateurs d'absence de changement de solde.

## 4. Comparaison des Approches de Modélisation

Trois stratégies principales ont été explorées dans les scripts de modélisation :

| Script | Approche | Points Forts |
|--------|----------|--------------|
| `train_final_model_with_catboost.py` | **Blending simple** | Utilise un ensemble de XGBoost (80%), LightGBM (10%) et CatBoost (10%). Chaque modèle est moyenné sur 5 graines (seeds). |
| `train_stacking_model.py` | **Stacking 5-Fold** | Utilise une régression logistique comme méta-modèle pour combiner les prédictions OOF (Out-Of-Fold) de XGB, LGB et CatBoost. |
| `train_v2_temporal.py` | **Optimisation temporelle** | Filtre l'entraînement uniquement sur `op_03`. Optimise les poids du mélange (XGB/LGB/Cat) pour maximiser directement la PR-AUC. Utilise des features de vélocité plus riches. |

**Post-traitement systématique :** Tous les scripts forcent la prédiction à 0.0 pour toute transaction n'appartenant pas à la catégorie `op_03`, ce qui booste significativement la performance finale.

## 5. Forces du Répertoire et Recommandations

### Forces
- **Rigueur méthodologique :** Utilisation systématique du Target Encoding chronologique pour éviter le surapprentissage.
- **Exploitation des découvertes EDA :** Focalisation sur `op_03` et application de règles de post-traitement intelligentes.
- **Diversité des modèles :** Combinaison de XGBoost, LightGBM et CatBoost pour capturer des signaux variés.
- **Validation robuste :** Respect strict de la chronologie pour les splits de validation.

### Recommandations et Pistes d'Amélioration
- **Feature Engineering Temporel Fin** : Ajouter des statistiques glissantes sur des fenêtres plus courtes (ex: 5 dernières transactions) pour détecter des rafales (bursts) de fraude.
- **Optimisation par Graphe** : Créer des variables mesurant la complexité du réseau de transactions autour d'un compte (ex: nombre de voisins distincts).
- **Fine-Tuning avec Optuna** : Automatiser la recherche d'hyperparamètres pour chaque modèle de l'ensemble afin de grappiller des points de PR-AUC.
- **Calibration des Modèles** : Utiliser Isotonic Regression pour s'assurer que les probabilités prédites sont bien calibrées, ce qui améliore la qualité du blending.
- **Analyse de Stabilité** : Vérifier que les features les plus importantes (Feature Importance) restent stables à travers le temps pour éviter la dégradation du modèle en production (drift).
