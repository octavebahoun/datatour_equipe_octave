# DataTour 2026 - Soumission Phase Nationale

## 1. Méthodologie

### 1.1. Approche Globale
Notre solution aborde le problème de détection de fraude Mobile Money comme une tâche de **classification binaire**, optimisée spécifiquement pour la métrique **Average Precision (PR-AUC)**. L'architecture centrale repose sur un ensemble (Blend) de modèles basés sur les arbres de décision optimisés par descente de gradient (XGBoost, LightGBM, et CatBoost).

### 1.2. Ingénierie des Features (Feature Engineering)
Les caractéristiques (features) suivantes ont été créées pour capter les signaux de fraude, avec une attention particulière aux fuites de données (data leakage) :

*   **Règle d'exclusion métier** : L'EDA a prouvé que 100% de la fraude se trouvait dans l'opération `op_03`. Toutes les transactions hors `op_03` sont post-traitées à une probabilité de 0.0.
*   **Encodage de la Cible (Target Encoding) Chronologique** : La moyenne de fraude par compte est calculée de manière cumulative dans le temps (uniquement avec les données des périodes précédentes) pour éviter toute fuite temporelle.
*   **Signaux d'évolution des soldes** : Variables capturant si les soldes émetteur/destinataire changent ou non, ainsi que le ratio montant/solde.
*   **Features de Réseau et Graphe** : Calcul cumulatif des degrés d'entrée et de sortie (nombre de correspondants uniques par compte au fil du temps).
*   **Features de Vélocité** : Calcul du "Burstiness" ou vélocité du montant par rapport au temps écoulé depuis la dernière transaction d'un même compte.
*   **Features Catégorielles Haute Cardinalité** : Les identifiants de comptes (`origin_account`, `destination_account`) sont passés directement comme variables catégorielles natives dans XGBoost et LightGBM pour capter l'historique intrinsèque des comptes mules récidivistes.

### 1.3. Validation
Nous avons utilisé une validation croisée stratifiée à 5 plis (**5-Fold Stratified CV**). L'utilisation de cette méthode sur l'ensemble des données d'entraînement garantit que nos modèles généralisent bien et nous permet d'obtenir des prédictions (Out-Of-Fold) fiables pour l'optimisation des poids de l'ensemble.

### 1.4. Modèles et Hyperparamètres
L'ensemble est composé de trois algorithmes avec les hyperparamètres clés suivants (les *random seeds* sont fixées pour la reproductibilité) :
*   **XGBoost** : `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`, `enable_categorical=True`.
*   **LightGBM** : `learning_rate=0.05`, `num_leaves=31`, `max_depth=-1`, `min_data_in_leaf=20`, `bagging_fraction=0.8`.
*   **CatBoost** : `learning_rate=0.08`, `depth=6`, `iterations=1000`. CatBoost est entraîné sans les features de comptes (haute cardinalité) pour maintenir un temps d'inférence raisonnable.

### 1.5. Optimisation de l'Ensemble (Blending)
Les prédictions des trois modèles ne sont pas moyennées de façon naïve. Nous avons utilisé `scipy.optimize.minimize` (algorithme SLSQP) sur les prédictions Out-Of-Fold pour trouver la pondération exacte (ex: ~60% LGBM, ~40% XGBoost, ~0% CB) qui **maximise directement le score PR-AUC global**.

---

## 2. Instructions d'Exécution

### 2.1. Prérequis
Assurez-vous que Python 3.8 ou supérieur est installé. Installez les dépendances requises via la commande suivante :
```bash
pip install -r requirements.txt
```

### 2.2. Configuration des chemins de données
Le script `solution.py` cherche automatiquement les données dans un dossier nommé `dataset/` situé dans le même répertoire que le script. Vous devez y placer les fichiers suivants :
- `dataset/train.csv`
- `dataset/test.csv`

*(Note : Si le script est exécuté sur la plateforme Kaggle, il détectera automatiquement les chemins internes `/kaggle/input/datasets/octavebahoun/dataset/`)*.

### 2.3. Lancement de l'entraînement et de la prédiction
Pour exécuter la solution de bout en bout (chargement des données, ingénierie des features, entraînement des modèles en validation croisée, assemblage et prédiction), lancez :

```bash
python solution.py
```

### 2.4. Sortie (Output)
À la fin de l'exécution (qui prendra environ 15 à 30 minutes selon la machine), le script générera un fichier nommé de façon déterministe :
- `submission.csv`

Ce fichier est le fichier de prédiction correspondant à la soumission retenue dans le classement final.
