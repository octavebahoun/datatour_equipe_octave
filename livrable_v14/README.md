# DataTour 2026 - Phase Nationale
## Détection de Fraude Mobile Money - Soumission V14

Ce dossier contient la solution finale correspondant à la version 14 de notre approche.

### 1. Approche et Méthodologie

Notre solution repose sur une modélisation par **Ensemble Stacking (Niveau 2)** avec un soin particulier apporté à l'ingénierie des caractéristiques temporelles et relationnelles (graphes de transactions).

L'analyse exploratoire a révélé que la quasi-totalité des fraudes se produit lors des transactions de type `op_03`. Par conséquent, un post-traitement force toutes les prédictions des autres types d'opérations à 0.0, ce qui garantit un taux de faux positifs nul sur les opérations structurellement sûres.

### 2. Ingénierie des Caractéristiques (Features)

Les features ont été construites pour capturer le comportement des utilisateurs et la dynamique des montants :

*   **Âge et Historique** : Âge du compte expéditeur et destinataire depuis leur première apparition (`orig_account_age`, `dest_account_age`).
*   **Rank et Percentiles** : Suivi des montants maximums historiques pour repérer les anomalies et rang du montant (`orig_amount_rank`, `dest_amount_rank`).
*   **Dynamiques de relations (Edge Dynamics)** : Temps écoulé depuis la dernière transaction entre le même expéditeur et destinataire (`edge_time_diff`), détection de montants répétés, et encodages cibles (target encodings) sur les arêtes (edges).
*   **Target Encodings Chronologiques** : Calcul des taux de fraude historiques par expéditeur et par destinataire (lissé par moyennes mobiles) basés uniquement sur les transactions strictement antérieures.

### 3. Validation

La validation repose sur une **Validation Croisée Stratifiée K-Fold (5 plis)** sur l'ensemble d'entraînement. Cette méthode permet de conserver la proportion de fraudes dans chaque pli et assure une bonne corrélation entre notre évaluation locale (PR-AUC) et la performance sur le jeu de test.

### 4. Modèles et Hyperparamètres

L'architecture est un **Stacking** :

**Niveau 1 (Base Models)** : 3 modèles entraînés sur les features extraites.
*   **XGBoost** : `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`
*   **LightGBM** : `learning_rate=0.05`, `num_leaves=31`, `min_data_in_leaf=20`
*   **CatBoost** : `learning_rate=0.08`, `depth=6`

**Niveau 2 (Meta-Learner / Blending)** : 
Le script compare deux méthodes et sélectionne automatiquement la meilleure :
1. **Mélange Linéaire** : Optimisation des pondérations avec `scipy.optimize`.
2. **Stacking LightGBM** : Un méta-modèle LightGBM qui prend en entrée les prédictions (OOF) ainsi que quelques caractéristiques importantes du niveau 1 (comme le montant et les rangs) pour optimiser les prédictions.

### 5. Instructions d'Exécution

Pour reproduire la soumission exacte (`submission.csv`) :

1.  **Environnement** : Assurez-vous d'avoir Python 3.8+ et d'installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

2.  **Données** : Placez les fichiers de données fournis (`train.csv` et `test.csv`) dans le même répertoire que le script `solution.py`.

3.  **Exécution** : Lancez le script :
    ```bash
    python solution.py
    ```
    
    *(Le code chargera les données depuis le répertoire courant, construira les caractéristiques, entraînera les modèles, et générera le fichier `submission.csv`.)*

Toutes les graines (random seeds) sont fixées à `SEED = 42` pour garantir la reproductibilité.
