# DataTour 2026 - Phase Nationale
## Détection de Fraude Mobile Money - Soumission V22

Ce dossier contient la solution finale correspondant à la version 22 de notre approche.

### 1. Approche et Méthodologie

Notre solution repose sur une approche de modélisation par **Ensemble Stacking (Niveau 2)** avec un soin particulier apporté à l'ingénierie des caractéristiques temporelles, relationnelles et **de graphe (PageRank sans fuite / leak-free PageRank)**.

L'analyse exploratoire a révélé que la quasi-totalité des fraudes se produit lors des transactions de type `op_03`. Par conséquent, un post-traitement rigoureux force toutes les prédictions des autres types d'opérations à 0.0, garantissant un taux de faux positifs minimal sur les opérations sûres.

### 2. Ingénierie des Caractéristiques (Features) et Correction de Fuite (Leakage)

Les features ont été construites pour capturer le comportement des utilisateurs, la dynamique des montants, la fréquence temporelle et la centralité dans le réseau de transactions :

*   **Caractéristiques de base** : Âge du compte expéditeur (`orig_account_age`), ratio des soldes avant/après transaction, drapeaux de montants ronds et de retraits totaux.
*   **Dynamiques de relations (Edge Dynamics)** : Temps écoulé depuis la dernière transaction entre le même expéditeur et destinataire (`edge_time_diff`), détection de montants répétés sur la même relation (`is_repeated_amount_on_edge`), volume cumulé des transactions.
*   **Target Encodings Chronologiques** : Calcul des taux de fraude historiques (avec lissage) par expéditeur, par destinataire et par relation, basés uniquement sur les transactions strictement antérieures (pas de fuite temporelle).
*   **Rangs par Montant (Amount Rank)** : Rang relatif pour identifier si une transaction est exceptionnellement élevée par rapport à l'historique du compte (`orig_amount_rank`, `dest_amount_rank`), optimisé pour la gestion de la mémoire.
*   **PageRank Correct (Leak-Free)** : 
    *   *Ancienne version (V19)* : Le score PageRank était calculé sur un graphe contenant toutes les transactions cumulées (train et test). Cela introduisait une fuite de données (data leakage) puisque les relations futures du test set influaient sur les scores d'apprentissage.
    *   *Version 22 (Correcte)* : Le graphe orienté pondéré est construit **uniquement à partir des données d'entraînement (`train_raw`)**. Les scores PageRank sont calculés sur cette base saine, puis fusionnés avec le train et le test. Les comptes non vus durant la phase d'entraînement reçoivent un score de centralité par défaut de 0.0.

### 3. Validation

La stratégie de validation repose sur une **Validation Croisée Stratifiée K-Fold (5 plis)** (StratifiedKFold) sur l'ensemble de données d'entraînement. Cette approche garantit que la proportion de transactions frauduleuses est maintenue constante dans chaque pli, permettant une évaluation robuste de l'Average Precision (PR-AUC).

L'apprentissage est supervisé à chaque étape par une évaluation sur un ensemble de validation (early stopping).

### 4. Modèles et Hyperparamètres

L'architecture est un **Stacking** à deux niveaux :

**Niveau 1 (Base Models)** : 3 modèles entraînés sur les features extraites.
*   **XGBoost** : `iterations=400`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`
*   **LightGBM** : `iterations=600`, `learning_rate=0.05`, `num_leaves=31`, `min_data_in_leaf=20`
*   **CatBoost** : `iterations=1000`, `learning_rate=0.08`, `depth=6`

**Niveau 2 (Meta-Learner / Blending)** :
Le script évalue deux méthodes et conserve automatiquement la meilleure :
1.  **Mélange Linéaire (Scipy SLSQP)** : Optimisation des pondérations des 3 modèles de base.
2.  **Stacking LightGBM** : Un modèle de niveau 2 (`learning_rate=0.03`, `num_leaves=21`, `min_data_in_leaf=30`, `lambda_l1=1.0`, `lambda_l2=1.0`) qui prend en entrée les prédictions OOF ainsi qu'une sélection de caractéristiques clés du niveau 1, dont les scores PageRank corrigés.

### 5. Instructions d'Exécution

Pour reproduire la soumission exacte (`submission.csv`) :

1.  **Environnement** : Assurez-vous d'avoir Python 3.8+ et d'installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

2.  **Données** : Placez les fichiers de données fournis (`train.csv` et `test.csv`) dans le même dossier que le script `solution.py`.

3.  **Exécution** : Lancez simplement le script :
    ```bash
    python solution.py
    ```

Toutes les graines (random seeds) sont fixées à `SEED = 42` pour garantir la reproductibilité.
