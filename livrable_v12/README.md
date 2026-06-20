# DataTour 2026 - Soumission Phase Nationale (V12 Edge Dynamics)

## 1. Méthodologie

### 1.1. Approche Globale
Notre solution aborde le problème de détection de fraude Mobile Money comme une tâche de classification binaire optimisée spécifiquement pour la métrique Average Precision (PR-AUC). Cette version "V12" (notre meilleure performance) introduit l'Analyse Dynamique de l'Arête ("Edge Dynamics").

### 1.2. Ingénierie des Features (Feature Engineering)
Les caractéristiques clés créées sont :
*   **Edge Dynamics** : Création de `edge_id` (`origin_account + destination_account`). Calcul exclusif pour cette arête :
    *   `edge_time_diff` : Temps écoulé depuis la dernière transaction entre ces deux mêmes comptes.
    *   `is_repeated_amount_on_edge` : Indique si le montant est strictement identique au précédent transfert sur cette même arête (fraude automatisée/mule).
    *   `edge_cum_tx_count` et `edge_cum_amount_sum` : Historique cumulé de la relation.
*   **Edge Target Encoding** : Moyenne chronologique de la fraude sur l'arête.
*   **Psychologie des montants** : `is_round_1000` et `is_round_5000`.
*   **Graphes Cumulatifs** : Calcul du nombre unique d'expéditeurs/destinataires interagissant avec un compte.
*   **Comptes Catégoriels** : `origin_account`, `destination_account` et `edge_id` (seulement pour LGBM pour éviter l'OOM sur XGBoost) gérés de façon native comme variables catégorielles.

### 1.3. Validation
La solution repose sur une Validation Croisée Stratifiée (5-Fold Stratified CV).

### 1.4. Modèles et Ensembling (Blend 3 Modèles)
L'ensemble est composé de 3 modèles de Gradient Boosting :
*   **XGBoost** 
*   **LightGBM**
*   **CatBoost**

Les modèles sont entraînés sur l'ensemble des plis. Une étape finale utilise `scipy.optimize.minimize` (SLSQP) pour trouver la combinaison linéaire optimale des 3 modèles sur les prédictions OOF afin de maximiser strictement le PR-AUC.

### 1.5. Exigences de Reproductibilité
Toutes les `random_seeds` sont fixées à `42`. Le script s'exécute de façon déterministe de bout en bout (Feature engineering -> Entraînement 5-Fold -> Scipy Blend -> Prédiction).

---

## 2. Instructions d'Exécution

### 2.1. Prérequis
Installez les bibliothèques requises :
```bash
pip install -r requirements.txt
```

### 2.2. Emplacement des Données
Les données doivent être dans le dossier `dataset/` (à côté du script) :
- `dataset/train.csv`
- `dataset/test.csv`

### 2.3. Exécution Complète
Lancez le script :
```bash
python solution.py
```
*Le script effectuera lui-même la création des features, l'entraînement de la validation croisée, l'optimisation des poids Scipy et l'inférence.*

### 2.4. Résultat
Un fichier `submission.csv` sera généré, prêt à être évalué.
