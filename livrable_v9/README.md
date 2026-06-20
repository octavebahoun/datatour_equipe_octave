# DataTour 2026 - Soumission Phase Nationale (V9 Optuna)

## 1. Méthodologie

### 1.1. Approche Globale
Notre solution aborde le problème de détection de fraude Mobile Money comme une tâche de classification binaire optimisée spécifiquement pour la métrique Average Precision (PR-AUC). Cette version "V9" intègre une recherche d'hyperparamètres automatisée via Optuna et des variables dérivées avancées pour la détection des comptes de transit (mules).

### 1.2. Ingénierie des Features (Feature Engineering)
Les caractéristiques (features) suivantes ont été créées :
*   **Exclusion Métier** : Forçage à 0.0 de toutes les probabilités hors `op_03`.
*   **Mule Ratio (Pass-through accounts)** : Variables comparant le montant total reçu dans le passé par rapport au montant total envoyé dans le passé pour l'émetteur et le destinataire. Un ratio proche de 1 caractérise souvent un compte de transit (mule).
*   **Encodage Chronologique Strict** : Target encoding lissé et calculé dans le temps sans regard vers le futur.
*   **Graphes Cumulatifs** : Calcul du nombre unique d'expéditeurs/destinataires interagissant avec un compte donné au fil du temps.
*   **Comptes Catégoriels** : `origin_account` et `destination_account` sont utilisés de façon native comme variables catégorielles par LightGBM et XGBoost.

### 1.3. Validation et Optuna
La solution repose sur une Validation Croisée Stratifiée (5-Fold Stratified CV).
**Recherche d'Hyperparamètres** : Avant l'entraînement final, la bibliothèque `Optuna` est utilisée pour chercher mathématiquement les meilleurs paramètres (learning_rate, profondeur, régularisation L1/L2) de LightGBM afin de maximiser strictement l'AUC-PR.

### 1.4. Modèles et Ensembling
L'ensemble est composé de :
*   **LightGBM** (Avec les meilleurs hyperparamètres trouvés par Optuna de façon dynamique à l'exécution).
*   **XGBoost** (Avec des paramètres fixes testés robustes).

Les deux modèles sont entraînés sur l'ensemble des plis. Une étape finale utilise `scipy.optimize.minimize` (SLSQP) pour trouver la combinaison linéaire optimale des deux modèles sur les prédictions Out-Of-Fold.

### 1.5. Exigences de Reproductibilité
Toutes les `random_seeds` (KFold, Optuna, LightGBM, XGBoost) sont fixées à `42`. Le script s'exécute de façon déterministe de bout en bout.

---

## 2. Instructions d'Exécution

### 2.1. Prérequis
Installez les bibliothèques requises :
```bash
pip install -r requirements.txt
pip install optuna
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
*Le script effectuera lui-même le tuning Optuna (15 essais), l'entraînement de la validation croisée, l'optimisation des poids Scipy et l'inférence.*

### 2.4. Résultat
Un fichier `submission.csv` sera généré, prêt à être évalué.
