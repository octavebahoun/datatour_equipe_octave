# DataTour 2026 — Détection de Fraude Mobile Money

**Équipe Octave — Solution Finale**

## Contenu du dossier

| Fichier | Description |
|---------|-------------|
| `solution.py` | Script d'entraînement et de prédiction exécutable de bout en bout |
| `METHODOLOGIE.md` | Documentation complète de l'approche, features et modèles |
| `requirements.txt` | Dépendances Python nécessaires |
| `submission.csv` | Fichier de prédiction correspondant à la soumission retenue |
| `README.md` | Présent fichier d'instructions |

## Exécution

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer les données
# Les fichiers train.csv et test.csv doivent être dans ./dataset/
# (ou définir DATA_DIR=./chemin/vers/dataset)

# 3. Lancer la solution
python solution.py
```

Variables d'environnement optionnelles :
- `DATA_DIR` : chemin vers le dossier contenant `train.csv` et `test.csv` (défaut : `dataset`)
- `OUTPUT_DIR` : dossier de sortie pour `submission.csv` (défaut : `.`)

## Reproduction

La solution est déterministe (toutes les seeds fixées). Le fichier `submission.csv` généré doit reproduire le score de la soumission retenue dans la tolérance de 1e-6.

## Dépendances

- Python ≥ 3.9
- pandas, numpy, scikit-learn, scipy
- xgboost, lightgbm, catboost
- networkx
