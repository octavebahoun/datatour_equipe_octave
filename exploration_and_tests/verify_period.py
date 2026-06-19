"""
Description: Vérification des périodes représentées dans les données d'entraînement et de test.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")
test = pd.read_csv("/home/precieux/datatour/dataset/test.csv")

print("Train periods:")
print(f"Min: {train['period'].min()}, Max: {train['period'].max()}")
print(f"Unique values count: {train['period'].nunique()}")

print("\nTest periods:")
print(f"Min: {test['period'].min()}, Max: {test['period'].max()}")
print(f"Unique values count: {test['period'].nunique()}")

print("\nAre all train periods < test periods?")
print(train['period'].max() < test['period'].min())
