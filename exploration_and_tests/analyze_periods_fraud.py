"""
Description: Analyse de l'évolution temporelle du taux de fraude à travers les périodes.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("dataset/train.csv")
op3 = train[train['operation'] == 'op_03']

print("Fraud rate by period (first 10 periods):")
print(op3.groupby('period')['fraud_flag'].agg(['count', 'mean']).head(10))

print("\nFraud rate by period (last 10 periods):")
print(op3.groupby('period')['fraud_flag'].agg(['count', 'mean']).tail(10))
