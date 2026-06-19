"""
Description: Analyse de la relation entre les variations de solde (notamment l'absence de changement) et la fraude.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")
op3 = train[train['operation'] == 'op_03'].copy()

# Draining origin account
op3['origin_drained'] = ((op3['origin_balance_after'].abs() < 0.1) & (op3['origin_balance_before'] > 0.1)).astype(int)
print("Fraud rate by origin_drained:")
print(op3.groupby('origin_drained')['fraud_flag'].agg(['count', 'mean']))

# Destination was empty before
op3['destination_empty_before'] = (op3['destination_balance_before'].abs() < 0.1).astype(int)
print("\nFraud rate by destination_empty_before:")
print(op3.groupby('destination_empty_before')['fraud_flag'].agg(['count', 'mean']))

# Both origin drained and destination empty before
op3['both_indicators'] = op3['origin_drained'] & op3['destination_empty_before']
print("\nFraud rate by both indicators:")
print(op3.groupby('both_indicators')['fraud_flag'].agg(['count', 'mean']))
