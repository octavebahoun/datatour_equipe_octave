"""
Description: Analyse spécifique des comptes destinataires de transactions frauduleuses.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")
frauds = train[train['fraud_flag'] == 1]
print(f"Total fraud cases: {len(frauds)}")
print(f"Unique destination accounts for fraud: {frauds['destination_account'].nunique()}")
print("Top 10 destination accounts for fraud:")
print(frauds['destination_account'].value_counts().head(10))

print("\nTop 10 origin accounts for fraud:")
print(frauds['origin_account'].value_counts().head(10))
