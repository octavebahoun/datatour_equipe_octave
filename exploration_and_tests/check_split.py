"""
Description: Vérification des dimensions et de la répartition du découpage de validation temporelle (périodes <= 90 vs > 90).
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("dataset/train.csv")

print("Rows per period range:")
train_split = train[train['period'] <= 90]
val_split = train[train['period'] > 90]

print(f"Train (0-90): {len(train_split)} rows, {train_split['fraud_flag'].sum()} fraud ({train_split['fraud_flag'].mean()*100:.2f}%)")
print(f"Val (91-105): {len(val_split)} rows, {val_split['fraud_flag'].sum()} fraud ({val_split['fraud_flag'].mean()*100:.2f}%)")
