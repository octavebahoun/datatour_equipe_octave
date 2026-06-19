"""
Description: Inspection générale des données (colonnes, valeurs manquantes, chevauchements train/test).
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd
import numpy as np

train_path = "/home/precieux/datatour/dataset/train.csv"
test_path = "/home/precieux/datatour/dataset/test.csv"

print("Loading train...")
train = pd.read_csv(train_path)
print("Loading test...")
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\n--- Columns and Types ---")
print(train.dtypes)

print("\n--- Target distribution ---")
print(train['fraud_flag'].value_counts(dropna=False))
print(f"Fraud rate: {train['fraud_flag'].mean() * 100:.4f}%")

print("\n--- Missing values in Train ---")
print(train.isnull().sum())

print("\n--- Missing values in Test ---")
print(test.isnull().sum())

print("\n--- Cardinality of Categorical Columns ---")
for col in ['operation', 'origin_account', 'destination_account', 'period']:
    if col in train.columns:
        print(f"Column '{col}': {train[col].nunique()} unique values in train, {test[col].nunique()} in test")
        overlap = len(set(train[col]).intersection(set(test[col])))
        print(f"  Overlap train/test for '{col}': {overlap} unique values")
