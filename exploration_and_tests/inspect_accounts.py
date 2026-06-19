"""
Description: Inspection de la cardinalité et du taux de recouvrement des comptes émetteurs et récepteurs entre train et test.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd
import numpy as np

train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")
test = pd.read_csv("/home/precieux/datatour/dataset/test.csv")

print("Origin account stats in Train:")
orig_counts = train['origin_account'].value_counts()
print(orig_counts.describe())

print("\nDestination account stats in Train:")
dest_counts = train['destination_account'].value_counts()
print(dest_counts.describe())

print("\nDo accounts span multiple periods?")
# Let's check a few accounts
sample_accounts = train['origin_account'].value_counts().index[:5]
for acc in sample_accounts:
    periods = train[train['origin_account'] == acc]['period'].unique()
    print(f"Account {acc}: {len(periods)} periods, min={min(periods)}, max={max(periods)}")

print("\nFraud rate by operation:")
print(train.groupby('operation')['fraud_flag'].agg(['count', 'mean']))

# Is there any relation between amount and balance?
# Typically: origin_balance_change = origin_balance_after - origin_balance_before
# Let's see if there's any discrepancies
print("\nDiscrepancies:")
train['diff_orig'] = train['origin_balance_after'] - train['origin_balance_before']
print("Orig balance change vs amount (mean, std of diff):")
print((train['diff_orig'] + train['amount']).describe())
