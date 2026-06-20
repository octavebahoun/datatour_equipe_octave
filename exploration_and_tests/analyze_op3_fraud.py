"""
Description: Vérification de la distribution de la fraude par type d'opération, démontrant qu'elle ne concerne que 'op_03'.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("dataset/train.csv")
op3 = train[train['operation'] == 'op_03']

print("For op_03:")
print(f"Total transactions: {len(op3)}")
print(f"Fraud count: {op3['fraud_flag'].sum()}")
print(f"Fraud rate: {op3['fraud_flag'].mean() * 100:.2f}%")

op3 = op3.copy()
op3['orig_diff'] = op3['origin_balance_after'] - op3['origin_balance_before']
op3['orig_diff_eq_amount'] = (op3['orig_diff'] + op3['amount']).abs() < 0.1
op3['orig_no_change'] = (op3['orig_diff']).abs() < 0.1

print("\nFraud rate when origin balance changes by -amount:")
print(op3.groupby('orig_diff_eq_amount')['fraud_flag'].agg(['count', 'mean']))

print("\nFraud rate when origin balance remains unchanged:")
print(op3.groupby('orig_no_change')['fraud_flag'].agg(['count', 'mean']))

# Let's check destination balance
op3['dest_diff'] = op3['destination_balance_after'] - op3['destination_balance_before']
op3['dest_diff_eq_amount'] = (op3['dest_diff'] - op3['amount']).abs() < 0.1
op3['dest_no_change'] = (op3['dest_diff']).abs() < 0.1

print("\nFraud rate when destination balance changes by +amount:")
print(op3.groupby('dest_diff_eq_amount')['fraud_flag'].agg(['count', 'mean']))

print("\nFraud rate when destination balance remains unchanged:")
print(op3.groupby('dest_no_change')['fraud_flag'].agg(['count', 'mean']))
