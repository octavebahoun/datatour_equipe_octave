"""
Description: Exploration générale des variables comportementales dans les jeux d'entraînement et de test.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd

train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")

print("Sum of fraud_flag by operation:")
print(train.groupby('operation')['fraud_flag'].sum())

# Let's inspect the math for each operation
for op in sorted(train['operation'].unique()):
    sub = train[train['operation'] == op]
    diff_orig = sub['origin_balance_after'] - sub['origin_balance_before']
    diff_dest = sub['destination_balance_after'] - sub['destination_balance_before']
    
    print(f"\n--- Operation: {op} ---")
    print(f"Sample size: {len(sub)}")
    print("Does origin balance change by -amount?")
    orig_eq = (sub['origin_balance_before'] - sub['amount'] - sub['origin_balance_after']).abs() < 0.1
    print(f"  True: {orig_eq.mean() * 100:.2f}%")
    
    print("Does origin balance remain unchanged?")
    orig_no_change = (sub['origin_balance_before'] - sub['origin_balance_after']).abs() < 0.1
    print(f"  True: {orig_no_change.mean() * 100:.2f}%")
    
    print("Does destination balance change by +amount?")
    dest_eq = (sub['destination_balance_before'] + sub['amount'] - sub['destination_balance_after']).abs() < 0.1
    print(f"  True: {dest_eq.mean() * 100:.2f}%")
    
    print("Does destination balance remain unchanged?")
    dest_no_change = (sub['destination_balance_before'] - sub['destination_balance_after']).abs() < 0.1
    print(f"  True: {dest_no_change.mean() * 100:.2f}%")
