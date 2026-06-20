"""
Description: Entraînement d'un modèle LightGBM amélioré intégrant les variables comportementales avancées.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import time

print("Loading data...")
train = pd.read_csv("dataset/train.csv")

# Sort by period to be safe
train = train.sort_values('period').reset_index(drop=True)

# Split train / validation
train_df = train[train['period'] <= 90].copy()
val_df = train[train['period'] > 90].copy()

print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

def build_features(train_part, val_part):
    # Add dummy label to val_part for concatenation if needed
    val_part = val_part.copy()
    train_part = train_part.copy()
    
    # Save original indices to split back later
    train_part['is_val'] = 0
    val_part['is_val'] = 1
    
    combined = pd.concat([train_part, val_part], axis=0).sort_values('period').reset_index(drop=True)
    
    print("Building basic features...")
    combined["amount_log1p"] = np.log1p(np.maximum(combined["amount"], 0))
    combined["origin_balance_change"] = combined["origin_balance_after"] - combined["origin_balance_before"]
    combined["destination_balance_change"] = combined["destination_balance_after"] - combined["destination_balance_before"]
    
    eps = 1e-6
    combined["amount_to_origin_before"] = combined["amount"] / (np.abs(combined["origin_balance_before"]) + eps)
    combined["amount_to_destination_before"] = combined["amount"] / (np.abs(combined["destination_balance_before"]) + eps)
    
    combined["origin_no_change"] = (combined["origin_balance_change"].abs() < 0.1).astype(int)
    combined["destination_no_change"] = (combined["destination_balance_change"].abs() < 0.1).astype(int)
    combined["amount_equals_origin_before"] = ((combined["amount"] - combined["origin_balance_before"]).abs() < 0.1).astype(int)
    
    print("Building sequence features...")
    combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
    combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()
    
    combined['orig_cum_amount'] = combined.groupby('origin_account')['amount'].cumsum() - combined['amount']
    combined['dest_cum_amount'] = combined.groupby('destination_account')['amount'].cumsum() - combined['amount']
    
    combined['orig_last_period'] = combined.groupby('origin_account')['period'].shift(1)
    combined['orig_time_diff'] = combined['period'] - combined['orig_last_period']
    combined['orig_time_diff'] = combined['orig_time_diff'].fillna(999)
    
    combined['dest_last_period'] = combined.groupby('destination_account')['period'].shift(1)
    combined['dest_time_diff'] = combined['period'] - combined['dest_last_period']
    combined['dest_time_diff'] = combined['dest_time_diff'].fillna(999)
    
    print("Building target encoding features...")
    global_mean = train_part['fraud_flag'].mean()
    smoothing = 10
    
    orig_te = train_part.groupby('origin_account')['fraud_flag'].agg(['sum', 'count'])
    orig_te['origin_te'] = (orig_te['sum'] + global_mean * smoothing) / (orig_te['count'] + smoothing)
    
    dest_te = train_part.groupby('destination_account')['fraud_flag'].agg(['sum', 'count'])
    dest_te['destination_te'] = (dest_te['sum'] + global_mean * smoothing) / (dest_te['count'] + smoothing)
    
    combined = combined.merge(orig_te[['origin_te']], on='origin_account', how='left')
    combined['origin_te'] = combined['origin_te'].fillna(global_mean)
    
    combined = combined.merge(dest_te[['destination_te']], on='destination_account', how='left')
    combined['destination_te'] = combined['destination_te'].fillna(global_mean)
    
    combined['operation'] = combined['operation'].astype('category')
    combined['origin_account'] = combined['origin_account'].astype('category')
    combined['destination_account'] = combined['destination_account'].astype('category')
    
    # Split back
    train_fe = combined[combined['is_val'] == 0].copy().drop(columns=['is_val'])
    val_fe = combined[combined['is_val'] == 1].copy().drop(columns=['is_val'])
    
    return train_fe, val_fe

train_fe, val_fe = build_features(train_df, val_df)

features = [
    "period", "operation", "amount", "amount_log1p",
    "origin_account", "origin_balance_before", "origin_balance_after", "origin_balance_change",
    "destination_account", "destination_balance_before", "destination_balance_after", "destination_balance_change",
    "amount_to_origin_before", "amount_to_destination_before",
    "origin_no_change", "destination_no_change", "amount_equals_origin_before",
    "orig_tx_idx", "dest_tx_idx", "orig_cum_amount", "dest_cum_amount",
    "orig_time_diff", "dest_time_diff", "origin_te", "destination_te"
]

cat_features = ["operation", "origin_account", "destination_account"]

X_train = train_fe[features]
y_train = train_fe["fraud_flag"]
X_val = val_fe[features]
y_val = val_fe["fraud_flag"]

print("Training LightGBM with advanced features...")
start_time = time.time()

train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=cat_features)

params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

model = lgb.train(
    params,
    train_dataset,
    num_boost_round=1000,
    valid_sets=[train_dataset, val_dataset],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

print(f"Training took {time.time() - start_time:.2f}s")

# Evaluate
print("Evaluating on validation set...")
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
val_pr_auc = average_precision_score(y_val, y_pred_val)
print(f"Validation PR-AUC (Average Precision): {val_pr_auc:.6f}")

# Print feature importances
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\n--- Feature Importances (Gain) ---")
print(importance.to_string(index=False))
