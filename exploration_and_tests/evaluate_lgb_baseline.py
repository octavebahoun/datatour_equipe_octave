"""
Description: Entraînement et évaluation d'un modèle LightGBM de référence sur le jeu de validation.
Projet: Hackathon Détection de Fraude Mobile Money
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import time

print("Loading data...")
train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")

# Sort by period to be safe
train = train.sort_values('period').reset_index(drop=True)

# Split train / validation
train_df = train[train['period'] <= 90].copy()
val_df = train[train['period'] > 90].copy()

print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

def build_features(df):
    # Make a copy to avoid warnings
    df = df.copy()
    
    # 1. Log amount
    df["amount_log1p"] = np.log1p(np.maximum(df["amount"], 0))
    
    # 2. Balance changes
    df["origin_balance_change"] = df["origin_balance_after"] - df["origin_balance_before"]
    df["destination_balance_change"] = df["destination_balance_after"] - df["destination_balance_before"]
    
    # 3. Ratio features
    eps = 1e-6
    df["amount_to_origin_before"] = df["amount"] / (np.abs(df["origin_balance_before"]) + eps)
    df["amount_to_destination_before"] = df["amount"] / (np.abs(df["destination_balance_before"]) + eps)
    
    # 4. Indicators
    df["origin_no_change"] = (df["origin_balance_change"].abs() < 0.1).astype(int)
    df["destination_no_change"] = (df["destination_balance_change"].abs() < 0.1).astype(int)
    
    # Let's also check if the transaction amount is exactly equal to the balance before
    df["amount_equals_origin_before"] = ((df["amount"] - df["origin_balance_before"]).abs() < 0.1).astype(int)
    
    # 5. Type conversions for LightGBM categoricals
    df['operation'] = df['operation'].astype('category')
    df['origin_account'] = df['origin_account'].astype('category')
    df['destination_account'] = df['destination_account'].astype('category')
    
    return df

print("Building features...")
train_fe = build_features(train_df)
val_fe = build_features(val_df)

features = [
    "period", "operation", "amount", "amount_log1p",
    "origin_account", "origin_balance_before", "origin_balance_after", "origin_balance_change",
    "destination_account", "destination_balance_before", "destination_balance_after", "destination_balance_change",
    "amount_to_origin_before", "amount_to_destination_before",
    "origin_no_change", "destination_no_change", "amount_equals_origin_before"
]

cat_features = ["operation", "origin_account", "destination_account"]

X_train = train_fe[features]
y_train = train_fe["fraud_flag"]
X_val = val_fe[features]
y_val = val_fe["fraud_flag"]

print("Training LightGBM baseline...")
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

# We will train for 500 rounds with early stopping
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
