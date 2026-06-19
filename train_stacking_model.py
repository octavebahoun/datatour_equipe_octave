import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import time

print("Loading datasets...")
train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")

print(f"Train shape: {train_raw.shape}")
print(f"Test shape: {test_raw.shape}")

# Ensure sort by period for chronological feature engineering
train_raw = train_raw.sort_values('period').reset_index(drop=True)
test_raw = test_raw.sort_values('period').reset_index(drop=True)

# Combine for leak-free chronological feature engineering
train_raw['is_test'] = 0
test_raw['is_test'] = 1
test_raw['fraud_flag'] = np.nan

combined = pd.concat([train_raw, test_raw], axis=0).sort_values('period').reset_index(drop=True)

def compute_chronological_te(df, group_col, target_col, smoothing=10):
    print(f"Computing chronological target encoding for {group_col}...")
    period_stats = df.groupby([group_col, 'period'])[target_col].agg(['sum', 'count']).reset_index()
    period_stats = period_stats.sort_values([group_col, 'period'])
    period_stats['cum_sum'] = period_stats.groupby(group_col)['sum'].cumsum()
    period_stats['cum_count'] = period_stats.groupby(group_col)['count'].cumsum()
    period_stats['prev_cum_sum'] = period_stats.groupby(group_col)['cum_sum'].shift(1).fillna(0)
    period_stats['prev_cum_count'] = period_stats.groupby(group_col)['cum_count'].shift(1).fillna(0)
    
    global_period_stats = df.groupby('period')[target_col].agg(['sum', 'count']).reset_index()
    global_period_stats = global_period_stats.sort_values('period')
    global_period_stats['cum_sum'] = global_period_stats['sum'].cumsum()
    global_period_stats['cum_count'] = global_period_stats['count'].cumsum()
    global_period_stats['prev_cum_sum'] = global_period_stats['cum_sum'].shift(1).fillna(0)
    global_period_stats['prev_cum_count'] = global_period_stats['cum_count'].shift(1).fillna(0)
    
    global_period_stats['global_mean'] = (global_period_stats['prev_cum_sum'] + 1e-5) / (global_period_stats['prev_cum_count'] + 1e-5)
    period_stats = period_stats.merge(global_period_stats[['period', 'global_mean']], on='period', how='left')
    period_stats['te'] = (period_stats['prev_cum_sum'] + period_stats['global_mean'] * smoothing) / (period_stats['prev_cum_count'] + smoothing)
    df_merged = df.merge(period_stats[[group_col, 'period', 'te']], on=[group_col, 'period'], how='left')
    return df_merged['te']

print("Building features...")
combined["amount_log1p"] = np.log1p(np.maximum(combined["amount"], 0))
combined["origin_balance_change"] = combined["origin_balance_after"] - combined["origin_balance_before"]
combined["destination_balance_change"] = combined["destination_balance_after"] - combined["destination_balance_before"]

eps = 1e-6
combined["amount_to_origin_before"] = combined["amount"] / (np.abs(combined["origin_balance_before"]) + eps)
combined["amount_to_destination_before"] = combined["amount"] / (np.abs(combined["destination_balance_before"]) + eps)

combined["origin_no_change"] = (combined["origin_balance_change"].abs() < 0.1).astype(int)
combined["destination_no_change"] = (combined["destination_balance_change"].abs() < 0.1).astype(int)
combined["amount_equals_origin_before"] = ((combined["amount"] - combined["origin_balance_before"]).abs() < 0.1).astype(int)

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

# Multi-lag time differences (velocity features)
combined['orig_time_diff_2'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(2)).fillna(999)
combined['dest_time_diff_2'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(2)).fillna(999)
combined['orig_time_diff_3'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(3)).fillna(999)
combined['dest_time_diff_3'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(3)).fillna(999)

# Chronological target encodings with optimal smoothing
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

# Interaction features for op_03
combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

# Categoricals to category type (for XGBoost / LightGBM)
combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')

# Split back to train and test
train_fe = combined[combined['is_test'] == 0].copy().drop(columns=['is_test'])
test_fe = combined[combined['is_test'] == 1].copy().drop(columns=['is_test'])

features = [
    "period", "operation", "amount", "amount_log1p",
    "origin_account", "origin_balance_before", "origin_balance_after", "origin_balance_change",
    "destination_account", "destination_balance_before", "destination_balance_after", "destination_balance_change",
    "amount_to_origin_before", "amount_to_destination_before",
    "origin_no_change", "destination_no_change", "amount_equals_origin_before",
    "orig_tx_idx", "dest_tx_idx", "orig_cum_amount", "dest_cum_amount",
    "orig_time_diff", "dest_time_diff",
    "orig_time_diff_2", "dest_time_diff_2",
    "orig_time_diff_3", "dest_time_diff_3",
    "origin_te", "destination_te",
    "is_op3", "op3_orig_no_change", "op3_dest_no_change"
]

cat_features = ["operation", "origin_account", "destination_account"]

# High cardinality variables dropped for CatBoost to stay fast
catboost_features = [f for f in features if f not in ["origin_account", "destination_account"]]

X_train = train_fe[features].reset_index(drop=True)
y_train = train_fe["fraud_flag"].reset_index(drop=True)
X_test = test_fe[features].reset_index(drop=True)

# Define 5-Fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Out-of-fold predictions
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))

# Test predictions
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))

print("\n=== Starting 5-Fold Stratified Stacking ===")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1} / {n_splits} ---")
    
    # Train / Val Split
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # --- 1. XGBOOST ---
    print(f"Training XGBoost on Fold {fold + 1}...")
    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = xgb.DMatrix(X_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42 + fold,
        "tree_method": "hist"
    }
    
    xgb_model = xgb.train(xgb_params, dtr, num_boost_round=120)
    oof_xgb[val_idx] = xgb_model.predict(dval)
    test_xgb += xgb_model.predict(dtest) / n_splits
    
    # --- 2. LIGHTGBM ---
    print(f"Training LightGBM on Fold {fold + 1}...")
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    
    lgb_params = {
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
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1
    }
    
    lgb_model = lgb.train(lgb_params, dtr_lgb, num_boost_round=220)
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / n_splits
    
    # --- 3. CATBOOST ---
    print(f"Training CatBoost on Fold {fold + 1}...")
    X_tr_cat = X_tr[catboost_features].copy()
    X_val_cat = X_val[catboost_features].copy()
    X_test_cat = X_test[catboost_features].copy()
    
    X_tr_cat['operation'] = X_tr_cat['operation'].astype(str)
    X_val_cat['operation'] = X_val_cat['operation'].astype(str)
    X_test_cat['operation'] = X_test_cat['operation'].astype(str)
    
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=0,
        task_type='CPU'
    )
    
    cb_model.fit(X_tr_cat, y_tr, cat_features=["operation"])
    oof_cb[val_idx] = cb_model.predict_proba(X_val_cat)[:, 1]
    test_cb += cb_model.predict_proba(X_test_cat)[:, 1] / n_splits

# --- Print individual model performance ---
print("\n=== Individual Out-Of-Fold (OOF) Performances ===")
xgb_ap = average_precision_score(y_train, oof_xgb)
lgb_ap = average_precision_score(y_train, oof_lgb)
cb_ap = average_precision_score(y_train, oof_cb)
print(f"XGBoost OOF PR-AUC : {xgb_ap:.6f}")
print(f"LightGBM OOF PR-AUC: {lgb_ap:.6f}")
print(f"CatBoost OOF PR-AUC: {cb_ap:.6f}")

# --- 4. META-MODEL TRAINING (STACKING) ---
print("\n=== Training Stacking Meta-Model ===")
oof_preds = np.column_stack([oof_xgb, oof_lgb, oof_cb])
test_preds = np.column_stack([test_xgb, test_lgb, test_cb])

meta_model = LogisticRegression(random_state=42)
meta_model.fit(oof_preds, y_train)

# Evaluate Stacking performance on OOF
oof_stack = meta_model.predict_proba(oof_preds)[:, 1]
stacked_ap = average_precision_score(y_train, oof_stack)
print(f"Stacked Meta-Model OOF PR-AUC: {stacked_ap:.6f}")
print(f"Meta-Model coefficients: {meta_model.coef_[0]}")

# Make final predictions
final_pred = meta_model.predict_proba(test_preds)[:, 1]

# Map predictions back to match original test ID order (preventing any alignment issues)
print("\nAligning predictions to test file order...")
pred_df = pd.DataFrame({
    'id': test_fe['id'],
    'target': final_pred
})

# Load the original test set to preserve exact row count and ID ordering
orig_test = pd.read_csv("dataset/test.csv")
submission = orig_test[['id']].merge(pred_df, on='id', how='left')

# Post-processing: force target = 0.0 for all operations except op_03
print("Applying operation post-processing rule...")
is_not_op3 = (orig_test['operation'] != 'op_03')
submission.loc[is_not_op3, 'target'] = 0.0

# Verify no NaNs in target
assert submission['target'].isna().sum() == 0, "Error: Target contains NaN values!"

submission.to_csv("submission.csv", index=False)
print("Submission file successfully created at submission.csv!")
