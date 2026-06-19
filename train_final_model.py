import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import time

print("Loading datasets...")
train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")

print(f"Train shape: {train_raw.shape}")
print(f"Test shape: {test_raw.shape}")

# Ensure sort by period for feature engineering
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

# Additional lag time differences for frequency tracking
combined['orig_time_diff_2'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(2)).fillna(999)
combined['dest_time_diff_2'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(2)).fillna(999)
combined['orig_time_diff_3'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(3)).fillna(999)
combined['dest_time_diff_3'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(3)).fillna(999)

# Chronological target encodings with optimal smoothing
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

# Interaction features for op_03 (the only operation category containing fraud)
combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

# Categoricals to category type
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

X_train = train_fe[features]
y_train = train_fe["fraud_flag"]
X_test = test_fe[features]

print("Preparing DMatrices for XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, enable_categorical=True)

seeds = [42, 123, 1337, 2026, 999]

# 1. Train 5-Seed XGBoost Ensemble
xgb_preds = []
for seed in seeds:
    print(f"Training XGBoost model with seed {seed}...")
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "tree_method": "hist"
    }
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=120)
    print("Predicting...")
    pred = xgb_model.predict(dtest)
    xgb_preds.append(pred)

mean_xgb_pred = np.mean(xgb_preds, axis=0)

# 2. Train 5-Seed LightGBM Ensemble
print("Preparing datasets for LightGBM...")
train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

lgb_preds = []
for seed in seeds:
    print(f"Training LightGBM model with seed {seed}...")
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
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1
    }
    lgb_model = lgb.train(lgb_params, train_dataset, num_boost_round=220)
    print("Predicting...")
    pred = lgb_model.predict(X_test)
    lgb_preds.append(pred)

mean_lgb_pred = np.mean(lgb_preds, axis=0)

# 3. Final Blend (90% XGBoost, 10% LightGBM)
print("Blending models...")
final_pred = 0.9 * mean_xgb_pred + 0.1 * mean_lgb_pred

# Map predictions back to match original test ID order (preventing any alignment issues)
print("Aligning predictions to test file order...")
pred_df = pd.DataFrame({
    'id': test_fe['id'],
    'target': final_pred
})

# Load the original test set to preserve exact row count and ID ordering
orig_test = pd.read_csv("dataset/test.csv")
submission = orig_test[['id']].merge(pred_df, on='id', how='left')

# 4. Post-processing: force target = 0.0 for all operations except op_03
print("Applying operation post-processing rule...")
is_not_op3 = (orig_test['operation'] != 'op_03')
submission.loc[is_not_op3, 'target'] = 0.0

# Verify no NaNs in target
assert submission['target'].isna().sum() == 0, "Error: Target contains NaN values!"

submission.to_csv("submission.csv", index=False)
print("Submission file successfully created at submission.csv!")
