import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

print("Loading data...")
train = pd.read_csv("/home/precieux/datatour/dataset/train.csv")
train = train.sort_values('period').reset_index(drop=True)

train_df = train[train['period'] <= 90].copy()
val_df = train[train['period'] > 90].copy()

def compute_chronological_te(df, group_col, target_col, smoothing=10):
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

val_part = val_df.copy()
train_part = train_df.copy()
train_part['is_val'] = 0
val_part['is_val'] = 1

combined = pd.concat([train_part, val_part], axis=0).sort_values('period').reset_index(drop=True)

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

# Additional lag time differences
combined['orig_time_diff_2'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(2)).fillna(999)
combined['dest_time_diff_2'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(2)).fillna(999)
combined['orig_time_diff_3'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(3)).fillna(999)
combined['dest_time_diff_3'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(3)).fillna(999)

# Amount ratio features
combined['orig_mean_amount'] = combined['orig_cum_amount'] / (combined['orig_tx_idx'] + eps)
combined['dest_mean_amount'] = combined['dest_cum_amount'] / (combined['dest_tx_idx'] + eps)
combined['amount_to_orig_mean'] = combined['amount'] / (combined['orig_mean_amount'] + eps)
combined['amount_to_dest_mean'] = combined['amount'] / (combined['dest_mean_amount'] + eps)

combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')

train_fe = combined[combined['is_val'] == 0].copy().drop(columns=['is_val'])
val_fe = combined[combined['is_val'] == 1].copy().drop(columns=['is_val'])

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
    "amount_to_orig_mean", "amount_to_dest_mean",
    "origin_te", "destination_te",
    "is_op3", "op3_orig_no_change", "op3_dest_no_change"
]

X_train = train_fe[features]
y_train = train_fe["fraud_flag"]
X_val = val_fe[features]
y_val = val_fe["fraud_flag"]

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "tree_method": "hist"
}

model = xgb.train(params, dtrain, num_boost_round=118)
pred = model.predict(dval)
score = average_precision_score(y_val, pred)
print(f"PR-AUC with amount ratios: {score:.6f}")
