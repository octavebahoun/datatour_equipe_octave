import pandas as pd
import numpy as np
import xgboost as xgb
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

def build_features(train_part, val_part):
    val_part = val_part.copy()
    train_part = train_part.copy()
    train_part['is_val'] = 0
    val_part['is_val'] = 1
    
    combined = pd.concat([train_part, val_part], axis=0).sort_values('period').reset_index(drop=True)
    
    # Basic features
    combined["amount_log1p"] = np.log1p(np.maximum(combined["amount"], 0))
    combined["origin_balance_change"] = combined["origin_balance_after"] - combined["origin_balance_before"]
    combined["destination_balance_change"] = combined["destination_balance_after"] - combined["destination_balance_before"]
    
    eps = 1e-6
    combined["amount_to_origin_before"] = combined["amount"] / (np.abs(combined["origin_balance_before"]) + eps)
    combined["amount_to_destination_before"] = combined["amount"] / (np.abs(combined["destination_balance_before"]) + eps)
    
    combined["origin_no_change"] = (combined["origin_balance_change"].abs() < 0.1).astype(int)
    combined["destination_no_change"] = (combined["destination_balance_change"].abs() < 0.1).astype(int)
    combined["amount_equals_origin_before"] = ((combined["amount"] - combined["origin_balance_before"]).abs() < 0.1).astype(int)
    
    # Tx index and cumulative amount
    combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
    combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()
    combined['orig_cum_amount'] = combined.groupby('origin_account')['amount'].cumsum() - combined['amount']
    combined['dest_cum_amount'] = combined.groupby('destination_account')['amount'].cumsum() - combined['amount']
    
    # Time diffs
    combined['orig_last_period'] = combined.groupby('origin_account')['period'].shift(1)
    combined['orig_time_diff'] = combined['period'] - combined['orig_last_period']
    combined['orig_time_diff'] = combined['orig_time_diff'].fillna(999)
    
    combined['dest_last_period'] = combined.groupby('destination_account')['period'].shift(1)
    combined['dest_time_diff'] = combined['period'] - combined['dest_last_period']
    combined['dest_time_diff'] = combined['dest_time_diff'].fillna(999)
    
    # Chronological TE
    combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
    combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)
    
    # Interaction features for op_03
    combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
    combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
    combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)
    
    # --- NEW BEHAVIORAL FEATURES ---
    
    # 1. Historical amount stats for origin
    print("Computing historical amount stats for origin...")
    combined['orig_cum_mean_amount'] = (combined.groupby('origin_account')['amount'].cumsum() - combined['amount']) / (combined['orig_tx_idx'] + eps)
    # Difference of current amount from historical mean
    combined['orig_amount_ratio'] = combined['amount'] / (combined['orig_cum_mean_amount'] + eps)
    
    # 2. Historical amount stats for destination
    print("Computing historical amount stats for destination...")
    combined['dest_cum_mean_amount'] = (combined.groupby('destination_account')['amount'].cumsum() - combined['amount']) / (combined['dest_tx_idx'] + eps)
    combined['dest_amount_ratio'] = combined['amount'] / (combined['dest_cum_mean_amount'] + eps)
    
    # 3. Transaction count of this account in the same period (frequency/burstiness)
    print("Computing intra-period transaction frequency...")
    combined['orig_tx_in_period'] = combined.groupby(['origin_account', 'period']).cumcount()
    combined['dest_tx_in_period'] = combined.groupby(['destination_account', 'period']).cumcount()
    
    # Total txs of the account in the previous period (lagged frequency)
    print("Computing lagged period transaction counts...")
    orig_period_counts = combined.groupby(['origin_account', 'period']).size().reset_index(name='orig_period_total')
    orig_period_counts['period'] = orig_period_counts['period'] + 1 # shift to represent previous period
    combined = combined.merge(orig_period_counts, on=['origin_account', 'period'], how='left')
    combined['orig_period_total'] = combined['orig_period_total'].fillna(0)
    
    dest_period_counts = combined.groupby(['destination_account', 'period']).size().reset_index(name='dest_period_total')
    dest_period_counts['period'] = dest_period_counts['period'] + 1
    combined = combined.merge(dest_period_counts, on=['destination_account', 'period'], how='left')
    combined['dest_period_total'] = combined['dest_period_total'].fillna(0)
    
    # 4. Target encoding of the pair of origin and destination accounts
    # Since cardinality is extremely high, we will do a joint TE with smoothing
    print("Computing joint origin-destination target encoding...")
    combined['orig_dest_pair'] = combined['origin_account'].astype(str) + "_" + combined['destination_account'].astype(str)
    combined['pair_te'] = compute_chronological_te(combined, 'orig_dest_pair', 'fraud_flag', smoothing=20)
    
    # Clean up categories
    combined['operation'] = combined['operation'].astype('category')
    combined['origin_account'] = combined['origin_account'].astype('category')
    combined['destination_account'] = combined['destination_account'].astype('category')
    
    train_fe = combined[combined['is_val'] == 0].copy().drop(columns=['is_val', 'orig_dest_pair'])
    val_fe = combined[combined['is_val'] == 1].copy().drop(columns=['is_val', 'orig_dest_pair'])
    
    return train_fe, val_fe

train_fe, val_fe = build_features(train_df, val_df)

features = [
    "period", "operation", "amount", "amount_log1p",
    "origin_account", "origin_balance_before", "origin_balance_after", "origin_balance_change",
    "destination_account", "destination_balance_before", "destination_balance_after", "destination_balance_change",
    "amount_to_origin_before", "amount_to_destination_before",
    "origin_no_change", "destination_no_change", "amount_equals_origin_before",
    "orig_tx_idx", "dest_tx_idx", "orig_cum_amount", "dest_cum_amount",
    "orig_time_diff", "dest_time_diff", "origin_te", "destination_te",
    "is_op3", "op3_orig_no_change", "op3_dest_no_change",
    
    # New features
    "orig_cum_mean_amount", "orig_amount_ratio",
    "dest_cum_mean_amount", "dest_amount_ratio",
    "orig_tx_in_period", "dest_tx_in_period",
    "orig_period_total", "dest_period_total",
    "pair_te"
]

X_train = train_fe[features]
y_train = train_fe["fraud_flag"]
X_val = val_fe[features]
y_val = val_fe["fraud_flag"]

print("Training XGBoost with new features...")
start_time = time.time()

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

evallist = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=50
)

print(f"Training took {time.time() - start_time:.2f}s")

# Evaluate
print("Evaluating on validation set...")
y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
val_pr_auc = average_precision_score(y_val, y_pred_val)
print(f"XGBoost Validation PR-AUC with new features: {val_pr_auc:.6f}")

# Force non-op_03 to 0
y_pred_val_forced = y_pred_val.copy()
is_not_op03 = (val_df['operation'] != 'op_03').values
y_pred_val_forced[is_not_op03] = 0.0
val_pr_auc_forced = average_precision_score(y_val, y_pred_val_forced)
print(f"XGBoost Validation PR-AUC (forced non-op_03 to 0): {val_pr_auc_forced:.6f}")
