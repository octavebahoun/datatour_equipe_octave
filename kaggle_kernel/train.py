import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import time, os, warnings, gc
warnings.filterwarnings('ignore')

t0 = time.time()
print("=== V8: Train on ALL data + High-Cardinality Categoricals + Advanced Features ===")

# ============================================================
# LOAD DATA
# ============================================================
if os.path.exists("/kaggle/input/datasets/octavebahoun/dataset"):
    train_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/train.csv")
    test_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
else:
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

train_raw = train_raw.sort_values('period').reset_index(drop=True)
test_raw = test_raw.sort_values('period').reset_index(drop=True)

train_raw['is_test'] = 0
test_raw['is_test'] = 1
test_raw['fraud_flag'] = np.nan

combined = pd.concat([train_raw, test_raw], axis=0).sort_values('period').reset_index(drop=True)
eps = 1e-6

# ============================================================
# CHRONOLOGICAL TARGET ENCODING (leak-free chronologically)
# ============================================================
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

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("Building features...")

combined["amount_log1p"] = np.log1p(np.maximum(combined["amount"], 0))
combined["origin_balance_change"] = combined["origin_balance_after"] - combined["origin_balance_before"]
combined["destination_balance_change"] = combined["destination_balance_after"] - combined["destination_balance_before"]
combined["amount_to_origin_before"] = combined["amount"] / (np.abs(combined["origin_balance_before"]) + eps)
combined["amount_to_destination_before"] = combined["amount"] / (np.abs(combined["destination_balance_before"]) + eps)
combined["origin_no_change"] = (combined["origin_balance_change"].abs() < 0.1).astype(int)
combined["destination_no_change"] = (combined["destination_balance_change"].abs() < 0.1).astype(int)
combined["amount_equals_origin_before"] = ((combined["amount"] - combined["origin_balance_before"]).abs() < 0.1).astype(int)

combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()
combined['orig_cum_amount'] = combined.groupby('origin_account')['amount'].cumsum() - combined['amount']
combined['dest_cum_amount'] = combined.groupby('destination_account')['amount'].cumsum() - combined['amount']

for lag in [1, 2, 3]:
    suffix = '' if lag == 1 else f'_{lag}'
    combined[f'orig_time_diff{suffix}'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(lag)).fillna(999)
    combined[f'dest_time_diff{suffix}'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(lag)).fillna(999)

# Origin per-period stats
orig_period = combined.groupby(['origin_account', 'period']).agg(
    orig_period_tx_count=('amount', 'count'),
    orig_period_amount_sum=('amount', 'sum'),
    orig_period_amount_mean=('amount', 'mean'),
).reset_index().sort_values(['origin_account', 'period'])
orig_period['orig_cum_tx_count'] = orig_period.groupby('origin_account')['orig_period_tx_count'].cumsum().groupby(orig_period['origin_account']).shift(1).fillna(0)
orig_period['orig_cum_amount_sum'] = orig_period.groupby('origin_account')['orig_period_amount_sum'].cumsum().groupby(orig_period['origin_account']).shift(1).fillna(0)
combined = combined.merge(orig_period[['origin_account', 'period', 'orig_cum_tx_count', 'orig_cum_amount_sum']], on=['origin_account', 'period'], how='left')

# Destination per-period stats
dest_period = combined.groupby(['destination_account', 'period']).agg(
    dest_period_tx_count=('amount', 'count'),
    dest_period_amount_sum=('amount', 'sum'),
    dest_period_amount_mean=('amount', 'mean'),
).reset_index().sort_values(['destination_account', 'period'])
dest_period['dest_cum_tx_count'] = dest_period.groupby('destination_account')['dest_period_tx_count'].cumsum().groupby(dest_period['destination_account']).shift(1).fillna(0)
dest_period['dest_cum_amount_sum'] = dest_period.groupby('destination_account')['dest_period_amount_sum'].cumsum().groupby(dest_period['destination_account']).shift(1).fillna(0)
combined = combined.merge(dest_period[['destination_account', 'period', 'dest_cum_tx_count', 'dest_cum_amount_sum']], on=['destination_account', 'period'], how='left')

combined['orig_avg_amount'] = combined['orig_cum_amount_sum'] / (combined['orig_cum_tx_count'] + 1)
combined['dest_avg_amount'] = combined['dest_cum_amount_sum'] / (combined['dest_cum_tx_count'] + 1)
combined['amount_vs_orig_avg'] = combined['amount'] / (combined['orig_avg_amount'] + eps)
combined['amount_vs_dest_avg'] = combined['amount'] / (combined['dest_avg_amount'] + eps)

# Network features
orig_dest_counts = combined.groupby(['origin_account', 'period'])['destination_account'].nunique().reset_index()
orig_dest_counts.columns = ['origin_account', 'period', 'orig_unique_dests_this_period']
orig_dest_counts = orig_dest_counts.sort_values(['origin_account', 'period'])
orig_dest_counts['orig_cum_unique_dests'] = orig_dest_counts.groupby('origin_account')['orig_unique_dests_this_period'].cumsum().groupby(orig_dest_counts['origin_account']).shift(1).fillna(0)
combined = combined.merge(orig_dest_counts[['origin_account', 'period', 'orig_cum_unique_dests']], on=['origin_account', 'period'], how='left')

dest_orig_counts = combined.groupby(['destination_account', 'period'])['origin_account'].nunique().reset_index()
dest_orig_counts.columns = ['destination_account', 'period', 'dest_unique_origins_this_period']
dest_orig_counts = dest_orig_counts.sort_values(['destination_account', 'period'])
dest_orig_counts['dest_cum_unique_origins'] = dest_orig_counts.groupby('destination_account')['dest_unique_origins_this_period'].cumsum().groupby(dest_orig_counts['destination_account']).shift(1).fillna(0)
combined = combined.merge(dest_orig_counts[['destination_account', 'period', 'dest_cum_unique_origins']], on=['destination_account', 'period'], how='left')

combined['origin_balance_ratio'] = combined['origin_balance_after'] / (combined['origin_balance_before'] + eps)
combined['dest_balance_ratio'] = combined['destination_balance_after'] / (combined['destination_balance_before'] + eps)

combined['amount_velocity_orig'] = combined['amount'] / (combined['orig_time_diff'] + eps)
combined['amount_velocity_dest'] = combined['amount'] / (combined['dest_time_diff'] + eps)

# Target Encodings
print("Computing target encodings...")
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')

# SPLIT BACK
train_fe = combined[combined['is_test'] == 0].copy().drop(columns=['is_test'])
test_fe = combined[combined['is_test'] == 1].copy().drop(columns=['is_test'])
del combined; gc.collect()

features = [
    "period", "operation", "origin_account", "destination_account", "amount", "amount_log1p",
    "origin_balance_before", "origin_balance_after", "origin_balance_change",
    "destination_balance_before", "destination_balance_after", "destination_balance_change",
    "amount_to_origin_before", "amount_to_destination_before",
    "origin_no_change", "destination_no_change", "amount_equals_origin_before",
    "orig_tx_idx", "dest_tx_idx", "orig_cum_amount", "dest_cum_amount",
    "orig_time_diff", "dest_time_diff",
    "orig_time_diff_2", "dest_time_diff_2",
    "orig_time_diff_3", "dest_time_diff_3",
    "origin_te", "destination_te",
    "is_op3", "op3_orig_no_change", "op3_dest_no_change",
    "orig_cum_tx_count", "orig_cum_amount_sum",
    "dest_cum_tx_count", "dest_cum_amount_sum",
    "orig_avg_amount", "dest_avg_amount",
    "amount_vs_orig_avg", "amount_vs_dest_avg",
    "orig_cum_unique_dests", "dest_cum_unique_origins",
    "origin_balance_ratio", "dest_balance_ratio",
    "amount_velocity_orig", "amount_velocity_dest",
]

cat_features = ["operation", "origin_account", "destination_account"]
# Drop high-cardinality features for CatBoost to avoid slowdown
cb_features = [f for f in features if f not in ["origin_account", "destination_account"]]

X_train = train_fe[features].reset_index(drop=True)
y_train = train_fe["fraud_flag"].reset_index(drop=True)
X_test = test_fe[features].reset_index(drop=True)

# We train on ALL DATA (no filtering to op_03) just like the successful 0.3537 script
print(f"Training set: {len(X_train)} rows, fraud rate: {y_train.mean():.4f}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cb  = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cb  = np.zeros(len(X_test))

print(f"\n=== Starting {n_splits}-Fold CV ===")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1} / {n_splits} ---")
    t_fold = time.time()
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # --- XGBoost ---
    print(f"  XGBoost...")
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest_xgb = xgb.DMatrix(X_test, enable_categorical=True)
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42 + fold,
        "tree_method": "hist",
        "verbosity": 0
    }
    
    xgb_model = xgb.train(
        xgb_params, dtr_xgb, num_boost_round=400,
        evals=[(dtr_xgb, 'train'), (dval_xgb, 'val')],
        early_stopping_rounds=50, verbose_eval=False
    )
    oof_xgb[val_idx] = xgb_model.predict(dval_xgb, iteration_range=(0, xgb_model.best_iteration + 1))
    test_xgb += xgb_model.predict(dtest_xgb, iteration_range=(0, xgb_model.best_iteration + 1)) / n_splits
    print(f"    best_iteration={xgb_model.best_iteration}")
    
    # --- LightGBM ---
    print(f"  LightGBM...")
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtr_lgb, categorical_feature=cat_features)
    
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
    
    lgb_model = lgb.train(
        lgb_params, dtr_lgb, num_boost_round=600,
        valid_sets=[dtr_lgb, dval_lgb],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
    print(f"    best_iteration={lgb_model.best_iteration}")
    
    # --- CatBoost ---
    print(f"  CatBoost...")
    X_tr_cb = X_tr[cb_features].copy()
    X_val_cb = X_val[cb_features].copy()
    X_test_cb = X_test[cb_features].copy()
    X_tr_cb['operation'] = X_tr_cb['operation'].astype(str)
    X_val_cb['operation'] = X_val_cb['operation'].astype(str)
    X_test_cb['operation'] = X_test_cb['operation'].astype(str)
    
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.08,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=0,
        task_type='CPU',
        early_stopping_rounds=50,
        use_best_model=True
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), cat_features=["operation"])
    oof_cb[val_idx] = cb_model.predict_proba(X_val_cb)[:, 1]
    test_cb += cb_model.predict_proba(X_test_cb)[:, 1] / n_splits
    print(f"    best_iteration={cb_model.best_iteration_}")
    
    print(f"  Fold {fold+1} done in {time.time()-t_fold:.0f}s")
    gc.collect()

print("\n=== Individual OOF Performances ===")
xgb_ap = average_precision_score(y_train, oof_xgb)
lgb_ap = average_precision_score(y_train, oof_lgb)
cb_ap  = average_precision_score(y_train, oof_cb)
print(f"XGBoost  OOF PR-AUC: {xgb_ap:.6f}")
print(f"LightGBM OOF PR-AUC: {lgb_ap:.6f}")
print(f"CatBoost OOF PR-AUC: {cb_ap:.6f}")

print("\n=== Optimizing Blend Weights ===")
def neg_ap(weights):
    w = weights / np.sum(weights)
    blend = w[0] * oof_xgb + w[1] * oof_lgb + w[2] * oof_cb
    return -average_precision_score(y_train, blend)

best_ap = -1
best_w = (0.33, 0.34, 0.33)
for w_xgb in np.arange(0.0, 1.05, 0.1):
    for w_lgb in np.arange(0.0, 1.05 - w_xgb, 0.1):
        w_cb = 1.0 - w_xgb - w_lgb
        if w_cb < -0.01: continue
        blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cb * oof_cb
        ap = average_precision_score(y_train, blend)
        if ap > best_ap:
            best_ap = ap
            best_w = (w_xgb, w_lgb, w_cb)

res = minimize(neg_ap, x0=list(best_w), method='SLSQP', bounds=[(0,1),(0,1),(0,1)], constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
opt_w = res.x / np.sum(res.x)
opt_ap = -res.fun
print(f"Scipy refined: XGB={opt_w[0]:.4f}, LGB={opt_w[1]:.4f}, CB={opt_w[2]:.4f} => PR-AUC={opt_ap:.6f}")

final_w = opt_w if opt_ap >= best_ap else np.array(best_w)
final_pred = final_w[0] * test_xgb + final_w[1] * test_lgb + final_w[2] * test_cb

print("\nAligning predictions to test file order...")
pred_df = pd.DataFrame({'id': test_fe['id'], 'target': final_pred})

if os.path.exists("/kaggle/input/datasets/octavebahoun/dataset"):
    orig_test = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
else:
    orig_test = pd.read_csv("dataset/test.csv")

submission = orig_test[['id']].merge(pred_df, on='id', how='left')

print("Applying post-processing (non-op_03 = 0)...")
is_not_op3 = (orig_test['operation'] != 'op_03')
submission.loc[is_not_op3, 'target'] = 0.0

assert submission['target'].isna().sum() == 0, "NaN in target!"
assert len(submission) == len(orig_test), "Row count mismatch!"

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved to submission.csv ({len(submission)} rows)")
print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
