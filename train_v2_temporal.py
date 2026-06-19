import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
print("Loading datasets...")
train_raw = pd.read_csv("dataset/train.csv")
test_raw = pd.read_csv("dataset/test.csv")
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

train_raw = train_raw.sort_values('period').reset_index(drop=True)
test_raw = test_raw.sort_values('period').reset_index(drop=True)

train_raw['is_test'] = 0
test_raw['is_test'] = 1
test_raw['fraud_flag'] = np.nan

combined = pd.concat([train_raw, test_raw], axis=0).sort_values('period').reset_index(drop=True)

# ============================================================
# CHRONOLOGICAL TARGET ENCODING (leak-free)
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

# Transaction index and cumulative amount
combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()
combined['orig_cum_amount'] = combined.groupby('origin_account')['amount'].cumsum() - combined['amount']
combined['dest_cum_amount'] = combined.groupby('destination_account')['amount'].cumsum() - combined['amount']

# Time diff lags (1, 2, 3)
for lag in [1, 2, 3]:
    suffix = '' if lag == 1 else f'_{lag}'
    combined[f'orig_time_diff{suffix}'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(lag)).fillna(999)
    combined[f'dest_time_diff{suffix}'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(lag)).fillna(999)

# ============================================================
# NEW: VELOCITY / BEHAVIORAL FEATURES
# ============================================================
print("Building velocity features...")

# --- Origin account rolling stats (per-period aggregations) ---
orig_period = combined.groupby(['origin_account', 'period']).agg(
    orig_period_tx_count=('amount', 'count'),
    orig_period_amount_sum=('amount', 'sum'),
    orig_period_amount_mean=('amount', 'mean'),
    orig_period_amount_max=('amount', 'max'),
).reset_index()

orig_period = orig_period.sort_values(['origin_account', 'period'])
# Cumulative stats up to previous period (leak-free)
orig_period['orig_cum_tx_count'] = orig_period.groupby('origin_account')['orig_period_tx_count'].cumsum()
orig_period['orig_cum_tx_count'] = orig_period.groupby('origin_account')['orig_cum_tx_count'].shift(1).fillna(0)
orig_period['orig_cum_amount_sum'] = orig_period.groupby('origin_account')['orig_period_amount_sum'].cumsum()
orig_period['orig_cum_amount_sum'] = orig_period.groupby('origin_account')['orig_cum_amount_sum'].shift(1).fillna(0)

# Lag the per-period stats (previous period's activity)
orig_period['orig_prev_tx_count'] = orig_period.groupby('origin_account')['orig_period_tx_count'].shift(1).fillna(0)
orig_period['orig_prev_amount_sum'] = orig_period.groupby('origin_account')['orig_period_amount_sum'].shift(1).fillna(0)
orig_period['orig_prev_amount_mean'] = orig_period.groupby('origin_account')['orig_period_amount_mean'].shift(1).fillna(0)

combined = combined.merge(
    orig_period[['origin_account', 'period', 'orig_cum_tx_count', 'orig_cum_amount_sum',
                 'orig_prev_tx_count', 'orig_prev_amount_sum', 'orig_prev_amount_mean']],
    on=['origin_account', 'period'], how='left'
)

# --- Destination account rolling stats ---
dest_period = combined.groupby(['destination_account', 'period']).agg(
    dest_period_tx_count=('amount', 'count'),
    dest_period_amount_sum=('amount', 'sum'),
    dest_period_amount_mean=('amount', 'mean'),
    dest_period_amount_max=('amount', 'max'),
).reset_index()

dest_period = dest_period.sort_values(['destination_account', 'period'])
dest_period['dest_cum_tx_count'] = dest_period.groupby('destination_account')['dest_period_tx_count'].cumsum()
dest_period['dest_cum_tx_count'] = dest_period.groupby('destination_account')['dest_cum_tx_count'].shift(1).fillna(0)
dest_period['dest_cum_amount_sum'] = dest_period.groupby('destination_account')['dest_period_amount_sum'].cumsum()
dest_period['dest_cum_amount_sum'] = dest_period.groupby('destination_account')['dest_cum_amount_sum'].shift(1).fillna(0)

dest_period['dest_prev_tx_count'] = dest_period.groupby('destination_account')['dest_period_tx_count'].shift(1).fillna(0)
dest_period['dest_prev_amount_sum'] = dest_period.groupby('destination_account')['dest_period_amount_sum'].shift(1).fillna(0)
dest_period['dest_prev_amount_mean'] = dest_period.groupby('destination_account')['dest_period_amount_mean'].shift(1).fillna(0)

combined = combined.merge(
    dest_period[['destination_account', 'period', 'dest_cum_tx_count', 'dest_cum_amount_sum',
                 'dest_prev_tx_count', 'dest_prev_amount_sum', 'dest_prev_amount_mean']],
    on=['destination_account', 'period'], how='left'
)

# --- Derived velocity features ---
# Amount deviation from account's average
combined['orig_avg_amount'] = (combined['orig_cum_amount_sum'] / (combined['orig_cum_tx_count'] + 1))
combined['dest_avg_amount'] = (combined['dest_cum_amount_sum'] / (combined['dest_cum_tx_count'] + 1))
combined['amount_vs_orig_avg'] = combined['amount'] / (combined['orig_avg_amount'] + eps)
combined['amount_vs_dest_avg'] = combined['amount'] / (combined['dest_avg_amount'] + eps)

# Number of unique destinations per origin (up to current period)
print("Computing unique pair counts...")
pair_counts = combined.groupby(['origin_account', 'period'])['destination_account'].nunique().reset_index()
pair_counts.columns = ['origin_account', 'period', 'orig_unique_dests_this_period']
pair_counts = pair_counts.sort_values(['origin_account', 'period'])
pair_counts['orig_cum_unique_dests'] = pair_counts.groupby('origin_account')['orig_unique_dests_this_period'].cumsum()
pair_counts['orig_cum_unique_dests'] = pair_counts.groupby('origin_account')['orig_cum_unique_dests'].shift(1).fillna(0)
combined = combined.merge(pair_counts[['origin_account', 'period', 'orig_cum_unique_dests']], on=['origin_account', 'period'], how='left')

# Balance ratio features
combined['origin_balance_ratio'] = combined['origin_balance_after'] / (combined['origin_balance_before'] + eps)
combined['dest_balance_ratio'] = combined['destination_balance_after'] / (combined['destination_balance_before'] + eps)

# ============================================================
# TARGET ENCODING
# ============================================================
print("Computing target encodings...")
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

# op_03 interaction features
combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

# Categoricals
combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')

# ============================================================
# SPLIT BACK
# ============================================================
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
    "is_op3", "op3_orig_no_change", "op3_dest_no_change",
    # NEW velocity features
    "orig_cum_tx_count", "orig_cum_amount_sum",
    "orig_prev_tx_count", "orig_prev_amount_sum", "orig_prev_amount_mean",
    "dest_cum_tx_count", "dest_cum_amount_sum",
    "dest_prev_tx_count", "dest_prev_amount_sum", "dest_prev_amount_mean",
    "orig_avg_amount", "dest_avg_amount",
    "amount_vs_orig_avg", "amount_vs_dest_avg",
    "orig_cum_unique_dests",
    "origin_balance_ratio", "dest_balance_ratio",
]

cat_features_lgb = ["operation", "origin_account", "destination_account"]
catboost_features = [f for f in features if f not in ["origin_account", "destination_account"]]

X_all = train_fe[features].reset_index(drop=True)
y_all = train_fe["fraud_flag"].reset_index(drop=True)
X_test = test_fe[features].reset_index(drop=True)

# ============================================================
# FILTER: TRAIN ONLY ON op_03 (all fraud is op_03)
# ============================================================
print("\nFiltering to op_03 only for training...")
is_op3_train = (X_all['operation'] == 'op_03').values
X_train = X_all[is_op3_train].reset_index(drop=True)
y_train = y_all[is_op3_train].reset_index(drop=True)
print(f"op_03 training set: {len(X_train)} rows, fraud rate: {y_train.mean():.4f}")

# ============================================================
# 5-FOLD STRATIFIED STACKING (on op_03 only)
# ============================================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cb  = np.zeros(len(X_train))

# For test: predict on ALL test rows, then post-process
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cb  = np.zeros(len(X_test))

print(f"\n=== Starting {n_splits}-Fold Stratified Stacking (op_03 only) ===")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1} / {n_splits} ---")
    t_fold = time.time()
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # --- XGBoost ---
    print(f"  XGBoost...")
    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_val, enable_categorical=True)
    dtest_xgb = xgb.DMatrix(X_test, enable_categorical=True)
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42 + fold,
        "tree_method": "hist"
    }
    
    xgb_model = xgb.train(xgb_params, dtr, num_boost_round=1500,
                          evals=[(dtr, 'train'), (xgb.DMatrix(X_val, label=y_val, enable_categorical=True), 'val')],
                          early_stopping_rounds=80, verbose_eval=0)
    oof_xgb[val_idx] = xgb_model.predict(dval_xgb, iteration_range=(0, xgb_model.best_iteration + 1))
    test_xgb += xgb_model.predict(dtest_xgb, iteration_range=(0, xgb_model.best_iteration + 1)) / n_splits
    print(f"    best_iteration={xgb_model.best_iteration}")
    
    # --- LightGBM ---
    print(f"  LightGBM...")
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features_lgb)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtr_lgb, categorical_feature=cat_features_lgb)
    
    lgb_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 30,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_fraction": 0.7,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1
    }
    
    lgb_model = lgb.train(lgb_params, dtr_lgb, num_boost_round=2000,
                          valid_sets=[dtr_lgb, dval_lgb],
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
    print(f"    best_iteration={lgb_model.best_iteration}")
    
    # --- CatBoost ---
    print(f"  CatBoost...")
    X_tr_cat = X_tr[catboost_features].copy()
    X_val_cat = X_val[catboost_features].copy()
    X_test_cat = X_test[catboost_features].copy()
    X_tr_cat['operation'] = X_tr_cat['operation'].astype(str)
    X_val_cat['operation'] = X_val_cat['operation'].astype(str)
    X_test_cat['operation'] = X_test_cat['operation'].astype(str)
    
    cb_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        eval_metric='Logloss',
        random_seed=42 + fold,
        verbose=0,
        task_type='CPU',
        early_stopping_rounds=80,
        use_best_model=True
    )
    cb_model.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), cat_features=["operation"])
    oof_cb[val_idx] = cb_model.predict_proba(X_val_cat)[:, 1]
    test_cb += cb_model.predict_proba(X_test_cat)[:, 1] / n_splits
    print(f"    best_iteration={cb_model.best_iteration_}")
    
    print(f"  Fold {fold+1} done in {time.time()-t_fold:.0f}s")

# ============================================================
# EVALUATE OOF
# ============================================================
print("\n=== Individual OOF Performances (op_03 only) ===")
xgb_ap = average_precision_score(y_train, oof_xgb)
lgb_ap = average_precision_score(y_train, oof_lgb)
cb_ap  = average_precision_score(y_train, oof_cb)
print(f"XGBoost  OOF PR-AUC: {xgb_ap:.6f}")
print(f"LightGBM OOF PR-AUC: {lgb_ap:.6f}")
print(f"CatBoost OOF PR-AUC: {cb_ap:.6f}")

# ============================================================
# OPTIMIZE BLEND WEIGHTS (grid search for robustness)
# ============================================================
print("\n=== Optimizing Blend Weights ===")

best_score = -1
best_w = None
for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        w3 = max(w3, 0)
        blend = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cb
        sc = average_precision_score(y_train, blend)
        if sc > best_score:
            best_score = sc
            best_w = (w1, w2, w3)

print(f"Best weights: XGB={best_w[0]:.2f}, LGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}")
print(f"Best blended OOF PR-AUC: {best_score:.6f}")

# Also try scipy for finer tuning
def loss_func(weights):
    w = np.abs(weights)
    w = w / np.sum(w)
    blend_oof = w[0] * oof_xgb + w[1] * oof_lgb + w[2] * oof_cb
    return -average_precision_score(y_train, blend_oof)

res = minimize(loss_func, x0=list(best_w), method='Nelder-Mead',
               options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
opt_w = np.abs(res.x) / np.sum(np.abs(res.x))
opt_score = -res.fun

if opt_score > best_score:
    best_w = tuple(opt_w)
    best_score = opt_score
    print(f"Scipy improved: XGB={best_w[0]:.4f}, LGB={best_w[1]:.4f}, CAT={best_w[2]:.4f}")
    print(f"Scipy blended OOF PR-AUC: {best_score:.6f}")

# ============================================================
# GENERATE FINAL SUBMISSION
# ============================================================
final_pred = best_w[0] * test_xgb + best_w[1] * test_lgb + best_w[2] * test_cb

print("\nAligning predictions to test file order...")
pred_df = pd.DataFrame({'id': test_fe['id'], 'target': final_pred})

orig_test = pd.read_csv("dataset/test.csv")
submission = orig_test[['id']].merge(pred_df, on='id', how='left')

# Post-processing: force 0 for non-op_03
print("Applying post-processing (non-op_03 = 0)...")
is_not_op3 = (orig_test['operation'] != 'op_03')
submission.loc[is_not_op3, 'target'] = 0.0

assert submission['target'].isna().sum() == 0, "NaN in target!"
assert len(submission) == len(orig_test), "Row count mismatch!"

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved to submission.csv ({len(submission)} rows)")
print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
