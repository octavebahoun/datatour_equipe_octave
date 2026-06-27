import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import time, os, warnings, gc
import networkx as nx
warnings.filterwarnings('ignore')

t0 = time.time()
print("=== V20: Base V19 + HITS (Hub/Authority) ===")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
if os.path.exists("/kaggle/input/datasets/octavebahoun/dataset"):
    train_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/train.csv")
    test_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
else:
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")

train_raw = train_raw.sort_values('period').reset_index(drop=True)
test_raw = test_raw.sort_values('period').reset_index(drop=True)

train_raw['is_test'] = 0
test_raw['is_test'] = 1
test_raw['fraud_flag'] = np.nan

combined = pd.concat([train_raw, test_raw], axis=0).sort_values('period').reset_index(drop=True)
eps = 1e-6

# ============================================================
# INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING)
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

print("Construction des caractéristiques de base...")
combined["amount_log1p"] = np.log1p(np.maximum(combined["amount"], 0))
combined["origin_balance_change"] = combined["origin_balance_after"] - combined["origin_balance_before"]
combined["destination_balance_change"] = combined["destination_balance_after"] - combined["destination_balance_before"]
combined["amount_to_origin_before"] = combined["amount"] / (np.abs(combined["origin_balance_before"]) + eps)
combined["amount_to_destination_before"] = combined["amount"] / (np.abs(combined["destination_balance_before"]) + eps)
combined["origin_no_change"] = (combined["origin_balance_change"].abs() < 0.1).astype(int)
combined["destination_no_change"] = (combined["destination_balance_change"].abs() < 0.1).astype(int)
combined["amount_equals_origin_before"] = ((combined["amount"] - combined["origin_balance_before"]).abs() < 0.1).astype(int)

combined['edge_id'] = combined['origin_account'].astype(str) + "_" + combined['destination_account'].astype(str)
combined['is_round_1000'] = (combined['amount'] % 1000 == 0).astype(int)
combined['is_round_5000'] = (combined['amount'] % 5000 == 0).astype(int)

print("Calcul des dynamiques de relations (Edge Dynamics)...")
combined['edge_time_diff'] = (combined['period'] - combined.groupby('edge_id')['period'].shift(1)).fillna(999)
combined['prev_edge_amount'] = combined.groupby('edge_id')['amount'].shift(1)
combined['is_repeated_amount_on_edge'] = (combined['amount'] == combined['prev_edge_amount']).astype(int)
combined.drop(columns=['prev_edge_amount'], inplace=True)
combined['edge_cum_tx_count'] = combined.groupby('edge_id').cumcount()
combined['edge_cum_amount_sum'] = combined.groupby('edge_id')['amount'].cumsum() - combined['amount']

print("Calcul du PageRank sur le graphe des transactions...")
edges_weights = combined.groupby(['origin_account', 'destination_account'])['amount'].count().reset_index()
edges_weights.rename(columns={'amount': 'weight'}, inplace=True)
G = nx.from_pandas_edgelist(edges_weights, 'origin_account', 'destination_account', ['weight'], create_using=nx.DiGraph())
pagerank_scores = nx.pagerank(G, weight='weight')
pagerank_df = pd.DataFrame(list(pagerank_scores.items()), columns=['account', 'pagerank'])
combined = combined.merge(pagerank_df.rename(columns={'account': 'origin_account', 'pagerank': 'orig_pagerank'}), on='origin_account', how='left')
combined = combined.merge(pagerank_df.rename(columns={'account': 'destination_account', 'pagerank': 'dest_pagerank'}), on='destination_account', how='left')
combined['orig_pagerank'] = combined['orig_pagerank'].fillna(0)
combined['dest_pagerank'] = combined['dest_pagerank'].fillna(0)

print("Calcul des scores HITS (Hub/Authority)...")
hits_hubs, hits_authorities = nx.hits(G, max_iter=100, normalized=True)
hits_hub_df = pd.DataFrame(list(hits_hubs.items()), columns=['account', 'hits_hub'])
hits_auth_df = pd.DataFrame(list(hits_authorities.items()), columns=['account', 'hits_auth'])
combined = combined.merge(hits_hub_df.rename(columns={'account': 'origin_account', 'hits_hub': 'orig_hits_hub'}), on='origin_account', how='left')
combined = combined.merge(hits_hub_df.rename(columns={'account': 'destination_account', 'hits_hub': 'dest_hits_hub'}), on='destination_account', how='left')
combined = combined.merge(hits_auth_df.rename(columns={'account': 'origin_account', 'hits_auth': 'orig_hits_auth'}), on='origin_account', how='left')
combined = combined.merge(hits_auth_df.rename(columns={'account': 'destination_account', 'hits_auth': 'dest_hits_auth'}), on='destination_account', how='left')
combined['orig_hits_hub'] = combined['orig_hits_hub'].fillna(0)
combined['dest_hits_hub'] = combined['dest_hits_hub'].fillna(0)
combined['orig_hits_auth'] = combined['orig_hits_auth'].fillna(0)
combined['dest_hits_auth'] = combined['dest_hits_auth'].fillna(0)
del G, edges_weights, pagerank_scores, pagerank_df, hits_hubs, hits_authorities, hits_hub_df, hits_auth_df; gc.collect()


print("Calcul des caractéristiques de Rank (Optimisées OOM)...")
orig_first_seen = combined.groupby('origin_account')['period'].transform('first')
dest_first_seen = combined.groupby('destination_account')['period'].transform('first')
combined['orig_account_age'] = combined['period'] - orig_first_seen
combined['dest_account_age'] = combined['period'] - dest_first_seen

# Optimisation OOM : shift() puis cummax()
combined['orig_max_amount_so_far'] = combined.groupby('origin_account')['amount'].shift(1)
combined['orig_max_amount_so_far'] = combined.groupby('origin_account')['orig_max_amount_so_far'].cummax().fillna(0)
combined['orig_amount_rank'] = combined['amount'] / (combined['orig_max_amount_so_far'] + eps)

combined['dest_max_amount_so_far'] = combined.groupby('destination_account')['amount'].shift(1)
combined['dest_max_amount_so_far'] = combined.groupby('destination_account')['dest_max_amount_so_far'].cummax().fillna(0)
combined['dest_amount_rank'] = combined['amount'] / (combined['dest_max_amount_so_far'] + eps)

print("Calcul des caractéristiques de base (V14)...")
combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()
combined['orig_cum_amount'] = combined.groupby('origin_account')['amount'].cumsum() - combined['amount']
combined['dest_cum_amount'] = combined.groupby('destination_account')['amount'].cumsum() - combined['amount']

for lag in [1, 2, 3]:
    suffix = '' if lag == 1 else f'_{lag}'
    combined[f'orig_time_diff{suffix}'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(lag)).fillna(999)
    combined[f'dest_time_diff{suffix}'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(lag)).fillna(999)

orig_period = combined.groupby(['origin_account', 'period']).agg(
    orig_period_tx_count=('amount', 'count'), orig_period_amount_sum=('amount', 'sum')
).reset_index().sort_values(['origin_account', 'period'])
orig_period['orig_cum_tx_count'] = orig_period.groupby('origin_account')['orig_period_tx_count'].cumsum().groupby(orig_period['origin_account']).shift(1).fillna(0)
orig_period['orig_cum_amount_sum'] = orig_period.groupby('origin_account')['orig_period_amount_sum'].cumsum().groupby(orig_period['origin_account']).shift(1).fillna(0)
combined = combined.merge(orig_period[['origin_account', 'period', 'orig_cum_tx_count', 'orig_cum_amount_sum']], on=['origin_account', 'period'], how='left')

dest_period = combined.groupby(['destination_account', 'period']).agg(
    dest_period_tx_count=('amount', 'count'), dest_period_amount_sum=('amount', 'sum')
).reset_index().sort_values(['destination_account', 'period'])
dest_period['dest_cum_tx_count'] = dest_period.groupby('destination_account')['dest_period_tx_count'].cumsum().groupby(dest_period['destination_account']).shift(1).fillna(0)
dest_period['dest_cum_amount_sum'] = dest_period.groupby('destination_account')['dest_period_amount_sum'].cumsum().groupby(dest_period['destination_account']).shift(1).fillna(0)
combined = combined.merge(dest_period[['destination_account', 'period', 'dest_cum_tx_count', 'dest_cum_amount_sum']], on=['destination_account', 'period'], how='left')

combined['orig_avg_amount'] = combined['orig_cum_amount_sum'] / (combined['orig_cum_tx_count'] + 1)
combined['dest_avg_amount'] = combined['dest_cum_amount_sum'] / (combined['dest_cum_tx_count'] + 1)
combined['amount_vs_orig_avg'] = combined['amount'] / (combined['orig_avg_amount'] + eps)
combined['amount_vs_dest_avg'] = combined['amount'] / (combined['dest_avg_amount'] + eps)

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

print("Calcul des target encodings chronologiques...")
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)
combined['edge_te'] = compute_chronological_te(combined, 'edge_id', 'fraud_flag', smoothing=5)

combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')
combined['edge_id'] = combined['edge_id'].astype('category')

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
    "orig_time_diff", "dest_time_diff", "orig_time_diff_2", "dest_time_diff_2", "orig_time_diff_3", "dest_time_diff_3",
    "origin_te", "destination_te", "is_op3", "op3_orig_no_change", "op3_dest_no_change",
    "orig_cum_tx_count", "orig_cum_amount_sum", "dest_cum_tx_count", "dest_cum_amount_sum",
    "orig_avg_amount", "dest_avg_amount", "amount_vs_orig_avg", "amount_vs_dest_avg",
    "orig_cum_unique_dests", "dest_cum_unique_origins",
    "origin_balance_ratio", "dest_balance_ratio",
    "amount_velocity_orig", "amount_velocity_dest",
    "edge_id", "is_round_1000", "is_round_5000", "edge_te", 
    "edge_time_diff", "is_repeated_amount_on_edge", "edge_cum_tx_count", "edge_cum_amount_sum",
    "orig_account_age", "dest_account_age", "orig_amount_rank", "dest_amount_rank",
    "orig_pagerank", "dest_pagerank",
    "orig_hits_hub", "dest_hits_hub", "orig_hits_auth", "dest_hits_auth"
]

cat_features = ["operation", "origin_account", "destination_account", "edge_id"]
xgb_features = [f for f in features if f != "edge_id"]
cb_features = [f for f in features if f not in ["origin_account", "destination_account", "edge_id"]]

# ============================================================
# ENTRAÎNEMENT DES MODÈLES DE NIVEAU 1
# ============================================================
print("\n" + "="*50)
print("ENTRAÎNEMENT DES MODÈLES DE NIVEAU 1")
print("="*50)

X_train = train_fe[features].reset_index(drop=True)
y_train = train_fe["fraud_flag"].reset_index(drop=True)
X_test = test_fe[features].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb, oof_lgb, oof_cb = np.zeros(len(X_train)), np.zeros(len(X_train)), np.zeros(len(X_train))
test_xgb, test_lgb, test_cb = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Plis (Fold) {fold + 1} ---")
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # 1. XGBoost
    dtr_xgb = xgb.DMatrix(X_tr[xgb_features], label=y_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_val[xgb_features], label=y_val, enable_categorical=True)
    xgb_m = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "aucpr", "learning_rate": 0.08, "max_depth": 6, 
         "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42+fold, "tree_method": "hist", "verbosity": 0},
        dtr_xgb, 400, evals=[(dtr_xgb, 'train'), (dval_xgb, 'val')], early_stopping_rounds=50, verbose_eval=False
    )
    oof_xgb[val_idx] = xgb_m.predict(dval_xgb, iteration_range=(0, xgb_m.best_iteration + 1))
    test_xgb += xgb_m.predict(xgb.DMatrix(X_test[xgb_features], enable_categorical=True), iteration_range=(0, xgb_m.best_iteration + 1)) / 5
    
    # 2. LightGBM
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtr_lgb, categorical_feature=cat_features)
    lgb_m = lgb.train(
        {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 31, 
         "max_depth": -1, "min_data_in_leaf": 20, "random_state": 42+fold, "n_jobs": -1, "verbose": -1},
        dtr_lgb, 600, valid_sets=[dtr_lgb, dval_lgb], callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = lgb_m.predict(X_val, num_iteration=lgb_m.best_iteration)
    test_lgb += lgb_m.predict(X_test, num_iteration=lgb_m.best_iteration) / 5
    
    # 3. CatBoost
    cb_m = CatBoostClassifier(iterations=1000, learning_rate=0.08, depth=6, eval_metric='AUC', random_seed=42+fold, verbose=0, early_stopping_rounds=50)
    cb_m.fit(X_tr[cb_features], y_tr, eval_set=(X_val[cb_features], y_val), cat_features=["operation"])
    oof_cb[val_idx] = cb_m.predict_proba(X_val[cb_features])[:, 1]
    test_cb += cb_m.predict_proba(X_test[cb_features])[:, 1] / 5

# ============================================================
# COMBINAISON (LEVEL-2) : MÉLANGE LINÉAIRE SCIPY VS STACKING LGB
# ============================================================
print("\n" + "="*50)
print("NIVEAU 2 : MÉLANGE LINÉAIRE SCIPY VS ENHANCED STACKING LGB")
print("="*50)

# --- Méthode 1 : Mélange linéaire (Blending) optimisé par Scipy ---
def neg_ap(weights):
    w = weights / np.sum(weights)
    blend = w[0] * oof_xgb + w[1] * oof_lgb + w[2] * oof_cb
    return -average_precision_score(y_train, blend)

res = minimize(neg_ap, x0=[0.33, 0.34, 0.33], method='SLSQP', bounds=[(0,1),(0,1),(0,1)], constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
w1 = res.x / np.sum(res.x)
scipy_oof_score = -res.fun
scipy_pred = w1[0]*test_xgb + w1[1]*test_lgb + w1[2]*test_cb
print(f"[Scipy]  Poids : XGB={w1[0]:.3f}, LGB={w1[1]:.3f}, CB={w1[2]:.3f} => OOF PR-AUC={scipy_oof_score:.5f}")

# --- Méthode 2 : Méta-Modèle LightGBM (Enhanced Stacking V17) ---
stack_train = pd.DataFrame({'xgb': oof_xgb, 'lgb': oof_lgb, 'cb': oof_cb})
stack_test = pd.DataFrame({'xgb': test_xgb, 'lgb': test_lgb, 'cb': test_cb})

# On injecte les features importantes de niveau 1 (sans Burst Detection)
stack_key_feats = [
    'amount', 'is_op3', 'origin_te', 'destination_te', 'edge_te',
    'orig_account_age', 'dest_account_age', 'orig_amount_rank', 'dest_amount_rank',
    'edge_time_diff', 'edge_cum_tx_count', 'amount_velocity_orig', 'orig_time_diff',
    'orig_pagerank', 'dest_pagerank',
    'orig_hits_hub', 'dest_hits_hub', 'orig_hits_auth', 'dest_hits_auth'
]
for f in stack_key_feats:
    stack_train[f] = X_train[f].values
    stack_test[f] = X_test[f].values

oof_stack = np.zeros(len(stack_train))
test_stack = np.zeros(len(stack_test))

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
for fold, (tr_idx, va_idx) in enumerate(skf2.split(stack_train, y_train)):
    dtr = lgb.Dataset(stack_train.iloc[tr_idx], label=y_train.iloc[tr_idx])
    dva = lgb.Dataset(stack_train.iloc[va_idx], label=y_train.iloc[va_idx], reference=dtr)
    meta_m = lgb.train(
        {"objective": "binary", "metric": "average_precision", "learning_rate": 0.03,
         "num_leaves": 21, "min_data_in_leaf": 30, "lambda_l1": 1.0, "lambda_l2": 1.0,
         "random_state": 123+fold, "verbose": -1},
        dtr, 400, valid_sets=[dtr, dva], callbacks=[lgb.early_stopping(40, verbose=False)]
    )
    oof_stack[va_idx] = meta_m.predict(stack_train.iloc[va_idx], num_iteration=meta_m.best_iteration)
    test_stack += meta_m.predict(stack_test, num_iteration=meta_m.best_iteration) / 5

stack_oof_score = average_precision_score(y_train, oof_stack)
print(f"[Stack]  LGB Méta-Modèle => OOF PR-AUC={stack_oof_score:.5f}")

if stack_oof_score > scipy_oof_score:
    print(f"\n✅ Le STACKING l'emporte ! ({stack_oof_score:.5f} > {scipy_oof_score:.5f})")
    final_pred = test_stack
else:
    print(f"\n✅ Le MÉLANGE SCIPY l'emporte ! ({scipy_oof_score:.5f} >= {stack_oof_score:.5f})")
    final_pred = scipy_pred

is_not_op3_test = (X_test['operation'] != 'op_03')
final_pred[is_not_op3_test] = 0.0

pred_df = pd.DataFrame({'id': test_fe['id'], 'target': final_pred})

if os.path.exists("/kaggle/input/datasets/octavebahoun/dataset/test.csv"):
    orig_test = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
else:
    orig_test = pd.read_csv("dataset/test.csv")

submission = orig_test[['id']].merge(pred_df, on='id', how='left')
submission.loc[(orig_test['operation'] != 'op_03'), 'target'] = 0.0

submission.to_csv("submission.csv", index=False)
print(f"\nSoumission enregistrée ! Temps total : {time.time()-t0:.0f}s")
