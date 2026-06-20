import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import optuna
import time, os, warnings, gc
warnings.filterwarnings('ignore')

t0 = time.time()
print("=== V9: Advanced Features (Mules, Ratios) + Optuna Tuning ===")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
# Localisation du jeu de données (Kaggle ou environnement local)
if os.path.exists("/kaggle/input/datasets/octavebahoun/dataset"):
    train_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/train.csv")
    test_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
else:
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")

# Tri temporel pour respecter la chronologie des transactions
train_raw = train_raw.sort_values('period').reset_index(drop=True)
test_raw = test_raw.sort_values('period').reset_index(drop=True)

# Alignement des structures pour le feature engineering conjoint
train_raw['is_test'] = 0
test_raw['is_test'] = 1
test_raw['fraud_flag'] = np.nan

combined = pd.concat([train_raw, test_raw], axis=0).sort_values('period').reset_index(drop=True)
eps = 1e-5

# ============================================================
# TARGET ENCODING CHRONOLOGIQUE (ANTI-DATA LEAKAGE)
# ============================================================
def compute_chronological_te(df, group_col, target_col, smoothing=10):
    """
    Calcule le Target Encoding chronologique en décalant d'une période (shift(1))
    les statistiques cumulées d'un groupe pour s'assurer que les informations
    futures ne fuitent pas dans les variables prédictives passées.
    """
    period_stats = df.groupby([group_col, 'period'])[target_col].agg(['sum', 'count']).reset_index()
    period_stats = period_stats.sort_values([group_col, 'period'])
    
    # Calcul des cumulés groupe
    period_stats['cum_sum'] = period_stats.groupby(group_col)['sum'].cumsum()
    period_stats['cum_count'] = period_stats.groupby(group_col)['count'].cumsum()
    
    # Décalage d'un pli temporel
    period_stats['prev_cum_sum'] = period_stats.groupby(group_col)['cum_sum'].shift(1).fillna(0)
    period_stats['prev_cum_count'] = period_stats.groupby(group_col)['cum_count'].shift(1).fillna(0)
    
    # Calcul des cumulés globaux pour l'ajustement global
    global_period_stats = df.groupby('period')[target_col].agg(['sum', 'count']).reset_index()
    global_period_stats = global_period_stats.sort_values('period')
    global_period_stats['cum_sum'] = global_period_stats['sum'].cumsum()
    global_period_stats['cum_count'] = global_period_stats['count'].cumsum()
    global_period_stats['prev_cum_sum'] = global_period_stats['cum_sum'].shift(1).fillna(0)
    global_period_stats['prev_cum_count'] = global_period_stats['cum_count'].shift(1).fillna(0)
    
    global_period_stats['global_mean'] = (global_period_stats['prev_cum_sum'] + 1e-5) / (global_period_stats['prev_cum_count'] + 1e-5)
    
    # Fusion des données et lissage
    period_stats = period_stats.merge(global_period_stats[['period', 'global_mean']], on='period', how='left')
    period_stats['te'] = (period_stats['prev_cum_sum'] + period_stats['global_mean'] * smoothing) / (period_stats['prev_cum_count'] + smoothing)
    
    df_merged = df.merge(period_stats[[group_col, 'period', 'te']], on=[group_col, 'period'], how='left')
    return df_merged['te']

# ============================================================
# INGÉNIERIE DES CARACTÉRISTIQUES DE BASE
# ============================================================
print("Building base features...")
# Montants à l'échelle logarithmique
combined["amount_log1p"] = np.log1p(np.maximum(combined["amount"], 0))

# Différences absolues de solde
combined["origin_balance_change"] = combined["origin_balance_after"] - combined["origin_balance_before"]
combined["destination_balance_change"] = combined["destination_balance_after"] - combined["destination_balance_before"]

# Ratios des montants de transactions sur solde initial
combined["amount_to_origin_before"] = combined["amount"] / (np.abs(combined["origin_balance_before"]) + eps)
combined["amount_to_destination_before"] = combined["amount"] / (np.abs(combined["destination_balance_before"]) + eps)

# Variables de détection d'absence de variation de solde
combined["origin_no_change"] = (combined["origin_balance_change"].abs() < 0.1).astype(int)
combined["destination_no_change"] = (combined["destination_balance_change"].abs() < 0.1).astype(int)

# Index cumulés d'activité par compte
combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()

# Différences de temps par rapport aux transactions antérieures (lags 1 à 2)
for lag in [1, 2]:
    suffix = '' if lag == 1 else f'_{lag}'
    combined[f'orig_time_diff{suffix}'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(lag)).fillna(999)
    combined[f'dest_time_diff{suffix}'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(lag)).fillna(999)

# Vitesse transactionnelle (Montant / Temps écoulé)
combined['amount_velocity_orig'] = combined['amount'] / (combined['orig_time_diff'] + eps)
combined['amount_velocity_dest'] = combined['amount'] / (combined['dest_time_diff'] + eps)

# Caractéristiques réseau basées sur les partenaires d'échange
print("Building network features...")
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

# ============================================================
# DÉTECTION DE COMPTES MULES (FLUX SORTANTS VS ENTRANTS)
# ============================================================
print("Building Mule Ratio features...")
# Somme des montants envoyés par compte et par période
sender_agg = combined.groupby(['origin_account', 'period'])['amount'].sum().reset_index()
sender_agg.rename(columns={'origin_account': 'account', 'amount': 'sent_amount'}, inplace=True)

# Somme des montants reçus par compte et par période
receiver_agg = combined.groupby(['destination_account', 'period'])['amount'].sum().reset_index()
receiver_agg.rename(columns={'destination_account': 'account', 'amount': 'received_amount'}, inplace=True)

# Fusion temporelle par compte
account_period = pd.merge(sender_agg, receiver_agg, on=['account', 'period'], how='outer').fillna(0)
account_period = account_period.sort_values(['account', 'period'])

# Cumul historique des flux entrants et sortants
account_period['cum_sent'] = account_period.groupby('account')['sent_amount'].cumsum()
account_period['cum_received'] = account_period.groupby('account')['received_amount'].cumsum()

# Décalage temporel d'une période (shift(1)) pour éviter strictement toute fuite de données
account_period['cum_sent_past'] = account_period.groupby('account')['cum_sent'].shift(1).fillna(0)
account_period['cum_received_past'] = account_period.groupby('account')['cum_received'].shift(1).fillna(0)

# Association des métriques de flux passés aux émetteurs
combined = combined.merge(
    account_period[['account', 'period', 'cum_sent_past', 'cum_received_past']].rename(
        columns={'account': 'origin_account', 'cum_sent_past': 'orig_cum_sent_past', 'cum_received_past': 'orig_cum_received_past'}
    ), on=['origin_account', 'period'], how='left'
)

# Association des métriques de flux passés aux destinataires
combined = combined.merge(
    account_period[['account', 'period', 'cum_sent_past', 'cum_received_past']].rename(
        columns={'account': 'destination_account', 'cum_sent_past': 'dest_cum_sent_past', 'cum_received_past': 'dest_cum_received_past'}
    ), on=['destination_account', 'period'], how='left'
)

# Ratios de mule (Reçu / Envoyé) : Un ratio proche de 1.0 indique un compte de transit (mule)
combined['orig_mule_ratio'] = combined['orig_cum_received_past'] / (combined['orig_cum_sent_past'] + eps)
combined['dest_mule_ratio'] = combined['dest_cum_received_past'] / (combined['dest_cum_sent_past'] + eps)

# Ratios des soldes avant/après transaction
combined['origin_balance_ratio'] = combined['origin_balance_after'] / (combined['origin_balance_before'] + eps)
combined['dest_balance_ratio'] = combined['destination_balance_after'] / (combined['destination_balance_before'] + eps)

# Encodages cibles chronologiques
print("Computing target encodings...")
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)

# Conversion catégorielle
combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')

# Découpage final des ensembles train et test
train_fe = combined[combined['is_test'] == 0].copy().drop(columns=['is_test'])
test_fe = combined[combined['is_test'] == 1].copy().drop(columns=['is_test'])
del combined, account_period, sender_agg, receiver_agg; gc.collect()

features = [
    "period", "operation", "origin_account", "destination_account", "amount", "amount_log1p",
    "origin_balance_before", "origin_balance_after", "origin_balance_change",
    "destination_balance_before", "destination_balance_after", "destination_balance_change",
    "amount_to_origin_before", "amount_to_destination_before",
    "origin_no_change", "destination_no_change",
    "orig_tx_idx", "dest_tx_idx",
    "orig_time_diff", "dest_time_diff", "orig_time_diff_2", "dest_time_diff_2",
    "origin_te", "destination_te", "is_op3",
    "amount_velocity_orig", "amount_velocity_dest",
    "orig_cum_unique_dests", "dest_cum_unique_origins",
    "orig_cum_sent_past", "orig_cum_received_past", "orig_mule_ratio",
    "dest_cum_sent_past", "dest_cum_received_past", "dest_mule_ratio",
    "origin_balance_ratio", "dest_balance_ratio"
]

cat_features = ["operation", "origin_account", "destination_account"]

X_train = train_fe[features].reset_index(drop=True)
y_train = train_fe["fraud_flag"].reset_index(drop=True)
X_test = test_fe[features].reset_index(drop=True)

print(f"Training set: {len(X_train)} rows")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ============================================================
# OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA (LightGBM)
# ============================================================
print("\n=== OPTUNA Tuning for LightGBM ===")
def objective(trial):
    # Paramètres de recherche de l'étude Optuna
    param = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    # Utilisation de 3 plis pour accélérer le processus de tuning
    fast_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in fast_skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=cat_features)
        
        model = lgb.train(
            param, dtr, num_boost_round=400,
            valid_sets=[dtr, dval],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        score = average_precision_score(y_val, preds)
        scores.append(score)
        
    return np.mean(scores)

# Limitation à 15 essais (trials) en raison des contraintes temporelles
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print("\nBest Optuna Params:", study.best_params)
best_lgb_params = study.best_params
best_lgb_params.update({
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
})

# ============================================================
# APPRENTISSAGE FINAL (XGBoost + LightGBM optimisé)
# ============================================================
print(f"\n=== Starting {n_splits}-Fold CV ===")
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1} / {n_splits} ---")
    t_fold = time.time()
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # Entraînement de XGBoost
    print(f"  XGBoost...")
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest_xgb = xgb.DMatrix(X_test, enable_categorical=True)
    
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "learning_rate": 0.05, "max_depth": 7,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "seed": 42 + fold, "tree_method": "hist", "verbosity": 0
    }
    xgb_model = xgb.train(xgb_params, dtr_xgb, num_boost_round=500, evals=[(dtr_xgb, 'train'), (dval_xgb, 'val')], early_stopping_rounds=40, verbose_eval=False)
    oof_xgb[val_idx] = xgb_model.predict(dval_xgb, iteration_range=(0, xgb_model.best_iteration + 1))
    test_xgb += xgb_model.predict(dtest_xgb, iteration_range=(0, xgb_model.best_iteration + 1)) / n_splits
    
    # Entraînement de LightGBM (avec hyperparamètres optimisés par Optuna)
    print(f"  LightGBM...")
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtr_lgb, categorical_feature=cat_features)
    lgb_model = lgb.train(best_lgb_params, dtr_lgb, num_boost_round=800, valid_sets=[dtr_lgb, dval_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
    
    print(f"  Fold done in {time.time()-t_fold:.0f}s")
    gc.collect()

print("\n=== Individual OOF Performances ===")
xgb_ap = average_precision_score(y_train, oof_xgb)
lgb_ap = average_precision_score(y_train, oof_lgb)
print(f"XGBoost  OOF PR-AUC: {xgb_ap:.6f}")
print(f"LightGBM OOF PR-AUC: {lgb_ap:.6f}")

print("\n=== Optimizing Blend Weights ===")
def neg_ap(weights):
    w = weights / np.sum(weights)
    blend = w[0] * oof_xgb + w[1] * oof_lgb
    return -average_precision_score(y_train, blend)

best_w = (0.5, 0.5)
res = minimize(neg_ap, x0=list(best_w), method='SLSQP', bounds=[(0,1),(0,1)], constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
opt_w = res.x / np.sum(res.x)
opt_ap = -res.fun
print(f"Scipy refined: XGB={opt_w[0]:.4f}, LGB={opt_w[1]:.4f} => PR-AUC={opt_ap:.6f}")

# Prédiction pondérée
final_pred = opt_w[0] * test_xgb + opt_w[1] * test_lgb

print("\nAligning predictions to test file order...")
pred_df = pd.DataFrame({'id': test_fe['id'], 'target': final_pred})
orig_test = pd.read_csv("dataset/test.csv") if not os.path.exists("/kaggle/input/datasets/octavebahoun/dataset") else pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
submission = orig_test[['id']].merge(pred_df, on='id', how='left')

print("Applying post-processing (non-op_03 = 0)...")
submission.loc[(orig_test['operation'] != 'op_03'), 'target'] = 0.0

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved! Total time: {time.time()-t0:.0f}s")
