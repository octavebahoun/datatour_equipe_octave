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
print("=== V10: 2-Stage Pseudo-Labeling ===")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
# Vérification de l'emplacement du dataset (Kaggle ou local)
if os.path.exists("/kaggle/input/datasets/octavebahoun/dataset"):
    train_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/train.csv")
    test_raw = pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
else:
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")

# Alignement temporel des enregistrements
train_raw = train_raw.sort_values('period').reset_index(drop=True)
test_raw = test_raw.sort_values('period').reset_index(drop=True)

# Assignation des labels d'ensemble pour le prétraitement conjoint
train_raw['is_test'] = 0
test_raw['is_test'] = 1
test_raw['fraud_flag'] = np.nan

combined = pd.concat([train_raw, test_raw], axis=0).sort_values('period').reset_index(drop=True)
eps = 1e-6

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
# INGÉNIERIE DES CARACTÉRISTIQUES (V8 BASE)
# ============================================================
print("Building features...")
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

# Détection de comptes complètement vidés
combined["amount_equals_origin_before"] = ((combined["amount"] - combined["origin_balance_before"]).abs() < 0.1).astype(int)

# Index cumulés d'activité par compte
combined['orig_tx_idx'] = combined.groupby('origin_account').cumcount()
combined['dest_tx_idx'] = combined.groupby('destination_account').cumcount()
combined['orig_cum_amount'] = combined.groupby('origin_account')['amount'].cumsum() - combined['amount']
combined['dest_cum_amount'] = combined.groupby('destination_account')['amount'].cumsum() - combined['amount']

# Différences de temps par rapport aux transactions antérieures (lags 1 à 3)
for lag in [1, 2, 3]:
    suffix = '' if lag == 1 else f'_{lag}'
    combined[f'orig_time_diff{suffix}'] = (combined['period'] - combined.groupby('origin_account')['period'].shift(lag)).fillna(999)
    combined[f'dest_time_diff{suffix}'] = (combined['period'] - combined.groupby('destination_account')['period'].shift(lag)).fillna(999)

# Agrégations temporelles pour l'expéditeur
orig_period = combined.groupby(['origin_account', 'period']).agg(
    orig_period_tx_count=('amount', 'count'), orig_period_amount_sum=('amount', 'sum')
).reset_index().sort_values(['origin_account', 'period'])
orig_period['orig_cum_tx_count'] = orig_period.groupby('origin_account')['orig_period_tx_count'].cumsum().groupby(orig_period['origin_account']).shift(1).fillna(0)
orig_period['orig_cum_amount_sum'] = orig_period.groupby('origin_account')['orig_period_amount_sum'].cumsum().groupby(orig_period['origin_account']).shift(1).fillna(0)
combined = combined.merge(orig_period[['origin_account', 'period', 'orig_cum_tx_count', 'orig_cum_amount_sum']], on=['origin_account', 'period'], how='left')

# Agrégations temporelles pour le destinataire
dest_period = combined.groupby(['destination_account', 'period']).agg(
    dest_period_tx_count=('amount', 'count'), dest_period_amount_sum=('amount', 'sum')
).reset_index().sort_values(['destination_account', 'period'])
dest_period['dest_cum_tx_count'] = dest_period.groupby('destination_account')['dest_period_tx_count'].cumsum().groupby(dest_period['destination_account']).shift(1).fillna(0)
dest_period['dest_cum_amount_sum'] = dest_period.groupby('destination_account')['dest_period_amount_sum'].cumsum().groupby(dest_period['destination_account']).shift(1).fillna(0)
combined = combined.merge(dest_period[['destination_account', 'period', 'dest_cum_tx_count', 'dest_cum_amount_sum']], on=['destination_account', 'period'], how='left')

# Montants moyens historiques
combined['orig_avg_amount'] = combined['orig_cum_amount_sum'] / (combined['orig_cum_tx_count'] + 1)
combined['dest_avg_amount'] = combined['dest_cum_amount_sum'] / (combined['dest_cum_tx_count'] + 1)
combined['amount_vs_orig_avg'] = combined['amount'] / (combined['orig_avg_amount'] + eps)
combined['amount_vs_dest_avg'] = combined['amount'] / (combined['dest_avg_amount'] + eps)

# Suivi de la diversité des partenaires d'échange
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

# Ratios de balance
combined['origin_balance_ratio'] = combined['origin_balance_after'] / (combined['origin_balance_before'] + eps)
combined['dest_balance_ratio'] = combined['destination_balance_after'] / (combined['destination_balance_before'] + eps)

# Vitesse transactionnelle
combined['amount_velocity_orig'] = combined['amount'] / (combined['orig_time_diff'] + eps)
combined['amount_velocity_dest'] = combined['amount'] / (combined['dest_time_diff'] + eps)

# Encodages cibles chronologiques
print("Computing target encodings...")
combined['origin_te'] = compute_chronological_te(combined, 'origin_account', 'fraud_flag', smoothing=10)
combined['destination_te'] = compute_chronological_te(combined, 'destination_account', 'fraud_flag', smoothing=10)

# Drapeaux spécifiques à op_03
combined['is_op3'] = (combined['operation'] == 'op_03').astype(int)
combined['op3_orig_no_change'] = (combined['is_op3'] & combined['origin_no_change']).astype(int)
combined['op3_dest_no_change'] = (combined['is_op3'] & combined['destination_no_change']).astype(int)

# Conversion catégorielle
combined['operation'] = combined['operation'].astype('category')
combined['origin_account'] = combined['origin_account'].astype('category')
combined['destination_account'] = combined['destination_account'].astype('category')

# Découpage du dataset
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
]
cat_features = ["operation", "origin_account", "destination_account"]
cb_features = [f for f in features if f not in ["origin_account", "destination_account"]]

# ============================================================
# STAGE 1: APPRENTISSAGE DU MODÈLE DE BASE
# ============================================================
print("\n" + "="*50)
print("STAGE 1: TRAINING BASE MODEL")
print("="*50)

X_train = train_fe[features].reset_index(drop=True)
y_train = train_fe["fraud_flag"].reset_index(drop=True)
X_test = test_fe[features].reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb, oof_lgb, oof_cb = np.zeros(len(X_train)), np.zeros(len(X_train)), np.zeros(len(X_train))
test_xgb_s1, test_lgb_s1, test_cb_s1 = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1} ---")
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # XGBoost
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    xgb_m = xgb.train({"objective": "binary:logistic", "eval_metric": "aucpr", "learning_rate": 0.08, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42+fold, "tree_method": "hist", "verbosity": 0}, dtr_xgb, 400, evals=[(dtr_xgb, 'train'), (dval_xgb, 'val')], early_stopping_rounds=50, verbose_eval=False)
    oof_xgb[val_idx] = xgb_m.predict(dval_xgb, iteration_range=(0, xgb_m.best_iteration + 1))
    test_xgb_s1 += xgb_m.predict(xgb.DMatrix(X_test, enable_categorical=True), iteration_range=(0, xgb_m.best_iteration + 1)) / 5
    
    # LightGBM
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtr_lgb, categorical_feature=cat_features)
    lgb_m = lgb.train({"objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 20, "random_state": 42+fold, "n_jobs": -1, "verbose": -1}, dtr_lgb, 600, valid_sets=[dtr_lgb, dval_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = lgb_m.predict(X_val, num_iteration=lgb_m.best_iteration)
    test_lgb_s1 += lgb_m.predict(X_test, num_iteration=lgb_m.best_iteration) / 5
    
    # CatBoost
    cb_m = CatBoostClassifier(iterations=1000, learning_rate=0.08, depth=6, eval_metric='AUC', random_seed=42+fold, verbose=0, early_stopping_rounds=50)
    cb_m.fit(X_tr[cb_features], y_tr, eval_set=(X_val[cb_features], y_val), cat_features=["operation"])
    oof_cb[val_idx] = cb_m.predict_proba(X_val[cb_features])[:, 1]
    test_cb_s1 += cb_m.predict_proba(X_test[cb_features])[:, 1] / 5

# Optimisation des poids de blending
def neg_ap(weights):
    w = weights / np.sum(weights)
    blend = w[0] * oof_xgb + w[1] * oof_lgb + w[2] * oof_cb
    return -average_precision_score(y_train, blend)

res = minimize(neg_ap, x0=[0.33, 0.34, 0.33], method='SLSQP', bounds=[(0,1),(0,1),(0,1)], constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
w1 = res.x / np.sum(res.x)
print(f"Stage 1 Weights: XGB={w1[0]:.3f}, LGB={w1[1]:.3f}, CB={w1[2]:.3f} => PR-AUC={-res.fun:.5f}")
test_preds_s1 = w1[0]*test_xgb_s1 + w1[1]*test_lgb_s1 + w1[2]*test_cb_s1

# Post-processing intermédiaire : forcer les non-op_03 à 0.0
is_not_op3_test = (X_test['operation'] != 'op_03')
test_preds_s1[is_not_op3_test] = 0.0

# ============================================================
# STAGE 2: PSEUDO-LABELING (SEMI-SUPERVISÉ)
# ============================================================
print("\n" + "="*50)
print("STAGE 2: PSEUDO-LABELING & RETRAINING")
print("="*50)

# Filtrage des prédictions tests à forte confiance pour créer des étiquettes synthétiques (pseudo-labels)
# Uniquement sur l'opération op_03 pour éviter de polluer l'apprentissage avec des certitudes théoriques
idx_0 = (test_preds_s1 < 0.15) & (~is_not_op3_test) # Prédiction de non-fraude hautement probable
idx_1 = (test_preds_s1 > 0.60) & (~is_not_op3_test) # Prédiction de fraude hautement probable

print(f"Pseudo Label 0 (<0.15): {idx_0.sum()} rows")
print(f"Pseudo Label 1 (>0.60): {idx_1.sum()} rows")

# Construction des jeux de données pseudo-étiquetés
pseudo_X_0 = X_test[idx_0].copy()
pseudo_y_0 = pd.Series([0.0]*len(pseudo_X_0))
pseudo_w_0 = pd.Series([1.0]*len(pseudo_X_0)) # Poids fort (1.0) car la non-fraude est plus sûre

pseudo_X_1 = X_test[idx_1].copy()
pseudo_y_1 = pd.Series([1.0]*len(pseudo_X_1))
pseudo_w_1 = pd.Series([0.3]*len(pseudo_X_1)) # Poids faible (0.3) pour mitiger le risque de faux positifs

# Définition du poids d'origine des données d'entraînement réelles (poids maximal)
train_w = pd.Series([1.0]*len(X_train))

# Concaténation des données réelles et des pseudo-labels
X_train_pl = pd.concat([X_train, pseudo_X_0, pseudo_X_1], axis=0).reset_index(drop=True)
y_train_pl = pd.concat([y_train, pseudo_y_0, pseudo_y_1], axis=0).reset_index(drop=True)
w_train_pl = pd.concat([train_w, pseudo_w_0, pseudo_w_1], axis=0).reset_index(drop=True)

# Réentraînement sélectif de XGBoost et LightGBM (car CatBoost nécessite une configuration de poids plus complexe)
oof_xgb_s2 = np.zeros(len(X_train)) # OOF calculé uniquement sur les vraies données d'origine
oof_lgb_s2 = np.zeros(len(X_train))
test_xgb_s2 = np.zeros(len(X_test))
test_lgb_s2 = np.zeros(len(X_test))

# Validation croisée stratifiée sur le train set initial pour conserver la validité scientifique de la métrique
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- PL Fold {fold + 1} ---")
    
    # L'ensemble d'entraînement comprend le pli courant + l'intégralité des données pseudo-étiquetées
    fold_train_idx = np.concatenate([train_idx, np.arange(len(X_train), len(X_train_pl))])
    
    X_tr = X_train_pl.iloc[fold_train_idx]
    y_tr = y_train_pl.iloc[fold_train_idx]
    w_tr = w_train_pl.iloc[fold_train_idx]
    
    X_val = X_train.iloc[val_idx] # Évaluation exclusive sur les vraies données de validation
    y_val = y_train.iloc[val_idx]
    
    # XGBoost
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, enable_categorical=True)
    dval_xgb = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    xgb_m = xgb.train({"objective": "binary:logistic", "eval_metric": "aucpr", "learning_rate": 0.08, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42+fold, "tree_method": "hist", "verbosity": 0}, dtr_xgb, 400, evals=[(dtr_xgb, 'train'), (dval_xgb, 'val')], early_stopping_rounds=50, verbose_eval=False)
    oof_xgb_s2[val_idx] = xgb_m.predict(dval_xgb, iteration_range=(0, xgb_m.best_iteration + 1))
    test_xgb_s2 += xgb_m.predict(xgb.DMatrix(X_test, enable_categorical=True), iteration_range=(0, xgb_m.best_iteration + 1)) / 5
    
    # LightGBM
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=cat_features)
    dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtr_lgb, categorical_feature=cat_features)
    lgb_m = lgb.train({"objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 20, "random_state": 42+fold, "n_jobs": -1, "verbose": -1}, dtr_lgb, 600, valid_sets=[dtr_lgb, dval_lgb], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb_s2[val_idx] = lgb_m.predict(X_val, num_iteration=lgb_m.best_iteration)
    test_lgb_s2 += lgb_m.predict(X_test, num_iteration=lgb_m.best_iteration) / 5

# Optimisation des poids sur l'étape de pseudo-labeling
def neg_ap_s2(weights):
    w = weights / np.sum(weights)
    blend = w[0] * oof_xgb_s2 + w[1] * oof_lgb_s2
    return -average_precision_score(y_train, blend)

res_s2 = minimize(neg_ap_s2, x0=[0.5, 0.5], method='SLSQP', bounds=[(0,1),(0,1)], constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
w2 = res_s2.x / np.sum(res_s2.x)
print(f"Stage 2 Weights: XGB={w2[0]:.3f}, LGB={w2[1]:.3f} => PR-AUC={-res_s2.fun:.5f}")

# Prédiction finale pondérée
final_pred = w2[0]*test_xgb_s2 + w2[1]*test_lgb_s2

pred_df = pd.DataFrame({'id': test_fe['id'], 'target': final_pred})
orig_test = pd.read_csv("dataset/test.csv") if not os.path.exists("/kaggle/input/datasets/octavebahoun/dataset") else pd.read_csv("/kaggle/input/datasets/octavebahoun/dataset/test.csv")
submission = orig_test[['id']].merge(pred_df, on='id', how='left')
submission.loc[(orig_test['operation'] != 'op_03'), 'target'] = 0.0

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved! Total time: {time.time()-t0:.0f}s")
