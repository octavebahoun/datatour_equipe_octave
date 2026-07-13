"""
Détection de fraude — Ensemble Stacking avec PageRank sans fuite (leak-free).

Pipeline complet :
    1. Chargement et fusion des données train/test.
    2. Ingénierie des caractéristiques (montants, soldes, dynamiques d'arêtes,
       PageRank, ancienneté des comptes, cumuls historiques, target encoding).
    3. Niveau 1 : entraînement de trois modèles (XGBoost, LightGBM, CatBoost)
       en validation croisée stratifiée à 5 plis, avec prédictions hors-sac (OOF).
    4. Niveau 2 : comparaison entre un mélange linéaire optimisé (Scipy)
       et un méta-modèle LightGBM empilant les prédictions de niveau 1.
    5. Génération du fichier de soumission.

Remarque : toute la logique préserve la causalité temporelle (usage du passé
uniquement via shift/cumsum) afin d'éviter les fuites d'information.
"""

import time
import os
import warnings
import gc

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
import networkx as nx

warnings.filterwarnings("ignore")


# =====================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================================

temps_debut = time.time()
print("===  Leak-Free PageRank ===")

DOSSIER_DONNEES = os.path.dirname(os.path.abspath(__file__))
donnees_entrainement_brutes = pd.read_csv(os.path.join(DOSSIER_DONNEES, "train.csv"))
donnees_test_brutes = pd.read_csv(os.path.join(DOSSIER_DONNEES, "test.csv"))

# Tri chronologique : indispensable pour tous les calculs cumulés qui suivent.
donnees_entrainement_brutes = donnees_entrainement_brutes.sort_values("period").reset_index(drop=True)
donnees_test_brutes = donnees_test_brutes.sort_values("period").reset_index(drop=True)

# Marqueur train/test et cible masquée pour le test.
donnees_entrainement_brutes["is_test"] = 0
donnees_test_brutes["is_test"] = 1
donnees_test_brutes["fraud_flag"] = np.nan

# Fusion des deux jeux pour calculer les caractéristiques de façon cohérente.
combine = pd.concat(
    [donnees_entrainement_brutes, donnees_test_brutes], axis=0
).sort_values("period").reset_index(drop=True)

epsilon = 1e-6


# =====================================================================
# 2. INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING)
# =====================================================================

def calculer_te_chronologique(df, group_col, target_col, smoothing=10):
    """
    Calcule un target encoding chronologique par groupe.

    Groupe les données par colonne de groupement et période, accumule la somme
    et le compte de la cible dans le passé, puis calcule un ratio lissé avec la
    moyenne globale. N'utilise que les périodes antérieures : aucun leakage.

    Args:
        df (DataFrame): données contenant les colonnes utilisées.
        group_col (str): colonne servant au groupement (ex. compte origine).
        target_col (str): colonne cible à encoder (fraud_flag).
        smoothing (int): parametre de lissage bayésien (défaut 10).

    Returns:
        pd.Series: valeur encodée pour chaque ligne de `df`.
    """
    statistiques_periode = (
        df.groupby([group_col, "period"])[target_col].agg(["sum", "count"]).reset_index()
    )
    statistiques_periode = statistiques_periode.sort_values([group_col, "period"])
    statistiques_periode["cum_sum"] = statistiques_periode.groupby(group_col)["sum"].cumsum()
    statistiques_periode["cum_count"] = statistiques_periode.groupby(group_col)["count"].cumsum()
    statistiques_periode["prev_cum_sum"] = (
        statistiques_periode.groupby(group_col)["cum_sum"].shift(1).fillna(0)
    )
    statistiques_periode["prev_cum_count"] = (
        statistiques_periode.groupby(group_col)["cum_count"].shift(1).fillna(0)
    )

    statistiques_globales_periode = (
        df.groupby("period")[target_col].agg(["sum", "count"]).reset_index()
    )
    statistiques_globales_periode = statistiques_globales_periode.sort_values("period")
    statistiques_globales_periode["cum_sum"] = statistiques_globales_periode["sum"].cumsum()
    statistiques_globales_periode["cum_count"] = statistiques_globales_periode["count"].cumsum()
    statistiques_globales_periode["prev_cum_sum"] = (
        statistiques_globales_periode["cum_sum"].shift(1).fillna(0)
    )
    statistiques_globales_periode["prev_cum_count"] = (
        statistiques_globales_periode["cum_count"].shift(1).fillna(0)
    )

    statistiques_globales_periode["global_mean"] = (
        statistiques_globales_periode["prev_cum_sum"] + 1e-5
    ) / (statistiques_globales_periode["prev_cum_count"] + 1e-5)
    statistiques_periode = statistiques_periode.merge(
        statistiques_globales_periode[["period", "global_mean"]], on="period", how="left"
    )
    statistiques_periode["te"] = (
        statistiques_periode["prev_cum_sum"]
        + statistiques_periode["global_mean"] * smoothing
    ) / (statistiques_periode["prev_cum_count"] + smoothing)
    trame_fusionnee = df.merge(
        statistiques_periode[[group_col, "period", "te"]],
        on=[group_col, "period"],
        how="left",
    )
    return trame_fusionnee["te"]


# --- 2.1 Caractéristiques de base : montants, soldes, indicateurs binaires ---
print("Construction des caractéristiques de base...")
combine["amount_log1p"] = np.log1p(np.maximum(combine["amount"], 0))
combine["origin_balance_change"] = combine["origin_balance_after"] - combine["origin_balance_before"]
combine["destination_balance_change"] = combine["destination_balance_after"] - combine["destination_balance_before"]
combine["amount_to_origin_before"] = combine["amount"] / (np.abs(combine["origin_balance_before"]) + epsilon)
combine["amount_to_destination_before"] = combine["amount"] / (np.abs(combine["destination_balance_before"]) + epsilon)
combine["origin_no_change"] = (combine["origin_balance_change"].abs() < 0.1).astype(int)
combine["destination_no_change"] = (combine["destination_balance_change"].abs() < 0.1).astype(int)
combine["amount_equals_origin_before"] = ((combine["amount"] - combine["origin_balance_before"]).abs() < 0.1).astype(int)

combine["edge_id"] = combine["origin_account"].astype(str) + "_" + combine["destination_account"].astype(str)
combine["is_round_1000"] = (combine["amount"] % 1000 == 0).astype(int)
combine["is_round_5000"] = (combine["amount"] % 5000 == 0).astype(int)


# --- 2.2 Dynamiques de relations entre comptes (Edge Dynamics) ---
print("Calcul des dynamiques de relations (Edge Dynamics)...")
combine["edge_time_diff"] = (combine["period"] - combine.groupby("edge_id")["period"].shift(1)).fillna(999)
combine["prev_edge_amount"] = combine.groupby("edge_id")["amount"].shift(1)
combine["is_repeated_amount_on_edge"] = (combine["amount"] == combine["prev_edge_amount"]).astype(int)
combine.drop(columns=["prev_edge_amount"], inplace=True)
combine["edge_cum_tx_count"] = combine.groupby("edge_id").cumcount()
combine["edge_cum_amount_sum"] = combine.groupby("edge_id")["amount"].cumsum() - combine["amount"]


# --- 2.3 PageRank sur le graphe des transactions (données train uniquement) ---
# Calculé exclusivement sur le train pour éviter toute fuite depuis le futur.
# Le poids des arêtes est le nombre de transactions entre deux comptes.
print("Calcul du PageRank sur le graphe des transactions (sans leakage)...")
entrainement_pour_graphe = combine[combine["is_test"] == 0]
poids_aretes = (
    entrainement_pour_graphe.groupby(["origin_account", "destination_account"])["amount"]
    .count()
    .reset_index()
)
poids_aretes.rename(columns={"amount": "weight"}, inplace=True)
graphe = nx.from_pandas_edgelist(
    poids_aretes, "origin_account", "destination_account", ["weight"], create_using=nx.DiGraph()
)
scores_pagerank = nx.pagerank(graphe, weight="weight")
trame_pagerank = pd.DataFrame(list(scores_pagerank.items()), columns=["account", "pagerank"])
combine = combine.merge(
    trame_pagerank.rename(columns={"account": "origin_account", "pagerank": "orig_pagerank"}),
    on="origin_account",
    how="left",
)
combine = combine.merge(
    trame_pagerank.rename(columns={"account": "destination_account", "pagerank": "dest_pagerank"}),
    on="destination_account",
    how="left",
)
combine["orig_pagerank"] = combine["orig_pagerank"].fillna(0)
combine["dest_pagerank"] = combine["dest_pagerank"].fillna(0)
del graphe, poids_aretes, scores_pagerank, trame_pagerank, entrainement_pour_graphe
gc.collect()


# --- 2.4 Ancienneté des comptes et rang du montant (optimisé mémoire OOM) ---
# shift() puis cummax() évitent l'explosion mémoire tout en restant causaux.
print("Calcul des caractéristiques de Rank (Optimisées OOM)...")
premiere_apparition_origine = combine.groupby("origin_account")["period"].transform("first")
premiere_apparition_destination = combine.groupby("destination_account")["period"].transform("first")
combine["orig_account_age"] = combine["period"] - premiere_apparition_origine
combine["dest_account_age"] = combine["period"] - premiere_apparition_destination

combine["orig_max_amount_so_far"] = combine.groupby("origin_account")["amount"].shift(1)
combine["orig_max_amount_so_far"] = (
    combine.groupby("origin_account")["orig_max_amount_so_far"].cummax().fillna(0)
)
combine["orig_amount_rank"] = combine["amount"] / (combine["orig_max_amount_so_far"] + epsilon)

combine["dest_max_amount_so_far"] = combine.groupby("destination_account")["amount"].shift(1)
combine["dest_max_amount_so_far"] = (
    combine.groupby("destination_account")["dest_max_amount_so_far"].cummax().fillna(0)
)
combine["dest_amount_rank"] = combine["amount"] / (combine["dest_max_amount_so_far"] + epsilon)


# --- 2.5 Caractéristiques historiques cumulées par compte (V14) ---
# Comptes de transactions, sommes cumulées, écarts temporels entre transactions.
print("Calcul des caractéristiques de base (V14)...")
combine["orig_tx_idx"] = combine.groupby("origin_account").cumcount()
combine["dest_tx_idx"] = combine.groupby("destination_account").cumcount()
combine["orig_cum_amount"] = combine.groupby("origin_account")["amount"].cumsum() - combine["amount"]
combine["dest_cum_amount"] = combine.groupby("destination_account")["amount"].cumsum() - combine["amount"]

for decalage in [1, 2, 3]:
    suffixe = "" if decalage == 1 else f"_{decalage}"
    combine[f"orig_time_diff{suffixe}"] = (
        combine["period"] - combine.groupby("origin_account")["period"].shift(decalage)
    ).fillna(999)
    combine[f"dest_time_diff{suffixe}"] = (
        combine["period"] - combine.groupby("destination_account")["period"].shift(decalage)
    ).fillna(999)


# --- 2.6 Agrégations par compte et période (cumuls et moyennes passées) ---
# Les shift(1) garantissent l'utilisation du passé uniquement.
periode_origine = (
    combine.groupby(["origin_account", "period"])
    .agg(
        orig_period_tx_count=("amount", "count"),
        orig_period_amount_sum=("amount", "sum"),
    )
    .reset_index()
    .sort_values(["origin_account", "period"])
)
periode_origine["orig_cum_tx_count"] = (
    periode_origine.groupby("origin_account")["orig_period_tx_count"]
    .cumsum()
    .groupby(periode_origine["origin_account"])
    .shift(1)
    .fillna(0)
)
periode_origine["orig_cum_amount_sum"] = (
    periode_origine.groupby("origin_account")["orig_period_amount_sum"]
    .cumsum()
    .groupby(periode_origine["origin_account"])
    .shift(1)
    .fillna(0)
)
combine = combine.merge(
    periode_origine[["origin_account", "period", "orig_cum_tx_count", "orig_cum_amount_sum"]],
    on=["origin_account", "period"],
    how="left",
)

periode_destination = (
    combine.groupby(["destination_account", "period"])
    .agg(
        dest_period_tx_count=("amount", "count"),
        dest_period_amount_sum=("amount", "sum"),
    )
    .reset_index()
    .sort_values(["destination_account", "period"])
)
periode_destination["dest_cum_tx_count"] = (
    periode_destination.groupby("destination_account")["dest_period_tx_count"]
    .cumsum()
    .groupby(periode_destination["destination_account"])
    .shift(1)
    .fillna(0)
)
periode_destination["dest_cum_amount_sum"] = (
    periode_destination.groupby("destination_account")["dest_period_amount_sum"]
    .cumsum()
    .groupby(periode_destination["destination_account"])
    .shift(1)
    .fillna(0)
)
combine = combine.merge(
    periode_destination[["destination_account", "period", "dest_cum_tx_count", "dest_cum_amount_sum"]],
    on=["destination_account", "period"],
    how="left",
)

combine["orig_avg_amount"] = combine["orig_cum_amount_sum"] / (combine["orig_cum_tx_count"] + 1)
combine["dest_avg_amount"] = combine["dest_cum_amount_sum"] / (combine["dest_cum_tx_count"] + 1)
combine["amount_vs_orig_avg"] = combine["amount"] / (combine["orig_avg_amount"] + epsilon)
combine["amount_vs_dest_avg"] = combine["amount"] / (combine["dest_avg_amount"] + epsilon)


# --- 2.7 Diversité des contreparties (destinations/origines uniques cumulées) ---
comptes_origine_destination = (
    combine.groupby(["origin_account", "period"])["destination_account"].nunique().reset_index()
)
comptes_origine_destination.columns = ["origin_account", "period", "orig_unique_dests_this_period"]
comptes_origine_destination = comptes_origine_destination.sort_values(["origin_account", "period"])
comptes_origine_destination["orig_cum_unique_dests"] = (
    comptes_origine_destination.groupby("origin_account")["orig_unique_dests_this_period"]
    .cumsum()
    .groupby(comptes_origine_destination["origin_account"])
    .shift(1)
    .fillna(0)
)
combine = combine.merge(
    comptes_origine_destination[["origin_account", "period", "orig_cum_unique_dests"]],
    on=["origin_account", "period"],
    how="left",
)

comptes_destination_origine = (
    combine.groupby(["destination_account", "period"])["origin_account"].nunique().reset_index()
)
comptes_destination_origine.columns = ["destination_account", "period", "dest_unique_origins_this_period"]
comptes_destination_origine = comptes_destination_origine.sort_values(["destination_account", "period"])
comptes_destination_origine["dest_cum_unique_origins"] = (
    comptes_destination_origine.groupby("destination_account")["dest_unique_origins_this_period"]
    .cumsum()
    .groupby(comptes_destination_origine["destination_account"])
    .shift(1)
    .fillna(0)
)
combine = combine.merge(
    comptes_destination_origine[["destination_account", "period", "dest_cum_unique_origins"]],
    on=["destination_account", "period"],
    how="left",
)


# --- 2.8 Ratios de soldes et vitesse des montants ---
combine["origin_balance_ratio"] = combine["origin_balance_after"] / (combine["origin_balance_before"] + epsilon)
combine["dest_balance_ratio"] = combine["destination_balance_after"] / (combine["destination_balance_before"] + epsilon)

combine["amount_velocity_orig"] = combine["amount"] / (combine["orig_time_diff"] + epsilon)
combine["amount_velocity_dest"] = combine["amount"] / (combine["dest_time_diff"] + epsilon)


# --- 2.9 Target encodings chronologiques (origine, destination, arête) ---
# Capture la propension à la fraude de chaque entité au fil du temps.
print("Calcul des target encodings chronologiques...")
combine["origin_te"] = calculer_te_chronologique(combine, "origin_account", "fraud_flag", smoothing=10)
combine["destination_te"] = calculer_te_chronologique(combine, "destination_account", "fraud_flag", smoothing=10)
combine["edge_te"] = calculer_te_chronologique(combine, "edge_id", "fraud_flag", smoothing=5)

combine["is_op3"] = (combine["operation"] == "op_03").astype(int)
combine["op3_orig_no_change"] = (combine["is_op3"] & combine["origin_no_change"]).astype(int)
combine["op3_dest_no_change"] = (combine["is_op3"] & combine["destination_no_change"]).astype(int)


# --- 2.10 Typage catégoriel et séparation train/test ---
combine["operation"] = combine["operation"].astype("category")
combine["origin_account"] = combine["origin_account"].astype("category")
combine["destination_account"] = combine["destination_account"].astype("category")
combine["edge_id"] = combine["edge_id"].astype("category")

entrainement_caracteristiques = combine[combine["is_test"] == 0].copy().drop(columns=["is_test"])
test_caracteristiques = combine[combine["is_test"] == 1].copy().drop(columns=["is_test"])
del combine
gc.collect()


# --- 2.11 Listes de caractéristiques par modèle ---
caracteristiques = [
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
]

caracteristiques_categorielles = ["operation", "origin_account", "destination_account", "edge_id"]
caracteristiques_xgb = [f for f in caracteristiques if f != "edge_id"]
caracteristiques_cb = [f for f in caracteristiques if f not in ["origin_account", "destination_account", "edge_id"]]


# =====================================================================
# 3. NIVEAU 1 — MODÈLES DE BASE (XGBoost / LightGBM / CatBoost)
# =====================================================================

print("\n" + "=" * 50)
print(" NIVEAU 1")
print("=" * 50)

X_entrainement = entrainement_caracteristiques[caracteristiques].reset_index(drop=True)
y_entrainement = entrainement_caracteristiques["fraud_flag"].reset_index(drop=True)
X_test = test_caracteristiques[caracteristiques].reset_index(drop=True)

valid_croisee = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prédictions hors-sac (OOF) pour le train, moyennées pour le test.
hors_sac_xgb, hors_sac_lgb, hors_sac_cb = (
    np.zeros(len(X_entrainement)),
    np.zeros(len(X_entrainement)),
    np.zeros(len(X_entrainement)),
)
pred_test_xgb, pred_test_lgb, pred_test_cb = (
    np.zeros(len(X_test)),
    np.zeros(len(X_test)),
    np.zeros(len(X_test)),
)

for pli, (indices_entrainement, indices_validation) in enumerate(
    valid_croisee.split(X_entrainement, y_entrainement)
):
    print(f"\n--- Plis (Fold) {pli + 1} ---")
    X_ent, y_ent = X_entrainement.iloc[indices_entrainement], y_entrainement.iloc[indices_entrainement]
    X_validation, y_validation = X_entrainement.iloc[indices_validation], y_entrainement.iloc[indices_validation]

    # 1. XGBoost
    donnees_ent_xgb = xgb.DMatrix(X_ent[caracteristiques_xgb], label=y_ent, enable_categorical=True)
    donnees_val_xgb = xgb.DMatrix(X_validation[caracteristiques_xgb], label=y_validation, enable_categorical=True)
    modele_xgb = xgb.train(
        {
            "objective": "binary:logistic", "eval_metric": "aucpr", "learning_rate": 0.08, "max_depth": 6,
            "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42 + pli, "tree_method": "hist", "verbosity": 0,
        },
        donnees_ent_xgb, 400,
        evals=[(donnees_ent_xgb, "train"), (donnees_val_xgb, "val")],
        early_stopping_rounds=50, verbose_eval=False,
    )
    hors_sac_xgb[indices_validation] = modele_xgb.predict(
        donnees_val_xgb, iteration_range=(0, modele_xgb.best_iteration + 1)
    )
    pred_test_xgb += modele_xgb.predict(
        xgb.DMatrix(X_test[caracteristiques_xgb], enable_categorical=True),
        iteration_range=(0, modele_xgb.best_iteration + 1),
    ) / 5

    # 2. LightGBM
    donnees_ent_lgb = lgb.Dataset(X_ent, label=y_ent, categorical_feature=caracteristiques_categorielles)
    donnees_val_lgb = lgb.Dataset(
        X_validation, label=y_validation, reference=donnees_ent_lgb,
        categorical_feature=caracteristiques_categorielles,
    )
    modele_lgb = lgb.train(
        {
            "objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 31,
            "max_depth": -1, "min_data_in_leaf": 20, "random_state": 42 + pli, "n_jobs": -1, "verbose": -1,
        },
        donnees_ent_lgb, 600,
        valid_sets=[donnees_ent_lgb, donnees_val_lgb],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    hors_sac_lgb[indices_validation] = modele_lgb.predict(X_validation, num_iteration=modele_lgb.best_iteration)
    pred_test_lgb += modele_lgb.predict(X_test, num_iteration=modele_lgb.best_iteration) / 5

    # 3. CatBoost
    modele_cb = CatBoostClassifier(
        iterations=1000, learning_rate=0.08, depth=6, eval_metric="AUC",
        random_seed=42 + pli, verbose=0, early_stopping_rounds=50,
    )
    modele_cb.fit(
        X_ent[caracteristiques_cb], y_ent,
        eval_set=(X_validation[caracteristiques_cb], y_validation),
        cat_features=["operation"],
    )
    hors_sac_cb[indices_validation] = modele_cb.predict_proba(X_validation[caracteristiques_cb])[:, 1]
    pred_test_cb += modele_cb.predict_proba(X_test[caracteristiques_cb])[:, 1] / 5


# =====================================================================
# 4. NIVEAU 2 — MÉLANGE LINÉAIRE (SCIPY) VS MÉTA-MODÈLE (LGB STACKING)
# =====================================================================

print("\n" + "=" * 50)
print("NIVEAU 2 : MÉLANGE LINÉAIRE SCIPY VS ENHANCED STACKING LGB")
print("=" * 50)


# --- Méthode 1 : mélange linéaire optimisé par Scipy (SLSQP) ---
# Optimise les poids (XGB, LGB, CB) pour maximiser la PR-AUC sur les OOF.
def neg_precision_moyenne(weights):
    """Renvoie l'opposé de la PR-AUC du mélange (à minimiser)."""
    w = weights / np.sum(weights)
    blend = w[0] * hors_sac_xgb + w[1] * hors_sac_lgb + w[2] * hors_sac_cb
    return -average_precision_score(y_entrainement, blend)


resultat_optimisation = minimize(
    neg_precision_moyenne,
    x0=[0.33, 0.34, 0.33],
    method="SLSQP",
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
)
poids_optimaux = resultat_optimisation.x / np.sum(resultat_optimisation.x)
score_hors_sac_scipy = -resultat_optimisation.fun
prediction_scipy = (
    poids_optimaux[0] * pred_test_xgb
    + poids_optimaux[1] * pred_test_lgb
    + poids_optimaux[2] * pred_test_cb
)
print(
    f"[Scipy]  Poids : XGB={poids_optimaux[0]:.3f}, LGB={poids_optimaux[1]:.3f}, "
    f"CB={poids_optimaux[2]:.3f} => OOF PR-AUC={score_hors_sac_scipy:.5f}"
)


# --- Méthode 2 : méta-modèle LightGBM (Enhanced Stacking V17) ---
# Entraîne un modèle de second niveau sur les prédictions OOF combinées
# aux caractéristiques de niveau 1 les plus importantes.
pile_entrainement = pd.DataFrame({"xgb": hors_sac_xgb, "lgb": hors_sac_lgb, "cb": hors_sac_cb})
pile_test = pd.DataFrame({"xgb": pred_test_xgb, "lgb": pred_test_lgb, "cb": pred_test_cb})

caracteristiques_cles_pile = [
    "amount", "is_op3", "origin_te", "destination_te", "edge_te",
    "orig_account_age", "dest_account_age", "orig_amount_rank", "dest_amount_rank",
    "edge_time_diff", "edge_cum_tx_count", "amount_velocity_orig", "orig_time_diff",
    "orig_pagerank", "dest_pagerank",
]
for f in caracteristiques_cles_pile:
    pile_entrainement[f] = X_entrainement[f].values
    pile_test[f] = X_test[f].values

hors_sac_pile = np.zeros(len(pile_entrainement))
pred_test_pile = np.zeros(len(pile_test))

valid_croisee2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
for pli, (ind_ent, ind_val) in enumerate(valid_croisee2.split(pile_entrainement, y_entrainement)):
    donnees_ent = lgb.Dataset(pile_entrainement.iloc[ind_ent], label=y_entrainement.iloc[ind_ent])
    donnees_val = lgb.Dataset(
        pile_entrainement.iloc[ind_val], label=y_entrainement.iloc[ind_val], reference=donnees_ent
    )
    modele_meta = lgb.train(
        {
            "objective": "binary", "metric": "average_precision", "learning_rate": 0.03,
            "num_leaves": 21, "min_data_in_leaf": 30, "lambda_l1": 1.0, "lambda_l2": 1.0,
            "random_state": 123 + pli, "verbose": -1,
        },
        donnees_ent, 400,
        valid_sets=[donnees_ent, donnees_val],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    hors_sac_pile[ind_val] = modele_meta.predict(
        pile_entrainement.iloc[ind_val], num_iteration=modele_meta.best_iteration
    )
    pred_test_pile += modele_meta.predict(pile_test, num_iteration=modele_meta.best_iteration) / 5

score_hors_sac_pile = average_precision_score(y_entrainement, hors_sac_pile)
print(f"[Stack]  LGB Méta-Modèle => OOF PR-AUC={score_hors_sac_pile:.5f}")


# --- Sélection de la meilleure stratégie sur la PR-AUC hors-sac ---
if score_hors_sac_pile > score_hors_sac_scipy:
    print(f"\nretenue : stacking({score_hors_sac_pile:.5f} > {score_hors_sac_scipy:.5f})")
    prediction_finale = pred_test_pile
else:
    print(f"\n retenue scipy ({score_hors_sac_scipy:.5f} >= {score_hors_sac_pile:.5f})")
    prediction_finale = prediction_scipy


# =====================================================================
# 5. POST-TRAITEMENT ET SOUMISSION
# =====================================================================

# La métrique n'évalue que les transactions op_03 : les autres sont forcées à 0.
pas_op3_test = X_test["operation"] != "op_03"
prediction_finale[pas_op3_test] = 0.0

trame_prediction = pd.DataFrame({"id": test_caracteristiques["id"], "target": prediction_finale})

test_original = pd.read_csv(os.path.join(DOSSIER_DONNEES, "test.csv"))
soumission = test_original[["id"]].merge(trame_prediction, on="id", how="left")
soumission.loc[(test_original["operation"] != "op_03"), "target"] = 0.0

soumission.to_csv("soumission.csv", index=False)
print(f"\nSoumission enregistrée ! Temps total : {time.time() - temps_debut:.0f}s")
