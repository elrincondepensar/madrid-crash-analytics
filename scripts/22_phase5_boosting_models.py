"""
Phase 5 · Iteration 1 (v2) — LGBM + CatBoost on 4-class target with Optuna.

Key changes vs v1:
  - Optuna with TPE sampler + MedianPruner (kills bad trials early)
  - Native early stopping (LGBM callbacks, CatBoost eval_set)
  - Narrower n_estimators range (200-800 instead of 300-2000)
  - No oversubscription: LGBMClassifier(n_jobs=1) inside Optuna trials;
    CatBoost thread_count=-1 but sequential trials
  - Estimated runtime: ~45-60 min total (vs 2-6 h in v1)

Run:
    pip install lightgbm catboost optuna
    python scripts/22_phase5_boosting_models.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]

NUMERIC_M2 = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "is_weekend",
    "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
    "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
    "coordenada_x_utm", "coordenada_y_utm",
]
CAT_M2 = ["tipo_accidente", "estado_meteorologico", "distrito"]
NUMERIC_NETWORK = ["edge_maxspeed", "edge_lanes", "edge_length_m",
                   "edge_distance_m", "edge_highway_ordinal"]
CAT_NETWORK = ["edge_highway", "tipo_via"]


# ===========================================================================
# Data prep
# ===========================================================================
def build_splits_v2(year: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(year <= 2022, "train",
                 np.where(year <= 2024, "val", "test")),
        index=year.index,
    )


def build_features(df: pd.DataFrame, with_network: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    hour = df["hour"].astype(float)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    month = df["fecha"].dt.month.astype(float)
    weekday = df["fecha"].dt.weekday.astype(float)
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    X["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    X["is_weekend"] = (weekday >= 5).astype(float)
    X["n_people"] = df["n_people"].astype(float)
    for c in ["n_vehicles", "n_conductores", "n_pasajeros", "n_peatones"]:
        X[c] = df[c].astype(float)
    for c in ["has_bicycle", "has_motorcycle", "has_heavy_vehicle",
              "has_pedestrian"]:
        X[c] = df[c].astype(float)
    X["coordenada_x_utm"] = df["coordenada_x_utm"].astype(float)
    X["coordenada_y_utm"] = df["coordenada_y_utm"].astype(float)
    X["tipo_accidente"] = df["tipo_accidente"].fillna("Desconocido").astype(str)
    X["estado_meteorologico"] = (df["estado_meteorologico"]
                                  .fillna("Se desconoce").astype(str))
    X["distrito"] = df["distrito"].fillna("DESCONOCIDO").astype(str)
    if with_network:
        X["edge_maxspeed"] = df["edge_maxspeed"].astype(float)
        X["edge_lanes"] = df["edge_lanes"].astype(float)
        X["edge_length_m"] = df["edge_length_m"].astype(float)
        X["edge_distance_m"] = df["edge_distance_m"].astype(float)
        X["edge_highway_ordinal"] = df["edge_highway_ordinal"].astype(float)
        X["edge_highway"] = df["edge_highway"].fillna("unknown").astype(str)
        X["tipo_via"] = df["tipo_via"].fillna("DESCONOCIDO").astype(str)
    return X


def make_ohe(with_network: bool) -> ColumnTransformer:
    cat = CAT_M2 + (CAT_NETWORK if with_network else [])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), cat),
    ], remainder="passthrough")


# ===========================================================================
# Metrics
# ===========================================================================
def ranking_metrics(y_true: np.ndarray, y_proba: np.ndarray, target: int = 3) -> dict:
    n = len(y_true)
    n_pos = int((y_true == target).sum())
    order = np.argsort(-y_proba[:, target])
    out = {"n_target": n_pos, "by_k": {}}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int((y_true[order[:k]] == target).sum())
        out["by_k"][f"k={int(pct*100)}%"] = {
            "k": int(k), "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_pos), 4),
            "lift": round((tp / k) / max(1e-12, n_pos / n), 3),
        }
    return out


def full_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    pred = y_proba.argmax(1)
    return {
        "kappa_q": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_true, pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - pred))),
        "pr_auc_fatal": float(average_precision_score(
            (y_true == 3).astype(int), y_proba[:, 3])),
        "pr_auc_macro": float(average_precision_score(y_true, y_proba, average="macro")),
        "roc_auc_macro": float(roc_auc_score(y_true, y_proba,
                                              multi_class="ovr", average="macro")),
        "recall": {n: float(r) for n, r in zip(
            LABEL_NAMES,
            recall_score(y_true, pred, labels=LABELS, average=None,
                         zero_division=0))},
        "confusion": {"labels": LABEL_NAMES,
                      "matrix": confusion_matrix(y_true, pred,
                                                  labels=LABELS).tolist()},
        "ranking_fatal": ranking_metrics(y_true, y_proba, target=3),
    }


def bootstrap_ranking(y_true, y_proba, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {k: [] for k in ("Lift@2", "Lift@5", "Recall@5", "Recall@10",
                              "PR_AUC_fatal")}
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if (yt == 3).sum() < 2:
            continue
        p = y_proba[idx]
        m = ranking_metrics(yt, p, 3)
        stats["Lift@2"].append(m["by_k"]["k=2%"]["lift"])
        stats["Lift@5"].append(m["by_k"]["k=5%"]["lift"])
        stats["Recall@5"].append(m["by_k"]["k=5%"]["recall"])
        stats["Recall@10"].append(m["by_k"]["k=10%"]["recall"])
        stats["PR_AUC_fatal"].append(
            average_precision_score((yt == 3).astype(int), p[:, 3]))
    return {
        k: {"median": float(np.median(v)),
            "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5))}
        for k, v in stats.items() if v
    }


# ===========================================================================
# Optuna objectives
# ===========================================================================
def make_lgbm_objective(Xt_train, y_train, groups):
    unique_groups = np.unique(groups)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "objective": "multiclass", "num_class": 4,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        fold_losses = []
        for i, g in enumerate(unique_groups):
            tr_mask = groups != g
            va_mask = groups == g
            if (y_train[va_mask] == 3).sum() == 0 and (y_train[va_mask] == 2).sum() == 0:
                # Fold without rare classes: log_loss undefined for class 2/3
                fold_losses.append(np.nan)
                continue
            X_tr, X_es, y_tr, y_es = train_test_split(
                Xt_train[tr_mask], y_train[tr_mask],
                test_size=0.15, random_state=42, stratify=y_train[tr_mask],
            )
            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                callbacks=[early_stopping(50, verbose=False),
                           log_evaluation(0)],
            )
            proba = model.predict_proba(Xt_train[va_mask])
            try:
                loss = log_loss(y_train[va_mask], proba, labels=[0, 1, 2, 3])
            except ValueError:
                loss = np.nan
            fold_losses.append(loss)
            valid = [v for v in fold_losses if not np.isnan(v)]
            if valid:
                trial.report(float(np.mean(valid)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        valid = [v for v in fold_losses if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 10.0
    return objective


def make_catboost_objective(Xt_train, y_train, groups):
    unique_groups = np.unique(groups)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        fold_losses = []
        for i, g in enumerate(unique_groups):
            tr_mask = groups != g
            va_mask = groups == g
            if (y_train[va_mask] == 3).sum() == 0 and (y_train[va_mask] == 2).sum() == 0:
                fold_losses.append(np.nan)
                continue
            X_tr, X_es, y_tr, y_es = train_test_split(
                Xt_train[tr_mask], y_train[tr_mask],
                test_size=0.15, random_state=42, stratify=y_train[tr_mask],
            )
            model = CatBoostClassifier(
                loss_function="MultiClass", classes_count=4,
                random_seed=42, verbose=0, thread_count=-1,
                allow_writing_files=False,
                od_type="Iter", od_wait=50,
                **params,
            )
            model.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
            proba = model.predict_proba(Xt_train[va_mask])
            try:
                loss = log_loss(y_train[va_mask], proba, labels=[0, 1, 2, 3])
            except ValueError:
                loss = np.nan
            fold_losses.append(loss)
            valid = [v for v in fold_losses if not np.isnan(v)]
            if valid:
                trial.report(float(np.mean(valid)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        valid = [v for v in fold_losses if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 10.0
    return objective


# ===========================================================================
# Train with Optuna, refit with best params on full train
# ===========================================================================
def optuna_search(objective, n_trials: int, name: str):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=1),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    n_pruned = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  [{name}] {n_trials} trials in {elapsed:.0f}s "
          f"({n_pruned} pruned), best log-loss={study.best_value:.4f}")
    return study


def refit_lgbm(Xt_train, y_train, best_params):
    params = dict(best_params)
    params.update({"objective": "multiclass", "num_class": 4,
                    "subsample_freq": 1, "random_state": 42,
                    "n_jobs": -1, "verbose": -1})
    model = LGBMClassifier(**params)
    model.fit(Xt_train, y_train)
    return model


def refit_catboost(Xt_train, y_train, best_params):
    params = dict(best_params)
    model = CatBoostClassifier(
        loss_function="MultiClass", classes_count=4,
        random_seed=42, verbose=0, thread_count=-1,
        allow_writing_files=False, **params,
    )
    model.fit(Xt_train, y_train)
    return model


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("Loading accidents + network...")
    acc = pd.read_parquet(OUT_DATA / "accidents_clean.parquet")
    net = pd.read_parquet(OUT_DATA / "accidents_with_network_features.parquet")
    base = set(acc.columns)
    net_extra = [c for c in net.columns if c not in base and c != "num_expediente"]
    df = acc.merge(net[["num_expediente"] + net_extra],
                    on="num_expediente", how="left")
    df = df.loc[df["edge_distance_m"].notna()
                & (df["edge_distance_m"] < 1000)].copy()
    df["split"] = build_splits_v2(df["year"].astype(int))
    df = df.loc[df["severity_code"].notna()].copy()
    print(f"  {len(df):,} accidents, splits={df['split'].value_counts().to_dict()}")

    y_all = df["severity_code"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    groups_train = df.loc[train_mask, "year"].astype(int).to_numpy()

    results = {}
    for variant, with_network in [("M2", False), ("M2+net", True)]:
        print(f"\n======== Variant: {variant} ========")
        X = build_features(df, with_network=with_network)
        ohe = make_ohe(with_network=with_network)
        Xt = ohe.fit_transform(X)
        print(f"  features after one-hot: {Xt.shape[1]}")
        Xt_train, Xt_val, Xt_test = Xt[train_mask], Xt[val_mask], Xt[test_mask]
        y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]

        # LGBM
        print(f"\n--- {variant}__lgbm (n_trials=60) ---")
        obj_l = make_lgbm_objective(Xt_train, y_train, groups_train)
        study_l = optuna_search(obj_l, n_trials=60, name=f"{variant}__lgbm")
        model_l = refit_lgbm(Xt_train, y_train, study_l.best_params)
        proba_val_l = model_l.predict_proba(Xt_val)
        proba_test_l = model_l.predict_proba(Xt_test)
        results[f"{variant}__lgbm"] = {
            "variant": variant, "model": "lgbm",
            "best_params": dict(study_l.best_params),
            "cv_log_loss": float(study_l.best_value),
            "val": full_metrics(y_val, proba_val_l),
            "test": full_metrics(y_test, proba_test_l),
            "bootstrap_test": bootstrap_ranking(y_test, proba_test_l, n_iter=500),
        }

        # CatBoost
        print(f"\n--- {variant}__catboost (n_trials=40) ---")
        obj_c = make_catboost_objective(Xt_train, y_train, groups_train)
        study_c = optuna_search(obj_c, n_trials=40, name=f"{variant}__catboost")
        model_c = refit_catboost(Xt_train, y_train, study_c.best_params)
        proba_val_c = model_c.predict_proba(Xt_val)
        proba_test_c = model_c.predict_proba(Xt_test)
        results[f"{variant}__catboost"] = {
            "variant": variant, "model": "catboost",
            "best_params": dict(study_c.best_params),
            "cv_log_loss": float(study_c.best_value),
            "val": full_metrics(y_val, proba_val_c),
            "test": full_metrics(y_test, proba_test_c),
            "bootstrap_test": bootstrap_ranking(y_test, proba_test_c, n_iter=500),
        }

    save_json(results, OUT_TABLES / "22_boosting_models.json")

    rows = []
    for key, r in results.items():
        m = r["test"]; rk = m["ranking_fatal"]["by_k"]
        rows.append({
            "key": key,
            "kappa_q": round(m["kappa_q"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "mae_ord": round(m["mae_ord"], 4),
            "pr_auc_fatal": round(m["pr_auc_fatal"], 5),
            "pr_auc_macro": round(m["pr_auc_macro"], 4),
            "rec_serious": round(m["recall"]["serious"], 3),
            "rec_fatal": round(m["recall"]["fatal"], 3),
            "Lift@2": rk["k=2%"]["lift"],
            "Lift@5": rk["k=5%"]["lift"],
            "Recall@5": rk["k=5%"]["recall"],
            "Recall@10": rk["k=10%"]["recall"],
        })
    comp = pd.DataFrame(rows)

    def _md(df_):
        h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
        s = "| " + " | ".join("---" for _ in df_.columns) + " |"
        rs = ["| " + " | ".join(str(v) for v in r) + " |"
              for r in df_.itertuples(index=False)]
        return "\n".join([h, s] + rs)

    best_key = max(results.keys(),
                   key=lambda k: (results[k]["test"]["kappa_q"],
                                   results[k]["test"]["pr_auc_fatal"]))

    sections = {
        "Boosting comparison (test = 2025)": _md(comp),
        "Best by kappa (then PR-AUC fatal)": best_key,
        "Best params": dumps_jsonable(results[best_key]["best_params"], indent=2),
        "Bootstrap CI (best)": dumps_jsonable(
            results[best_key]["bootstrap_test"], indent=2),
        "LR baseline from Fase 3": (
            "M4 (LR, no network): kappa=0.522, PR-AUC=0.0083, Lift@2=10.9, "
            "Recall@10=0.406\n"
            "M4+net (LR): kappa=0.526, PR-AUC=0.0089, Lift@2=10.9, Recall@10=0.500"
        ),
    }
    save_markdown_report(sections, Path("reports/22_boosting_models.md"))
    print(f"\nBest by kappa: {best_key}")


if __name__ == "__main__":
    main()