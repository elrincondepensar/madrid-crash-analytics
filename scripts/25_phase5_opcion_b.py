"""
Phase 5 · Iteration 4 — Opción B (3 clases: no_injury / minor / grave).

Target:
    0 = no_injury
    1 = minor
    2 = grave (serious ∪ fatal)

Configuraciones: {lgbm, catboost} × {none, smote, borderline, adasyn}
+ BART opcional (binario grave vs not_grave).

Preprocesado idéntico a M4 y script 23: SimpleImputer + OHE, fit en train.
Mismas features: M2 + numero + red + TE(edge_name) + TE(nombre_via).

Evalúa:
    - Clasificación 3 clases: kappa_q, f1_macro, MAE ordinal, recall por clase.
    - PR-AUC(grave), ROC-AUC(grave) — métricas principales.
    - Fatal como ranking secundario: usa P(grave) como score para Lift@k.

Run:
    python scripts/25_phase5_opcion_b.py

Paralelismo:
    Ajusta N_JOBS al principio: 8 si es el único script, 4 si comparte CPU con
    Opción A. Verifica en el log la velocidad del primer trial para calibrar.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ===========================================================================
# Configuración
# ===========================================================================
N_JOBS = 8  # 4 si compartes CPU con Opción A

FAST_SMOKE_TEST = False  # True → 2-3 trials por par (valida pipeline en ~10 min)

N_TRIALS = {
    ("lgbm", "none"):       3 if FAST_SMOKE_TEST else 15,
    ("lgbm", "smote"):      3 if FAST_SMOKE_TEST else 15,
    ("lgbm", "borderline"): 3 if FAST_SMOKE_TEST else 15,
    ("lgbm", "adasyn"):     3 if FAST_SMOKE_TEST else 15,
    ("catboost", "none"):       2 if FAST_SMOKE_TEST else 10,
    ("catboost", "smote"):      2 if FAST_SMOKE_TEST else 10,
    ("catboost", "borderline"): 2 if FAST_SMOKE_TEST else 10,
    ("catboost", "adasyn"):     2 if FAST_SMOKE_TEST else 10,
}

AUGMENTATIONS = ["none", "smote", "borderline", "adasyn"]
MODELS = ["lgbm", "catboost"]

# 3-class: 0=no_injury, 1=minor, 2=grave
N_CLASSES = 3
LABELS_3 = [0, 1, 2]
CLASS_NAMES = ["no_injury", "minor", "grave"]

# Grave es la clase minoritaria (~10%). Cap sintético a 20k por fold.
SAMPLING_STRATEGY_3 = {2: 20000}

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Features (idénticas a script 23)
NUMERIC_M2 = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "is_weekend",
    "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
    "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
    "coordenada_x_utm", "coordenada_y_utm",
    "numero",
]
CAT_M2 = ["tipo_accidente", "estado_meteorologico", "distrito"]
NUMERIC_NETWORK = [
    "edge_maxspeed", "edge_lanes", "edge_length_m",
    "edge_distance_m", "edge_highway_ordinal",
    "edge_maxspeed_is_imputed",
    "edge_name_te", "nombre_via_te",
]
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


def severity_to_3class(s: pd.Series) -> pd.Series:
    return s.map({0: 0, 1: 1, 2: 2, 3: 2}).astype(int)


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
    X["numero"] = df["numero"].astype(float)
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
        X["edge_maxspeed_is_imputed"] = df["edge_maxspeed_is_imputed"].astype(float)
        X["edge_highway"] = df["edge_highway"].fillna("unknown").astype(str)
        X["tipo_via"] = df["tipo_via"].fillna("DESCONOCIDO").astype(str)
    return X


def add_te_columns(X: pd.DataFrame, df: pd.DataFrame,
                    train_mask: np.ndarray, y_train: np.ndarray,
                    smoothing: float = 20.0) -> None:
    """Target encoding fit-on-train (target = P(grave))."""
    y_grave_train = (y_train == 2).astype(float)
    global_mean = float(y_grave_train.mean())

    def _fit_apply(col_src: str, col_te: str) -> None:
        train_vals = (df.loc[train_mask, col_src]
                        .fillna("__unknown__").astype(str))
        stats = (pd.DataFrame({"s": train_vals.values, "y": y_grave_train})
                   .groupby("s")["y"].agg(["mean", "count"]))
        smoothed = ((stats["mean"] * stats["count"]
                     + global_mean * smoothing)
                    / (stats["count"] + smoothing))
        all_vals = df[col_src].fillna("__unknown__").astype(str)
        X[col_te] = all_vals.map(smoothed).fillna(global_mean).values

    _fit_apply("edge_name", "edge_name_te")
    _fit_apply("nombre_via", "nombre_via_te")


def make_preprocessor(with_network: bool) -> ColumnTransformer:
    num = NUMERIC_M2 + (NUMERIC_NETWORK if with_network else [])
    cat = CAT_M2 + (CAT_NETWORK if with_network else [])
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False, min_frequency=20), cat),
    ])


# ===========================================================================
# Augmentation
# ===========================================================================
def apply_augmentation(name: str, X: np.ndarray, y: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    if name == "none":
        return X, y
    if not np.isfinite(X).all():
        raise ValueError(
            f"apply_augmentation({name}): X contains NaN/Inf. "
            "Check make_preprocessor imputation."
        )
    if name == "smote":
        sampler = SMOTE(sampling_strategy=SAMPLING_STRATEGY_3,
                        random_state=42, k_neighbors=5)
    elif name == "borderline":
        sampler = BorderlineSMOTE(sampling_strategy=SAMPLING_STRATEGY_3,
                                    random_state=42, kind="borderline-1",
                                    k_neighbors=5)
    elif name == "adasyn":
        sampler = ADASYN(sampling_strategy=SAMPLING_STRATEGY_3,
                          random_state=42, n_neighbors=5)
    else:
        raise ValueError(name)
    return sampler.fit_resample(X, y)


# ===========================================================================
# Metrics (3-class + fatal as secondary ranking)
# ===========================================================================
def fatal_ranking_from_grave_proba(y_true_code: np.ndarray,
                                     grave_proba: np.ndarray,
                                     target_fatal: int = 3) -> dict:
    """Usa P(grave) como score para rankear fatal."""
    n = len(y_true_code)
    n_fatal = int((y_true_code == target_fatal).sum())
    order = np.argsort(-grave_proba)
    out = {"n_fatal": n_fatal, "by_k": {}}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int((y_true_code[order[:k]] == target_fatal).sum())
        out["by_k"][f"k={int(pct*100)}%"] = {
            "k": k, "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_fatal), 4),
            "lift": round((tp / k) / max(1e-12, n_fatal / n), 3),
        }
    return out


def full_metrics_3class(y_true, y_proba) -> dict:
    pred = y_proba.argmax(1)
    return {
        "kappa_q": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_true, pred, labels=LABELS_3,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - pred))),
        "pr_auc_grave": float(average_precision_score(
            (y_true == 2).astype(int), y_proba[:, 2])),
        "roc_auc_grave": float(roc_auc_score(
            (y_true == 2).astype(int), y_proba[:, 2])),
        "pr_auc_macro": float(average_precision_score(
            y_true, y_proba, average="macro")),
        "recall": {n: float(r) for n, r in zip(
            CLASS_NAMES, recall_score(y_true, pred, labels=LABELS_3,
                                       average=None, zero_division=0))},
        "confusion": {"labels": CLASS_NAMES,
                      "matrix": confusion_matrix(y_true, pred,
                                                  labels=LABELS_3).tolist()},
    }


def bootstrap_3class(y_true, y_proba, n_iter=500, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pr_auc, rec_grave, prec_grave = [], [], []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if (yt == 2).sum() < 2:
            continue
        p = y_proba[idx]
        pr_auc.append(average_precision_score((yt == 2).astype(int), p[:, 2]))
        pred = p.argmax(1)
        rec_grave.append(float(recall_score(
            yt, pred, labels=[2], average="macro", zero_division=0)))
        tp = int(((pred == 2) & (yt == 2)).sum())
        fp = int(((pred == 2) & (yt != 2)).sum())
        prec_grave.append(tp / max(1, tp + fp))

    def _ci(vals):
        if not vals:
            return None
        arr = np.asarray(vals)
        return {"median": float(np.median(arr)),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5))}
    return {
        "pr_auc_grave": _ci(pr_auc),
        "recall_grave": _ci(rec_grave),
        "precision_grave": _ci( prec_grave),
    }


# ===========================================================================
# Model builders
# ===========================================================================
def suggest_params_lgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
    }


def suggest_params_catboost(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 500, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def build_model(model_type: str, params: dict) -> Any:
    if model_type == "lgbm":
        p = dict(params)
        p.update({
            "objective": "multiclass", "num_class": N_CLASSES,
            "random_state": 42, "n_jobs": N_JOBS, "verbose": -1,
        })
        return LGBMClassifier(**p)
    if model_type == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass", classes_count=N_CLASSES,
            random_seed=42, verbose=0, thread_count=N_JOBS,
            allow_writing_files=False, **params,
        )
    raise ValueError(model_type)


def fit_with_early_stop(model, model_type, X_tr, y_tr, X_es, y_es):
    if model_type == "lgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],
                   callbacks=[early_stopping(50, verbose=False),
                              log_evaluation(0)])
    else:
        if "od_type" not in model.get_params():
            model.set_params(od_type="Iter", od_wait=50)
        model.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
    return model


# ===========================================================================
# Optuna objective (con augmentation en el fold de train)
# ===========================================================================
def make_objective(model_type: str, augmentation: str,
                    Xt_train_full: np.ndarray, y_train: np.ndarray,
                    groups: np.ndarray):
    unique_groups = np.unique(groups)

    def objective(trial: optuna.Trial) -> float:
        if model_type == "lgbm":
            params = suggest_params_lgbm(trial)
        else:
            params = suggest_params_catboost(trial)

        fold_losses = []
        for i, g in enumerate(unique_groups):
            tr_mask = groups != g
            va_mask = groups == g

            X_tr_fold = Xt_train_full[tr_mask]
            y_tr_fold = y_train[tr_mask]

            # Augmentation SOLO en el fold de train
            X_tr_aug, y_tr_aug = apply_augmentation(
                augmentation, X_tr_fold, y_tr_fold
            )

            # Split interno para early stopping
            X_fit, X_es, y_fit, y_es = train_test_split(
                X_tr_aug, y_tr_aug, test_size=0.15,
                random_state=42, stratify=y_tr_aug,
            )
            model = build_model(model_type, params)
            fit_with_early_stop(model, model_type, X_fit, y_fit, X_es, y_es)

            proba = model.predict_proba(Xt_train_full[va_mask])
            try:
                loss = log_loss(y_train[va_mask], proba, labels=LABELS_3)
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


def tune_and_evaluate(model_type, augmentation,
                       Xt_train_full, Xt_val, Xt_test,
                       y_train, y_val, y_test,
                       y_code_val, y_code_test,
                       groups) -> dict:
    n_trials = N_TRIALS[(model_type, augmentation)]
    print(f"\n--- tuning {model_type} + {augmentation} "
          f"({n_trials} trials) ---")
    t0 = time.time()
    objective = make_objective(model_type, augmentation,
                                Xt_train_full, y_train, groups)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3,
                                            n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    n_pruned = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  tuning done in {elapsed:.0f}s ({n_pruned} pruned), "
          f"best log-loss={study.best_value:.4f}")

    # Refit con best params en train+augment completo (sin early stopping)
    best_params = dict(study.best_params)
    X_tr_aug, y_tr_aug = apply_augmentation(
        augmentation, Xt_train_full, y_train
    )
    print(f"  refit on augmented train ({len(y_tr_aug)} samples)...")
    model = build_model(model_type, best_params)
    t1 = time.time()
    model.fit(X_tr_aug, y_tr_aug)
    print(f"  refit in {time.time()-t1:.1f}s")

    p_val = model.predict_proba(Xt_val)
    p_test = model.predict_proba(Xt_test)
    m_val = full_metrics_3class(y_val, p_val)
    m_test = full_metrics_3class(y_test, p_test)
    boot = bootstrap_3class(y_test, p_test, n_iter=500)
    fatal_val = fatal_ranking_from_grave_proba(y_code_val, p_val[:, 2])
    fatal_test = fatal_ranking_from_grave_proba(y_code_test, p_test[:, 2])

    print(f"  TEST kappa={m_test['kappa_q']:.3f}  "
          f"PR-AUC(grave)={m_test['pr_auc_grave']:.4f}  "
          f"rec_grave={m_test['recall']['grave']:.3f}  "
          f"fatal_Lift@2={fatal_test['by_k']['k=2%']['lift']:.2f}")

    return {
        "model": model_type, "augmentation": augmentation,
        "n_trials": n_trials,
        "tuning_seconds": round(elapsed, 1),
        "cv_log_loss": float(study.best_value),
        "best_params": best_params,
        "val": m_val, "test": m_test,
        "fatal_ranking_val": fatal_val,
        "fatal_ranking_test": fatal_test,
        "bootstrap_test": boot,
    }


# ===========================================================================
# BART (opcional, binario grave vs not_grave)
# ===========================================================================
def run_bart_binary(Xt_train_full, Xt_test, y_train, y_test,
                     n_trees=50, n_samples=200) -> dict | None:
    """BART binario como comparación bayesiana. Ignora si pymc-bart no está."""
    try:
        import pymc as pm
        from pymc_bart import BART
    except ImportError:
        print("  [BART] pymc-bart no instalado — saltando. "
              "pip install pymc-bart")
        return None

    print(f"  [BART] fitting on train ({len(y_train)} samples, "
          f"{n_trees} trees, {n_samples} samples)...")
    y_bin = (y_train == 2).astype(int)
    t0 = time.time()
    with pm.Model() as model:
        X_data = pm.Data("X", Xt_train_full.astype(float))
        mu = pm.BART("mu", X_data, y_bin, m=n_trees, alpha=0.95, beta=2.0)
        p = pm.Deterministic("p", pm.math.sigmoid(mu))
        pm.Bernoulli("y", p=p, observed=y_bin)
        idata = pm.sample(n_samples, tune=200, chains=2, cores=1,
                           progressbar=False, random_seed=42)
    elapsed = time.time() - t0
    print(f"  [BART] sampling done in {elapsed:.0f}s")

    # Predicción sobre test usando el modelo entrenado
    with pm.Model() as predict_model:
        X_data = pm.Data("X", Xt_test.astype(float))
        mu = pm.BART("mu", X_data, y_bin, m=n_trees, alpha=0.95, beta=2.0)
        p = pm.Deterministic("p", pm.math.sigmoid(mu))
        pm.Bernoulli("y", p=p, shape=len(y_test))
        # Reutilizamos el BART del modelo entrenado reasignándolo
        # Nota: técnica simplificada; el modelo predictivo se reconstruye.
        # Para producción, guardar el BART entrenado.

    # Score = P(grave) posterior mean
    grave_proba_test = idata.posterior["p"].mean(axis=(0, 1)).values
    # score para fatal
    fatal_test = fatal_ranking_from_grave_proba(
        y_test, grave_proba_test.reshape(-1, 1).repeat(3, axis=1).reshape(-1, 3)  # dummy
    ) if False else fatal_ranking_from_grave_proba(
        y_test, np.column_stack([1 - grave_proba_test,
                                  np.zeros_like(grave_proba_test),
                                  grave_proba_test])
    )
    pr_auc = float(average_precision_score((y_test == 2).astype(int),
                                             grave_proba_test))
    return {
        "model": "bart",
        "augmentation": "none",
        "n_trees": n_trees,
        "n_samples": n_samples,
        "fit_seconds": round(elapsed, 1),
        "test_pr_auc_grave": round(pr_auc, 4),
        "fatal_ranking_test": fatal_test,
    }


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
    df["y3"] = severity_to_3class(df["severity_code"])
    print(f"  {len(df):,} accidents.")
    print(f"  3-class counts: {df['y3'].value_counts().sort_index().to_dict()}")
    print(f"  4-class counts: "
          f"{df['severity_code'].value_counts().sort_index().to_dict()}")
    print(f"  splits: {df['split'].value_counts().to_dict()}")

    y_all = df["y3"].astype(int).to_numpy()
    y_code_all = df["severity_code"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    y_code_val, y_code_test = (y_code_all[val_mask], y_code_all[test_mask])
    groups_train = df.loc[train_mask, "year"].astype(int).to_numpy()

    print("\nBuilding features (M2 + red + TE)...")
    X = build_features(df, with_network=True)
    add_te_columns(X, df, train_mask, y_train)
    pre = make_preprocessor(with_network=True)
    Xt_train_full = pre.fit_transform(X.loc[train_mask])
    Xt_val = pre.transform(X.loc[val_mask])
    Xt_test = pre.transform(X.loc[test_mask])
    print(f"  features after preprocess: {Xt_train_full.shape[1]}")
    print(f"  train={len(Xt_train_full)}, val={len(Xt_val)}, "
          f"test={len(Xt_test)}")

    results = {}
    for model_type in MODELS:
        for aug in AUGMENTATIONS:
            key = f"{model_type}__{aug}"
            try:
                r = tune_and_evaluate(
                    model_type=model_type, augmentation=aug,
                    Xt_train_full=Xt_train_full,
                    Xt_val=Xt_val, Xt_test=Xt_test,
                    y_train=y_train, y_val=y_val, y_test=y_test,
                    y_code_val=y_code_val, y_code_test=y_code_test,
                    groups=groups_train,
                )
                results[key] = r
                save_json(results, OUT_TABLES / "25_opcion_b.json")
            except KeyboardInterrupt:
                print("\n  Interrumpido. Guardando parcial...")
                save_json(results, OUT_TABLES / "25_opcion_b.json")
                raise
            except Exception as exc:
                print(f"  ERROR en {key}: {exc}")
                results[key] = {"error": str(exc)}

    # ---- Resumen ----
    valid = {k: r for k, r in results.items() if "error" not in r}
    if not valid:
        print("No hay resultados válidos.")
        return

    rows = []
    for key, r in valid.items():
        m = r["test"]
        fk = r["fatal_ranking_test"]["by_k"]
        rows.append({
            "key": key,
            "kappa_q": round(m["kappa_q"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "mae_ord": round(m["mae_ord"], 4),
            "pr_auc_grave": round(m["pr_auc_grave"], 4),
            "roc_auc_grave": round(m["roc_auc_grave"], 4),
            "rec_grave": round(m["recall"]["grave"], 3),
            "rec_minor": round(m["recall"]["minor"], 3),
            "rec_no_inj": round(m["recall"]["no_injury"], 3),
            "fatal_Lift@2": fk["k=2%"]["lift"],
            "fatal_Lift@5": fk["k=5%"]["lift"],
            "fatal_Recall@10": fk["k=10%"]["recall"],
        })
    comp = pd.DataFrame(rows).sort_values("pr_auc_grave", ascending=False)

    best_key = max(valid.keys(), key=lambda k: (
        valid[k]["test"]["pr_auc_grave"],
        valid[k]["test"]["kappa_q"],
    ))

    def _md(df_):
        h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
        s = "| " + " | ".join("---" for _ in df_.columns) + " |"
        rs = ["| " + " | ".join(str(v) for v in r) + " |"
              for r in df_.itertuples(index=False)]
        return "\n".join([h, s] + rs)

    sections = {
        "Opción B — comparison (test=2025) sorted by PR-AUC(grave)":
            _md(comp),
        "Best by PR-AUC(grave)": best_key,
        "Best params": dumps_jsonable(
            valid[best_key]["best_params"], indent=2),
        "Bootstrap CI (best)": dumps_jsonable(
            valid[best_key]["bootstrap_test"], indent=2),
        "Fatal ranking on best (P(grave) as score)":
            dumps_jsonable(valid[best_key]["fatal_ranking_test"], indent=2),
        "Tuning time per combo (s)": dumps_jsonable(
            {k: r["tuning_seconds"] for k, r in valid.items()}, indent=2
        ),
        "Reference: Opción A (4 clases)": (
            "Best LGBM: kappa 0.552, PR-AUC(fatal) 0.0053\n"
            "Best CatBoost: kappa 0.551, PR-AUC(fatal) 0.0081\n"
            "LR baseline: kappa 0.522, PR-AUC(fatal) 0.0083"
        ),
        "Notes": (
            "Target: 0=no_injury, 1=minor, 2=grave (serious ∪ fatal).\n"
            "Preprocesado M4: SimpleImputer(median, indicator) + OHE. "
            "Fit solo en train.\n"
            "Augmentation dentro de cada fold de LOGO.\n"
            "fatal se evalúa como ranking usando P(grave) como score."
        ),
    }
    save_markdown_report(sections, Path("reports/25_opcion_b.md"))
    print(f"\nBest: {best_key}")
    print("Done. See reports/25_opcion_b.md")


if __name__ == "__main__":
    main()