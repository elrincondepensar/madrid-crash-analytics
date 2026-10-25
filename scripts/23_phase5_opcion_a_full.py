"""
Phase 5 · Iteration 3 — Opción A (4 clases), features completas, tuning por
(modelo, augmentation).

Diseño:
  - Features completas: M2 + numero + red (maxspeed, lanes, distance, highway,
    imputed flag) + TE(edge_name) + TE(nombre_via).
  - 8 combinaciones = {lgbm, catboost} × {none, smote, borderline, adasyn}.
  - Cada combinación tiene su propia búsqueda Optuna (n_trials configurables).
  - La augmentation se aplica DENTRO de cada fold de LOGO(train-years) para
    evitar fuga; nunca se aplica a val/test.
  - Early stopping interno (15 % stratified split del fold de train, dentro
    de Optuna), y refit final con best params sin early stopping.

Tiempos estimados (config por defecto):
  - LGBM: 4 augmentations × 15 trials × 4 folds ≈ 60-90 min
  - CatBoost: 4 augmentations × 10 trials × 4 folds ≈ 90-120 min
  - Refits + evaluación: ~15 min
  - Total: ~3-4 h

Para smoke test rápido (~10 min):
  - Cambiar FAST_SMOKE_TEST = True

Run:
    pip install lightgbm catboost optuna imbalanced-learn
    python scripts/23_phase5_opcion_a_full.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ===========================================================================
# Configuración
# ===========================================================================
FAST_SMOKE_TEST = False  # True → 3 trials por par para validar el pipeline

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

# SMOTE sampling_strategy: cap minoritaria-serious a 12k, fatal a 2k
SAMPLING_STRATEGY = {3: 2000, 2: 12000}

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

LABELS_4 = [0, 1, 2, 3]
LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]

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
    """Fit-on-train target encoding (target = P(grave)). Idéntico al script 23 v1."""
    y_grave_train = (y_train >= 2).astype(float)
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
    """
    Mismo patrón que el preprocesado de M4 (script 20):
      - num: SimpleImputer(median, add_indicator=True)
      - cat: OneHotEncoder(handle_unknown="ignore", min_frequency=20)

    Sin StandardScaler: LGBM/CatBoost son invariantes a escala y el scaler
    solo añadiría cómputo. El imputador SÍ es necesario porque SMOTE
    y variantes no aceptan NaN.
    """
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
# Augmentation (aplicada a un ndarray de train; nunca a val/test)
# ===========================================================================
def make_sampler(name: str, seed: int = 42):
    if name == "smote":
        return SMOTE(sampling_strategy=SAMPLING_STRATEGY, random_state=seed,
                     k_neighbors=5)
    if name == "borderline":
        return BorderlineSMOTE(sampling_strategy=SAMPLING_STRATEGY,
                                random_state=seed, kind="borderline-1",
                                k_neighbors=5)
    if name == "adasyn":
        return ADASYN(sampling_strategy=SAMPLING_STRATEGY,
                       random_state=seed, n_neighbors=5)
    raise ValueError(name)


def apply_augmentation(name: str, X: np.ndarray, y: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    if name == "none":
        return X, y
    if not np.isfinite(X).all():
        raise ValueError(
            f"apply_augmentation({name}): X contains NaN/Inf. "
            "Check make_ohe imputation."
        )
    sampler = make_sampler(name)
    return sampler.fit_resample(X, y)


# ===========================================================================
# Metrics
# ===========================================================================
def ranking_metrics(y_true, y_proba, target: int = 3) -> dict:
    n = len(y_true)
    n_pos = int((y_true == target).sum())
    order = np.argsort(-y_proba[:, target])
    out = {"n_target": n_pos, "by_k": {}}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int((y_true[order[:k]] == target).sum())
        out["by_k"][f"k={int(pct*100)}%"] = {
            "k": k, "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_pos), 4),
            "lift": round((tp / k) / max(1e-12, n_pos / n), 3),
        }
    return out


def full_metrics(y_true, y_proba) -> dict:
    pred = y_proba.argmax(1)
    return {
        "kappa_q": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_true, pred, labels=LABELS_4,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - pred))),
        "pr_auc_fatal": float(average_precision_score(
            (y_true == 3).astype(int), y_proba[:, 3])),
        "pr_auc_macro": float(average_precision_score(
            y_true, y_proba, average="macro")),
        "roc_auc_macro": float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro")),
        "recall": {n: float(r) for n, r in zip(
            LABEL_NAMES, recall_score(y_true, pred, labels=LABELS_4,
                                       average=None, zero_division=0))},
        "confusion": {"labels": LABEL_NAMES,
                      "matrix": confusion_matrix(y_true, pred,
                                                  labels=LABELS_4).tolist()},
        "ranking_fatal": ranking_metrics(y_true, y_proba, target=3),
    }


def bootstrap_ranking(y_true, y_proba, n_iter=500, seed=42) -> dict:
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
            "objective": "multiclass", "num_class": 4,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        })
        return LGBMClassifier(**p)
    if model_type == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass", classes_count=4,
            random_seed=42, verbose=0, thread_count=-1,
            allow_writing_files=False, **params,
        )
    raise ValueError(model_type)


def fit_with_early_stop(model, model_type, X_tr, y_tr, X_es, y_es):
    if model_type == "lgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],
                   callbacks=[early_stopping(50, verbose=False),
                              log_evaluation(0)])
    else:  # catboost
        # Añadir overfitting detector si no está ya en params
        if "od_type" not in model.get_params():
            model.set_params(od_type="Iter", od_wait=50)
        model.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
    return model


# ===========================================================================
# Optuna objective per (model_type, augmentation)
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

            # Augmentation SOLO en el fold de entrenamiento (no leakage)
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
                loss = log_loss(y_train[va_mask], proba, labels=LABELS_4)
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


def tune_and_evaluate(model_type: str, augmentation: str,
                       Xt_train_full: np.ndarray, Xt_val: np.ndarray,
                       Xt_test: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray,
                       y_test: np.ndarray,
                       groups: np.ndarray) -> dict:
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
    model.fit(X_tr_aug, y_tr_aug)  # sin early stopping en refit final
    print(f"  refit in {time.time()-t1:.1f}s")

    p_val = model.predict_proba(Xt_val)
    p_test = model.predict_proba(Xt_test)
    m_val = full_metrics(y_val, p_val)
    m_test = full_metrics(y_test, p_test)
    boot = bootstrap_ranking(y_test, p_test, n_iter=500)
    rk = m_test["ranking_fatal"]["by_k"]
    print(f"  TEST kappa={m_test['kappa_q']:.3f}  "
          f"PR-AUC(fatal)={m_test['pr_auc_fatal']:.4f}  "
          f"Lift@2={rk['k=2%']['lift']:.2f}  "
          f"Recall@10={rk['k=10%']['recall']:.3f}")

    return {
        "model": model_type,
        "augmentation": augmentation,
        "n_trials": n_trials,
        "tuning_seconds": round(elapsed, 1),
        "cv_log_loss": float(study.best_value),
        "best_params": best_params,
        "val": m_val,
        "test": m_test,
        "bootstrap_test": boot,
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
    print(f"  {len(df):,} accidents.")
    print(f"  class counts: "
          f"{df['severity_code'].value_counts().sort_index().to_dict()}")
    print(f"  splits: {df['split'].value_counts().to_dict()}")

    y_all = df["severity_code"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    groups_train = df.loc[train_mask, "year"].astype(int).to_numpy()

    print("\nBuilding features (M2 + red + TE)...")
    X = build_features(df, with_network=True)
    add_te_columns(X, df, train_mask, y_train)
    pre = make_preprocessor(with_network=True)
    # fit SOLO en train (como en M4), transform en val/test
    Xt_train_full = pre.fit_transform(X.loc[train_mask])
    Xt_val = pre.transform(X.loc[val_mask])
    Xt_test = pre.transform(X.loc[test_mask])
    print(f"  features after one-hot+impute: {Xt_train_full.shape[1]}")
    print(f"  train={len(Xt_train_full)}, val={len(Xt_val)}, "
          f"test={len(Xt_test)}")
    print(f"  train years: {sorted(set(groups_train))}")

    # ---- Estimación de tiempo total ----
    est_trials = sum(N_TRIALS.values())
    print(f"\n  Total trials planificados: {est_trials} "
          f"(4 folds each).")
    if FAST_SMOKE_TEST:
        print("  *** FAST_SMOKE_TEST activo: ~10 min total ***")
    else:
        print("  *** Full run: ~3-4 h. Puedes parar con Ctrl+C. ***")

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
                    groups=groups_train,
                )
                results[key] = r
                # Guardado incremental (por si el proceso se corta)
                save_json(results, OUT_TABLES / "23_opcion_a_full.json")
            except KeyboardInterrupt:
                print("\n  Interrumpido por el usuario. Guardando parcial...")
                save_json(results, OUT_TABLES / "23_opcion_a_full.json")
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
    comp = pd.DataFrame(rows).sort_values("pr_auc_fatal", ascending=False)

    best_key = max(valid.keys(), key=lambda k: (
        valid[k]["test"]["pr_auc_fatal"],
        valid[k]["test"]["ranking_fatal"]["by_k"]["k=2%"]["lift"],
    ))

    def _md(df_):
        h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
        s = "| " + " | ".join("---" for _ in df_.columns) + " |"
        rs = ["| " + " | ".join(str(v) for v in r) + " |"
              for r in df_.itertuples(index=False)]
        return "\n".join([h, s] + rs)

    sections = {
        "Comparison (test=2025, 32 fatales) — sorted by PR-AUC(fatal)":
            _md(comp),
        "Best model": best_key,
        "Best params": dumps_jsonable(valid[best_key]["best_params"], indent=2),
        "Bootstrap CI (best)": dumps_jsonable(
            valid[best_key]["bootstrap_test"], indent=2),
        "Tuning time per combo (s)": dumps_jsonable(
            {k: r["tuning_seconds"] for k, r in valid.items()}, indent=2
        ),
        "Reference (from iter 21, LR baseline)": (
            "LR M4+net: kappa=0.526, PR-AUC(fatal)=0.0089, Lift@2=10.9"
        ),
        "Notes": (
            "Augmentation applied inside each fold (no leakage).\n"
            "SMOTE strategy caps: serious→12k, fatal→2k.\n"
            "Refit final sin early stopping (usa best params de Optuna).\n"
            "Tuning time real por combo está en 'Tuning time per combo'."
        ),
    }
    save_markdown_report(sections, Path("reports/23_opcion_a_full.md"))
    print(f"\nBest: {best_key}")
    print("Done. See reports/23_opcion_a_full.md")


if __name__ == "__main__":
    main()