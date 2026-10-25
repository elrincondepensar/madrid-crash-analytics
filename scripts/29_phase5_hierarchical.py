"""
Phase 5 · Iteration 9 — Jerarquía natural de severidad, con LGBM y LR.

Estructura jerárquica:
    Nivel 1: ¿injured?          → no_injury | resto
    Nivel 2: ¿grave|injured?    → minor | (serious ∪ fatal)
    Nivel 3: ¿fatal|grave?      → serious | fatal

Composición por Bayes (partición exacta de las 4 clases):
    P(no_injury) = 1 − p_injured
    P(minor)     = p_injured · (1 − p_grave|injured)
    P(serious)   = p_injured · p_grave|injured · (1 − p_fatal|grave)
    P(fatal)     = p_injured · p_grave|injured · p_fatal|grave

Modelos comparados (4 en total):
    - flat_4class_LGBM
    - flat_4class_LR
    - hierarchical_LGBM
    - hierarchical_LR

Evaluación (test=2025): argmax y thresholded (por clase, tuneado en val).
Reporta: kappa_q, F1-macro, MAE ordinal, recall por clase, PR-AUC(fatal),
PR-AUC(grave), Lift@k(fatal/grave), bootstrap CI.

Preprocesado: imputer+OHE (árboles), +scaler (LR). Fit solo en train.

Run:
    python scripts/29_phase5_hierarchical.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# LGBM params (best de script 28 para injured binario)
LGBM_PARAMS = {
    "n_estimators": 848, "learning_rate": 0.010210451114512976,
    "num_leaves": 76, "max_depth": 11, "min_child_samples": 31,
    "reg_alpha": 0.007734482156646798,
    "reg_lambda": 0.0030162092627967776,
    "colsample_bytree": 0.9442922333025374,
    "subsample": 0.7350460685614512,
    "subsample_freq": 1,
}

# Política de class_weight (consistente con script 28)
LGBM_CLASS_WEIGHT = None      # script 28 funcionó sin
LR_CLASS_WEIGHT = "balanced"  # script 28 funcionó con

LR_C = 1.0
N_JOBS = 8
RANDOM_SEED = 42
CLASS_NAMES = ["no_injury", "minor", "serious", "fatal"]
LABELS_4 = [0, 1, 2, 3]

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
def build_splits_v2(year):
    return pd.Series(
        np.where(year <= 2022, "train",
                 np.where(year <= 2024, "val", "test")),
        index=year.index,
    )


def build_features(df):
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
    X["edge_maxspeed"] = df["edge_maxspeed"].astype(float)
    X["edge_lanes"] = df["edge_lanes"].astype(float)
    X["edge_length_m"] = df["edge_length_m"].astype(float)
    X["edge_distance_m"] = df["edge_distance_m"].astype(float)
    X["edge_highway_ordinal"] = df["edge_highway_ordinal"].astype(float)
    X["edge_maxspeed_is_imputed"] = df["edge_maxspeed_is_imputed"].astype(float)
    X["edge_highway"] = df["edge_highway"].fillna("unknown").astype(str)
    X["tipo_via"] = df["tipo_via"].fillna("DESCONOCIDO").astype(str)
    return X


def add_te_columns(X, df, train_mask, y_train_code, smoothing=20.0):
    y_grave = (y_train_code >= 2).astype(float)
    global_mean = float(y_grave.mean())

    def _fit_apply(col_src, col_te):
        train_vals = (df.loc[train_mask, col_src]
                        .fillna("__unknown__").astype(str))
        stats = (pd.DataFrame({"s": train_vals.values, "y": y_grave})
                   .groupby("s")["y"].agg(["mean", "count"]))
        smoothed = ((stats["mean"] * stats["count"]
                     + global_mean * smoothing)
                    / (stats["count"] + smoothing))
        all_vals = df[col_src].fillna("__unknown__").astype(str)
        X[col_te] = all_vals.map(smoothed).fillna(global_mean).values

    _fit_apply("edge_name", "edge_name_te")
    _fit_apply("nombre_via", "nombre_via_te")


def make_preprocessor(with_scaler):
    num = NUMERIC_M2 + NUMERIC_NETWORK
    cat = CAT_M2 + CAT_NETWORK
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if with_scaler:
        steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False, min_frequency=20), cat),
    ])


# ===========================================================================
# Modelos
# ===========================================================================
def make_binary_model(model_type, level):
    """level: 1 (injured), 2 (grave|injured), 3 (fatal|grave)."""
    if model_type == "lgbm":
        p = dict(LGBM_PARAMS)
        p.update({"objective": "binary", "random_state": RANDOM_SEED,
                  "n_jobs": N_JOBS, "verbose": -1})
        if LGBM_CLASS_WEIGHT:
            p["class_weight"] = LGBM_CLASS_WEIGHT
        return LGBMClassifier(**p)
    else:  # lr
        return LogisticRegression(
            C=LR_C, class_weight=LR_CLASS_WEIGHT,
            max_iter=3000, random_state=RANDOM_SEED, solver="lbfgs",
        )


def make_multiclass_model(model_type):
    if model_type == "lgbm":
        p = dict(LGBM_PARAMS)
        p.update({"objective": "multiclass", "num_class": 4,
                  "random_state": RANDOM_SEED, "n_jobs": N_JOBS,
                  "verbose": -1})
        if LGBM_CLASS_WEIGHT:
            p["class_weight"] = LGBM_CLASS_WEIGHT
        return LGBMClassifier(**p)
    else:
        return LogisticRegression(
            C=LR_C, class_weight=LR_CLASS_WEIGHT,
            max_iter=3000, random_state=RANDOM_SEED, solver="lbfgs",
        )


def fit_flat(Xt_train, y_train, model_type):
    print(f"  [{model_type}] flat 4-class...", end="", flush=True)
    t0 = time.time()
    m = make_multiclass_model(model_type)
    m.fit(Xt_train, y_train)
    print(f" done in {time.time()-t0:.1f}s", flush=True)
    return m


def fit_hierarchy(Xt_train, y_train, model_type):
    """3 modelos binarios jerárquicos. Retorna dict de modelos."""
    models = {}

    # Nivel 1: injured
    y_lvl1 = (y_train >= 1).astype(int)
    print(f"  [{model_type}] L1 injured ({len(y_lvl1)} train, "
          f"{y_lvl1.sum()} pos)...", end="", flush=True)
    t0 = time.time()
    models["l1"] = make_binary_model(model_type, level=1)
    models["l1"].fit(Xt_train, y_lvl1)
    print(f" done in {time.time()-t0:.1f}s", flush=True)

    # Nivel 2: grave|injured — solo train[injured]
    mask_inj = y_train >= 1
    Xt_train_inj = Xt_train[mask_inj]
    y_lvl2 = (y_train[mask_inj] >= 2).astype(int)
    print(f"  [{model_type}] L2 grave|inj ({len(y_lvl2)} train, "
          f"{y_lvl2.sum()} pos)...", end="", flush=True)
    t0 = time.time()
    models["l2"] = make_binary_model(model_type, level=2)
    models["l2"].fit(Xt_train_inj, y_lvl2)
    print(f" done in {time.time()-t0:.1f}s", flush=True)

    # Nivel 3: fatal|grave — solo train[grave]
    mask_grave = y_train >= 2
    Xt_train_grave = Xt_train[mask_grave]
    y_lvl3 = (y_train[mask_grave] == 3).astype(int)
    print(f"  [{model_type}] L3 fatal|grave ({len(y_lvl3)} train, "
          f"{y_lvl3.sum()} pos)...", end="", flush=True)
    t0 = time.time()
    models["l3"] = make_binary_model(model_type, level=3)
    models["l3"].fit(Xt_train_grave, y_lvl3)
    print(f" done in {time.time()-t0:.1f}s", flush=True)

    return models


def predict_hierarchy(models, Xt):
    p1 = models["l1"].predict_proba(Xt)[:, 1]
    p2 = models["l2"].predict_proba(Xt)[:, 1]
    p3 = models["l3"].predict_proba(Xt)[:, 1]
    p_grave = p1 * p2
    p_no_inj = 1 - p1
    p_minor = p1 - p_grave
    p_serious = p_grave * (1 - p3)
    p_fatal = p_grave * p3
    probas = np.column_stack([
        np.clip(p_no_inj, 0, 1),
        np.clip(p_minor, 0, 1),
        np.clip(p_serious, 0, 1),
        np.clip(p_fatal, 0, 1),
    ])
    row_sum = probas.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return probas / row_sum, {"p_injured": p1, "p_grave_given_inj": p2,
                                "p_fatal_given_grave": p3}


# ===========================================================================
# Metrics
# ===========================================================================
def ranking(y_true, score, target=None, target_binary=False):
    n = len(y_true)
    y_pos = ((y_true >= 2) if target_binary
             else (y_true == target)).astype(int)
    n_pos = int(y_pos.sum())
    base = n_pos / n
    order = np.argsort(-score)
    out = {"n_target": n_pos, "base_rate": round(float(base), 5), "by_k": {}}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int(y_pos[order[:k]].sum())
        out["by_k"][f"k={int(pct*100)}%"] = {
            "k": k, "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_pos), 4),
            "lift": round((tp / k) / max(1e-12, base), 3),
        }
    return out


def metrics_4class(y_true, y_proba):
    pred = y_proba.argmax(1)
    return {
        "kappa_q": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_true, pred, labels=LABELS_4,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - pred))),
        "recall": {n: float(r) for n, r in zip(
            CLASS_NAMES, recall_score(y_true, pred, labels=LABELS_4,
                                       average=None, zero_division=0))},
        "precision": {n: float(p) for n, p in zip(
            CLASS_NAMES, precision_score(y_true, pred, labels=LABELS_4,
                                          average=None, zero_division=0))},
        "pr_auc_fatal": float(average_precision_score(
            (y_true == 3).astype(int), y_proba[:, 3])),
        "pr_auc_grave": float(average_precision_score(
            (y_true >= 2).astype(int), y_proba[:, 2] + y_proba[:, 3])),
        "roc_auc_macro": float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro")),
        "confusion": {"labels": CLASS_NAMES,
                      "matrix": confusion_matrix(
                          y_true, pred, labels=LABELS_4).tolist()},
        "ranking_fatal": ranking(y_true, y_proba[:, 3], target=3),
        "ranking_grave": ranking(
            y_true, y_proba[:, 2] + y_proba[:, 3], target_binary=True),
    }


def metrics_from_pred(y_true, y_pred):
    return {
        "kappa_q": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS_4,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - y_pred))),
        "recall": {n: float(r) for n, r in zip(
            CLASS_NAMES, recall_score(y_true, y_pred, labels=LABELS_4,
                                       average=None, zero_division=0))},
        "precision": {n: float(p) for n, p in zip(
            CLASS_NAMES, precision_score(y_true, y_pred, labels=LABELS_4,
                                          average=None, zero_division=0))},
        "confusion": {"labels": CLASS_NAMES,
                      "matrix": confusion_matrix(
                          y_true, y_pred, labels=LABELS_4).tolist()},
    }


def threshold_4class(y_val, y_proba_val, grid=100):
    thresholds = {0: 0.0}
    for k in (1, 2, 3):
        y_k = (y_val == k).astype(int)
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.01, 0.99, grid):
            yp = (y_proba_val[:, k] >= t).astype(int)
            if yp.sum() == 0:
                continue
            f1 = f1_score(y_k, yp, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
    return thresholds


def apply_thresholds(y_proba, thresholds):
    n = len(y_proba)
    scores = np.zeros((n, 4))
    for k in (1, 2, 3):
        scores[:, k] = y_proba[:, k] / max(1e-9, thresholds[k])
    pred = np.where(scores.max(axis=1) > 1.0, scores.argmax(axis=1), 0)
    return pred.astype(int)


def bootstrap_4class(y_true, y_proba, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s, kappas, pr_fatal, pr_grave, lifts2_fatal = [], [], [], [], []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, pp = y_true[idx], y_proba[idx]
        if (yt == 3).sum() < 2 or (yt >= 2).sum() < 2:
            continue
        pred = pp.argmax(1)
        f1s.append(f1_score(yt, pred, labels=LABELS_4,
                             average="macro", zero_division=0))
        kappas.append(cohen_kappa_score(yt, pred, weights="quadratic"))
        pr_fatal.append(average_precision_score(
            (yt == 3).astype(int), pp[:, 3]))
        pr_grave.append(average_precision_score(
            (yt >= 2).astype(int), pp[:, 2] + pp[:, 3]))
        k = max(1, int(np.ceil(n * 0.02)))
        order = np.argsort(-pp[:, 3])
        tp = int((yt[order[:k]] == 3).sum())
        base = max(1e-12, (yt == 3).sum() / n)
        lifts2_fatal.append((tp / k) / base)
    def _ci(vals):
        if not vals:
            return None
        arr = np.asarray(vals)
        return {"median": float(np.median(arr)),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5))}
    return {"f1_macro": _ci(f1s), "kappa_q": _ci(kappas),
            "pr_auc_fatal": _ci(pr_fatal),
            "pr_auc_grave": _ci(pr_grave),
            "lift@2_fatal": _ci(lifts2_fatal)}


def lvl3_quality_on_grave(models, Xt_test, y_test):
    """PR-AUC(fatal|grave) en test[grave] — el techo real del jerárquico."""
    mask = y_test >= 2
    if mask.sum() == 0:
        return None
    y_bin = (y_test[mask] == 3).astype(int)
    p3 = models["l3"].predict_proba(Xt_test[mask])[:, 1]
    return {
        "n_test_grave": int(mask.sum()),
        "n_fatal_in_grave": int(y_bin.sum()),
        "base_rate_fatal_in_grave": round(float(y_bin.mean()), 5),
        "pr_auc_fatal_given_grave": round(
            float(average_precision_score(y_bin, p3)), 4),
    }


# ===========================================================================
# Markdown helpers
# ===========================================================================
def _md(df_):
    h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
    s = "| " + " | ".join("---" for _ in df_.columns) + " |"
    rs = ["| " + " | ".join(str(v) for v in r) + " |"
          for r in df_.itertuples(index=False)]
    return "\n".join([h, s] + rs)


def _cm_to_md(cm):
    df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
    header = "| | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + str(idx) + " | " + " | ".join(str(v) for v in row) + " |"
            for idx, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


# ===========================================================================
# Main
# ===========================================================================
def main():
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
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        counts = sub["severity_code"].value_counts().sort_index().to_dict()
        print(f"  {split}: {len(sub):>6,} | {counts}")

    y_all = df["severity_code"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    X_train_df = df.loc[train_mask].copy()
    X_val_df = df.loc[val_mask].copy()
    X_test_df = df.loc[test_mask].copy()

    print("\nBuilding features...")
    X = build_features(df)
    add_te_columns(X, df, train_mask, y_train)
    X_train_df = X.loc[train_mask]
    X_val_df = X.loc[val_mask]
    X_test_df = X.loc[test_mask]

    # Preprocesadores
    pre_tree = make_preprocessor(with_scaler=False)
    Xt_tree_train = pre_tree.fit_transform(X_train_df)
    Xt_tree_val = pre_tree.transform(X_val_df)
    Xt_tree_test = pre_tree.transform(X_test_df)

    pre_lr = make_preprocessor(with_scaler=True)
    Xt_lr_train = pre_lr.fit_transform(X_train_df)
    Xt_lr_val = pre_lr.transform(X_val_df)
    Xt_lr_test = pre_lr.transform(X_test_df)

    print(f"  features tree: {Xt_tree_train.shape[1]}")
    print(f"  features lr:   {Xt_lr_train.shape[1]}\n")

    results = {}

    for model_type in ("lgbm", "lr"):
        print(f"\n{'='*60}")
        print(f"MODEL TYPE: {model_type.upper()}")
        print(f"{'='*60}")
        Xt_train = (Xt_tree_train if model_type == "lgbm" else Xt_lr_train)
        Xt_val = (Xt_tree_val if model_type == "lgbm" else Xt_lr_val)
        Xt_test = (Xt_tree_test if model_type == "lgbm" else Xt_lr_test)

        # --- Flat ---
        print(f"\n--- {model_type} flat 4-class ---")
        flat = fit_flat(Xt_train, y_train, model_type)
        p_val_flat = flat.predict_proba(Xt_val)
        p_test_flat = flat.predict_proba(Xt_test)
        thr_flat = threshold_4class(y_val, p_val_flat)
        res_flat = {
            "argmax_test": metrics_4class(y_test, p_test_flat),
            "thresholds": {str(k): round(v, 3) for k, v in thr_flat.items()},
            "thresholded_test": metrics_from_pred(
                y_test, apply_thresholds(p_test_flat, thr_flat)),
            "bootstrap_test": bootstrap_4class(y_test, p_test_flat),
        }
        results[f"flat_4class_{model_type}"] = res_flat
        m = res_flat["argmax_test"]
        print(f"  argmax: kappa={m['kappa_q']:.3f} f1={m['f1_macro']:.3f} "
              f"PR-AUC(fatal)={m['pr_auc_fatal']:.5f} "
              f"Lift@2={m['ranking_fatal']['by_k']['k=2%']['lift']:.2f}")

        # --- Hierarchy ---
        print(f"\n--- {model_type} hierarchical ---")
        hier = fit_hierarchy(Xt_train, y_train, model_type)
        p_val_hier, _ = predict_hierarchy(hier, Xt_val)
        p_test_hier, _ = predict_hierarchy(hier, Xt_test)
        thr_hier = threshold_4class(y_val, p_val_hier)
        res_hier = {
            "argmax_test": metrics_4class(y_test, p_test_hier),
            "thresholds": {str(k): round(v, 3) for k, v in thr_hier.items()},
            "thresholded_test": metrics_from_pred(
                y_test, apply_thresholds(p_test_hier, thr_hier)),
            "bootstrap_test": bootstrap_4class(y_test, p_test_hier),
            "lvl3_quality_on_grave": lvl3_quality_on_grave(
                hier, Xt_test, y_test),
        }
        results[f"hierarchical_{model_type}"] = res_hier
        m = res_hier["argmax_test"]
        print(f"  argmax: kappa={m['kappa_q']:.3f} f1={m['f1_macro']:.3f} "
              f"PR-AUC(fatal)={m['pr_auc_fatal']:.5f} "
              f"Lift@2={m['ranking_fatal']['by_k']['k=2%']['lift']:.2f}")
        if res_hier["lvl3_quality_on_grave"]:
            q = res_hier["lvl3_quality_on_grave"]
            print(f"  L3 en test[grave]: PR-AUC(fatal|grave)="
                  f"{q['pr_auc_fatal_given_grave']:.4f} "
                  f"(base={q['base_rate_fatal_in_grave']:.4f})")

    save_json(results, OUT_TABLES / "29_hierarchical.json")

    # ===========================================================
    # Resumen
    # ===========================================================
    rows = []
    for name, r in results.items():
        for mode in ("argmax_test", "thresholded_test"):
            m = r[mode]
            rows.append({
                "model": name,
                "mode": mode.replace("_test", ""),
                "kappa_q": round(m["kappa_q"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "mae_ord": round(m["mae_ord"], 4),
                "rec_minor": round(m["recall"]["minor"], 3),
                "rec_serious": round(m["recall"]["serious"], 3),
                "rec_fatal": round(m["recall"]["fatal"], 3),
            })
    comp = pd.DataFrame(rows)

    # Ranking tables
    rank_rows = []
    for name, r in results.items():
        for target, key in (("fatal", "ranking_fatal"),
                             ("grave", "ranking_grave")):
            m = r["argmax_test"][key]
            bk = m["by_k"]
            pr_auc = (r["argmax_test"]["pr_auc_fatal"] if target == "fatal"
                      else r["argmax_test"]["pr_auc_grave"])
            rank_rows.append({
                "model": name, "target": target,
                "pr_auc": round(pr_auc, 5),
                "lift@2": bk["k=2%"]["lift"],
                "lift@5": bk["k=5%"]["lift"],
                "recall@10": bk["k=10%"]["recall"],
            })
    rank_df = pd.DataFrame(rank_rows)

    # Level 3 quality (aislamiento)
    lvl3_rows = []
    for name, r in results.items():
        q = r.get("lvl3_quality_on_grave")
        if q is None:
            continue
        lvl3_rows.append({
            "model": name,
            "n_grave": q["n_test_grave"],
            "n_fatal": q["n_fatal_in_grave"],
            "base_rate": q["base_rate_fatal_in_grave"],
            "pr_auc_fatal_given_grave": q["pr_auc_fatal_given_grave"],
        })
    lvl3_df = pd.DataFrame(lvl3_rows) if lvl3_rows else pd.DataFrame()

    best_key = max(results.keys(),
                    key=lambda k: results[k]["argmax_test"]["pr_auc_fatal"])

    sections = {
        "Comparación (test=2025)": _md(comp),
        "Ranking fatal y grave": _md(rank_df),
        "Nivel 3 (fatal|grave) — calidad condicional": (
            _md(lvl3_df) if not lvl3_df.empty else "N/A"
        ),
        "Bootstrap CI (test, 500 iter)": dumps_jsonable(
            {n: r["bootstrap_test"] for n, r in results.items()}, indent=2),
        "Thresholds tuneados en val": dumps_jsonable(
            {n: r["thresholds"] for n, r in results.items()}, indent=2),
        "Confusion matrix — best (" + best_key + ", argmax)": _cm_to_md(
            results[best_key]["argmax_test"]["confusion"]),
        "Best by PR-AUC(fatal)": best_key,
        "Política class_weight": (
            f"LGBM: {LGBM_CLASS_WEIGHT or 'None'}\n"
            f"LR:   {LR_CLASS_WEIGHT or 'None'}\n"
            "Consistente con script 28 (funcionó con estos valores)."
        ),
        "Composición jerárquica": (
            "P(no_injury) = 1 − p_injured\n"
            "P(minor)     = p_injured · (1 − p_grave|injured)\n"
            "P(serious)   = p_injured · p_grave|injured · (1 − p_fatal|grave)\n"
            "P(fatal)     = p_injured · p_grave|injured · p_fatal|grave\n\n"
            "Nivel 1: entrenado en 74k train (47% injured)\n"
            "Nivel 2: entrenado en train[injured] (~35k, 22% grave)\n"
            "Nivel 3: entrenado en train[grave] (~7.8k, 1.5% fatal)"
        ),
    }
    save_markdown_report(sections, Path("reports/29_hierarchical.md"))
    print(f"\nBest by PR-AUC(fatal): {best_key}")
    print("Done. See reports/29_hierarchical.md.")


if __name__ == "__main__":
    main()