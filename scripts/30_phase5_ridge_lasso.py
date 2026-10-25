"""
Phase 5 · Iteration 10 — Última prueba: Ridge y Lasso con/sin PCA.

8 configuraciones:

  Target A: injured binario (comparar con LGBM D094, F1=0.77, PR-AUC=0.83)
    1. RidgeClassifier sin PCA
    2. RidgeClassifier con PCA(0.95)
    3. L1-Logistic (Lasso) sin PCA
    4. L1-Logistic (Lasso) con PCA

  Target B: 4-clases (comparar con flat LGBM D098, PR-AUC fatal=0.0064)
    5. RidgeClassifier OVR sin PCA
    6. RidgeClassifier OVR con PCA
    7. L1-Logistic multiclase sin PCA
    8. L1-Logistic multiclase con PCA

Preprocesado: imputer + StandardScaler + OHE, fit en train.
PCA fit en train (sin leakage).
Threshold tuning en val; evaluado en test=2025.

Referencias:
  - LGBM injured (D094): F1=0.7718, PR-AUC=0.8297, Recall=0.8523
  - LR injured (D094):   F1=0.7647, PR-AUC=0.8142, Recall=0.8423
  - flat LGBM 4-class (D098): PR-AUC(fatal)=0.0064, kappa=0.554

Run:
    python scripts/30_phase5_ridge_lasso.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso, LogisticRegression, Ridge, RidgeClassifier,
)
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["no_injury", "minor", "serious", "fatal"]
LABELS_4 = [0, 1, 2, 3]
CLASS_NAMES_BIN = ["no_injury", "injured"]
LABELS_BIN = [0, 1]

RANDOM_SEED = 42

# Features (idénticas a scripts 28/29)
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


def make_preprocessor():
    num = NUMERIC_M2 + NUMERIC_NETWORK
    cat = CAT_M2 + CAT_NETWORK
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False, min_frequency=20), cat),
    ])


# ===========================================================================
# Model builders (Ridge / Lasso con/sin PCA)
# ===========================================================================
def make_ridge_binary_pipeline(use_pca: bool) -> Pipeline:
    steps = [("pre", make_preprocessor())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)))
    steps.append(("clf", RidgeClassifier(alpha=1.0)))
    return Pipeline(steps)


def make_lasso_binary_pipeline(use_pca: bool) -> Pipeline:
    steps = [("pre", make_preprocessor())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)))
    steps.append(("clf", LogisticRegression(
        penalty="l1", solver="saga", C=1.0, max_iter=3000,
        class_weight="balanced", random_state=RANDOM_SEED,
    )))
    return Pipeline(steps)


def make_ridge_multiclass_pipeline(use_pca: bool) -> Pipeline:
    steps = [("pre", make_preprocessor())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)))
    steps.append(("clf", RidgeClassifier(alpha=1.0)))
    return Pipeline(steps)


def make_lasso_multiclass_pipeline(use_pca: bool) -> Pipeline:
    steps = [("pre", make_preprocessor())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)))
    steps.append(("clf", LogisticRegression(
        penalty="l1", solver="saga", C=1.0, max_iter=3000,
        class_weight="balanced", random_state=RANDOM_SEED,
    )))
    return Pipeline(steps)


def get_proba(model, X) -> np.ndarray:
    """
    Interfaz uniforme de probabilidad:
      - LogisticRegression: predict_proba directamente.
      - RidgeClassifier (binary): decision_function → sigmoid.
      - RidgeClassifier (multiclass): decision_function → softmax.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    df = model.decision_function(X)
    if df.ndim == 1:
        # binary: sigmoid(decision_function)
        p1 = 1.0 / (1.0 + np.exp(-df))
        return np.column_stack([1 - p1, p1])
    # multiclass: softmax
    return softmax(df, axis=1)


# ===========================================================================
# Threshold tuning
# ===========================================================================
def tune_threshold_binary(y_val, p_val):
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p_val >= t).astype(int)
        f1 = f1_score(y_val, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def tune_thresholds_multiclass(y_val, p_val, grid=100):
    thresholds = {0: 0.0}
    for k in (1, 2, 3):
        y_k = (y_val == k).astype(int)
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.01, 0.99, grid):
            yp = (p_val[:, k] >= t).astype(int)
            if yp.sum() == 0:
                continue
            f1 = f1_score(y_k, yp, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
    return thresholds


def apply_thresholds_multiclass(p, thresholds):
    n = len(p)
    scores = np.zeros((n, 4))
    for k in (1, 2, 3):
        scores[:, k] = p[:, k] / max(1e-9, thresholds[k])
    pred = np.where(scores.max(axis=1) > 1.0, scores.argmax(axis=1), 0)
    return pred.astype(int)


# ===========================================================================
# Metrics
# ===========================================================================
def metrics_binary(y_true, y_pred, y_prob):
    return {
        "threshold": "tuned",
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "f1_injured": float(f1_score(y_true, y_pred, pos_label=1,
                                       zero_division=0)),
        "precision_injured": float(precision_score(
            y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_injured": float(recall_score(
            y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc_injured": float(average_precision_score(y_true, y_prob)),
        "roc_auc_injured": float(roc_auc_score(y_true, y_prob)),
    }


def metrics_4class(y_true, y_pred, y_proba):
    return {
        "kappa_q": float(cohen_kappa_score(y_true, y_pred,
                                             weights="quadratic")),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS_4,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - y_pred))),
        "recall": {n: float(r) for n, r in zip(
            CLASS_NAMES, recall_score(y_true, y_pred, labels=LABELS_4,
                                       average=None, zero_division=0))},
        "pr_auc_fatal": float(average_precision_score(
            (y_true == 3).astype(int), y_proba[:, 3])),
        "pr_auc_grave": float(average_precision_score(
            (y_true >= 2).astype(int), y_proba[:, 2] + y_proba[:, 3])),
        "confusion": {"labels": CLASS_NAMES,
                      "matrix": confusion_matrix(
                          y_true, y_pred, labels=LABELS_4).tolist()},
    }


def ranking_fatal(y_true, y_score):
    n = len(y_true)
    n_pos = int((y_true == 3).sum())
    base = n_pos / n
    order = np.argsort(-y_score)
    out = {"n_target": n_pos, "base_rate": round(float(base), 5), "by_k": {}}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int((y_true[order[:k]] == 3).sum())
        out["by_k"][f"k={int(pct*100)}%"] = {
            "k": k, "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_pos), 4),
            "lift": round((tp / k) / max(1e-12, base), 3),
        }
    return out


def bootstrap_binary(y_true, y_prob, threshold, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s, prs, rocs = [], [], []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, pp = y_true[idx], y_prob[idx]
        if (yt == 1).sum() < 2 or (yt == 0).sum() < 2:
            continue
        pred = (pp >= threshold).astype(int)
        f1s.append(f1_score(yt, pred, pos_label=1, zero_division=0))
        prs.append(average_precision_score(yt, pp))
        rocs.append(roc_auc_score(yt, pp))
    def _ci(vals):
        if not vals:
            return None
        arr = np.asarray(vals)
        return {"median": float(np.median(arr)),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5))}
    return {"f1_injured": _ci(f1s),
            "pr_auc_injured": _ci(prs),
            "roc_auc_injured": _ci(rocs)}


def bootstrap_fatal(y_true, y_score, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    prs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, pt = y_true[idx], y_score[idx]
        if (yt == 3).sum() < 2:
            continue
        prs.append(average_precision_score((yt == 3).astype(int), pt))
    if not prs:
        return None
    arr = np.asarray(prs)
    return {"median": float(np.median(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5))}


# ===========================================================================
# Markdown helper
# ===========================================================================
def _md(df_):
    h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
    s = "| " + " | ".join("---" for _ in df_.columns) + " |"
    rs = ["| " + " | ".join(str(v) for v in r) + " |"
          for r in df_.itertuples(index=False)]
    return "\n".join([h, s] + rs)


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

    y_all = df["severity_code"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]

    print("\nBuilding features...")
    X = build_features(df)
    add_te_columns(X, df, train_mask, y_train)
    X_train_df = X.loc[train_mask]
    X_val_df = X.loc[val_mask]
    X_test_df = X.loc[test_mask]
    print(f"  features: {X.shape[1]}\n")

    results = {}

    # ===========================================================
    # TARGET A: injured binario
    # ===========================================================
    y_train_bin = (y_train >= 1).astype(int)
    y_val_bin = (y_val >= 1).astype(int)
    y_test_bin = (y_test >= 1).astype(int)

    print("=" * 60)
    print("TARGET A: injured binario")
    print("=" * 60)

    binary_configs = [
        ("ridge_binary_no_pca", make_ridge_binary_pipeline(False)),
        ("ridge_binary_pca",    make_ridge_binary_pipeline(True)),
        ("lasso_binary_no_pca", make_lasso_binary_pipeline(False)),
        ("lasso_binary_pca",    make_lasso_binary_pipeline(True)),
    ]

    for name, pipeline in binary_configs:
        print(f"\n--- {name} ---")
        t0 = time.time()
        pipeline.fit(X_train_df, y_train_bin)
        fit_s = time.time() - t0
        p_val = get_proba(pipeline, X_val_df)[:, 1]
        p_test = get_proba(pipeline, X_test_df)[:, 1]
        thr = tune_threshold_binary(y_val_bin, p_val)
        pred_test = (p_test >= thr).astype(int)
        m = metrics_binary(y_test_bin, pred_test, p_test)
        boot = bootstrap_binary(y_test_bin, p_test, thr)
        n_feats = (Xt.shape[1] if (Xt := pipeline.named_steps["pre"]
                                              .transform(X_train_df)
                                    ) is not None else None)
        # Número de componentes si hay PCA
        n_components = None
        if "pca" in pipeline.named_steps:
            n_components = int(pipeline.named_steps["pca"].n_components_)
        results[name] = {
            "fit_seconds": round(fit_s, 1),
            "n_components": n_components,
            "threshold_tuned": round(thr, 3),
            "test": m,
            "bootstrap_test": boot,
        }
        print(f"  fit {fit_s:.1f}s | feat={n_feats}"
              f"{f' → PCA {n_components}' if n_components else ''}")
        print(f"  F1(inj)={m['f1_injured']:.4f} | PR-AUC={m['pr_auc_injured']:.4f}"
              f" | Recall={m['recall_injured']:.4f}"
              f" | Kappa={m['kappa']:.4f}")

    # ===========================================================
    # TARGET B: 4-clases (multiclase)
    # ===========================================================
    print("\n" + "=" * 60)
    print("TARGET B: 4-clases")
    print("=" * 60)

    multiclass_configs = [
        ("ridge_4class_no_pca", make_ridge_multiclass_pipeline(False)),
        ("ridge_4class_pca",    make_ridge_multiclass_pipeline(True)),
        ("lasso_4class_no_pca", make_lasso_multiclass_pipeline(False)),
        ("lasso_4class_pca",    make_lasso_multiclass_pipeline(True)),
    ]

    for name, pipeline in multiclass_configs:
        print(f"\n--- {name} ---")
        t0 = time.time()
        pipeline.fit(X_train_df, y_train)
        fit_s = time.time() - t0
        p_val = get_proba(pipeline, X_val_df)
        p_test = get_proba(pipeline, X_test_df)
        thr = tune_thresholds_multiclass(y_val, p_val)
        pred_test = apply_thresholds_multiclass(p_test, thr)
        m = metrics_4class(y_test, pred_test, p_test)
        boot_fatal = bootstrap_fatal(y_test, p_test[:, 3])
        rk_fatal = ranking_fatal(y_test, p_test[:, 3])
        n_feats = (Xt.shape[1] if (Xt := pipeline.named_steps["pre"]
                                              .transform(X_train_df)
                                    ) is not None else None)
        n_components = None
        if "pca" in pipeline.named_steps:
            n_components = int(pipeline.named_steps["pca"].n_components_)
        results[name] = {
            "fit_seconds": round(fit_s, 1),
            "n_components": n_components,
            "thresholds": {str(k): round(v, 3) for k, v in thr.items()},
            "test": m,
            "ranking_fatal": rk_fatal,
            "bootstrap_fatal": boot_fatal,
        }
        print(f"  fit {fit_s:.1f}s | feat={n_feats}"
              f"{f' → PCA {n_components}' if n_components else ''}")
        print(f"  Kappa={m['kappa_q']:.4f} | F1-macro={m['f1_macro']:.4f}"
              f" | PR-AUC(fatal)={m['pr_auc_fatal']:.5f}"
              f" | Lift@2={rk_fatal['by_k']['k=2%']['lift']:.2f}")

    save_json(results, OUT_TABLES / "30_ridge_lasso.json")

    # ===========================================================
    # Reportes
    # ===========================================================
    # Tabla A: injured binario
    rows_a = []
    for name in ("ridge_binary_no_pca", "ridge_binary_pca",
                 "lasso_binary_no_pca", "lasso_binary_pca"):
        r = results[name]
        m = r["test"]
        rows_a.append({
            "config": name,
            "n_components": r["n_components"] or "—",
            "f1_injured": round(m["f1_injured"], 4),
            "precision": round(m["precision_injured"], 4),
            "recall": round(m["recall_injured"], 4),
            "pr_auc": round(m["pr_auc_injured"], 4),
            "roc_auc": round(m["roc_auc_injured"], 4),
            "kappa": round(m["kappa"], 4),
        })
    comp_a = pd.DataFrame(rows_a)

    # Tabla B: 4-class
    rows_b = []
    for name in ("ridge_4class_no_pca", "ridge_4class_pca",
                 "lasso_4class_no_pca", "lasso_4class_pca"):
        r = results[name]
        m = r["test"]
        rk = r["ranking_fatal"]["by_k"]
        rows_b.append({
            "config": name,
            "n_components": r["n_components"] or "—",
            "kappa_q": round(m["kappa_q"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "mae_ord": round(m["mae_ord"], 4),
            "pr_auc_fatal": round(m["pr_auc_fatal"], 5),
            "pr_auc_grave": round(m["pr_auc_grave"], 4),
            "lift@2": rk["k=2%"]["lift"],
            "recall@10": rk["k=10%"]["recall"],
        })
    comp_b = pd.DataFrame(rows_b)

    sections = {
        "TARGET A — Injured binario": _md(comp_a),
        "Referencia TARGET A (D094, script 28)": (
            "LGBM: F1=0.7718, PR-AUC=0.8297, Recall=0.8523, Kappa=0.5754\n"
            "LR:   F1=0.7647, PR-AUC=0.8142, Recall=0.8423, Kappa=0.5629"
        ),
        "TARGET B — 4-clases": _md(comp_b),
        "Referencia TARGET B (D098, script 29)": (
            "flat LGBM: kappa=0.554, PR-AUC(fatal)=0.0064, Lift@2=4.68\n"
            "flat LR:   kappa=0.417, PR-AUC(fatal)=0.0053, Lift@2=4.68"
        ),
        "Bootstrap CI — injured": dumps_jsonable(
            {k: v["bootstrap_test"] for k, v in results.items()
             if "bootstrap_test" in v}, indent=2),
        "Bootstrap CI — fatal": dumps_jsonable(
            {k: v["bootstrap_fatal"] for k, v in results.items()
             if "bootstrap_fatal" in v}, indent=2),
        "Notes": (
            "RidgeClassifier: Ridge regression interna (OVR para multiclase). "
            "Sin predict_proba; se usa decision_function + sigmoid/softmax.\n"
            "Lasso: LogisticRegression con penalty='l1', solver='saga'.\n"
            "PCA: n_components=0.95 (retiene 95% varianza), fit en train.\n"
            "Preprocesado: imputer + StandardScaler + OHE, fit en train.\n"
            "Threshold tuning en val, evaluado en test=2025."
        ),
    }
    save_markdown_report(sections, Path("reports/30_ridge_lasso.md"))
    print("\nDone. See reports/30_ridge_lasso.md.")


if __name__ == "__main__":
    main()