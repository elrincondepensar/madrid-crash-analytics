"""
Phase 5 · Iteration 7 — Threshold tuning for Opción B.

Los modelos de Opción B (script 25) tienen argmax que colapsa (recall_grave
7%). Este script:

  1. Reentrena los 3 mejores modelos de Opción B (catboost__adasyn,
     catboost__smote, lgbm__smote) con los mismos best_params congelados.
  2. Ajusta thresholds por clase (0,1,2) en val para maximizar F1(grave).
  3. Aplica los thresholds a test y reporta recall_grave, precision_grave,
     kappa y fatal_Lift@2.
  4. Compara contra el argmax.

También incluye LR(C=10) con scaler como baseline correcto.

Run:
    python scripts/27_phase5_opcion_b_thresholds.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

N_CLASSES = 3
LABELS_3 = [0, 1, 2]
CLASS_NAMES = ["no_injury", "minor", "grave"]

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

# Modelos ganadores del script 25 con best_params congelados
BEST_MODELS = {
    "catboost__adasyn": {
        "model": "catboost",
        "params": {
            "iterations": 2165, "learning_rate": 0.018891200276189388,
            "depth": 5, "l2_leaf_reg": 0.04342298948286135,
            "random_strength": 0.016480446427978974,
            "bagging_temperature": 0.5247564316322378,
        },
        "augmentation": "adasyn",
    },
    "catboost__smote": {
        "model": "catboost",
        "params": {
            "iterations": 2165, "learning_rate": 0.018891200276189388,
            "depth": 5, "l2_leaf_reg": 0.04342298948286135,
            "random_strength": 0.016480446427978974,
            "bagging_temperature": 0.5247564316322378,
        },
        "augmentation": "smote",
    },
    "lgbm__smote": {
        "model": "lgbm",
        "params": {
            "n_estimators": 858, "learning_rate": 0.010641910110563977,
            "num_leaves": 64, "max_depth": 8, "min_child_samples": 9,
            "reg_alpha": 0.0015336539197540766,
            "reg_lambda": 0.0026364370299579373,
            "colsample_bytree": 0.7240148908422406,
            "subsample": 0.7802615489179725,
        },
        "augmentation": "smote",
    },
}

SAMPLING_STRATEGY = {2: 20000}


# ===========================================================================
# Data prep
# ===========================================================================
def build_splits_v2(year):
    return pd.Series(
        np.where(year <= 2022, "train",
                 np.where(year <= 2024, "val", "test")),
        index=year.index,
    )


def severity_to_3class(s):
    return s.map({0: 0, 1: 1, 2: 2, 3: 2}).astype(int)


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
    y_grave_train = (y_train_code >= 2).astype(float)
    global_mean = float(y_grave_train.mean())

    def _fit_apply(col_src, col_te):
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


def make_preprocessor_with_scaler():
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
# Augmentation (para refit)
# ===========================================================================
def apply_augmentation(name, X, y):
    if name == "none":
        return X, y
    from imblearn.over_sampling import ADASYN, SMOTE
    if name == "smote":
        s = SMOTE(sampling_strategy=SAMPLING_STRATEGY, random_state=42,
                  k_neighbors=5)
    elif name == "adasyn":
        s = ADASYN(sampling_strategy=SAMPLING_STRATEGY, random_state=42,
                   n_neighbors=5)
    else:
        raise ValueError(name)
    return s.fit_resample(X, y)


# ===========================================================================
# Threshold tuning
# ===========================================================================
def tune_thresholds_3class(y_val, p_val, grid=200):
    """Thresholds por clase para maximizar F1 one-vs-rest en val."""
    thresholds = {0: 0.0}
    for k in (1, 2):
        y_k = (y_val == k).astype(int)
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.01, 0.99, grid):
            yp = (p_val[:, k] >= t).astype(int)
            tp = ((yp == 1) & (y_k == 1)).sum()
            fp = ((yp == 1) & (y_k == 0)).sum()
            fn = ((yp == 0) & (y_k == 1)).sum()
            if tp + fp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t
    return thresholds


def apply_thresholds_3class(p, thresholds):
    """Decisión por score normalizado."""
    n = len(p)
    scores = np.zeros((n, N_CLASSES))
    for k in (1, 2):
        scores[:, k] = p[:, k] / max(1e-9, thresholds[k])
    pred = np.where(scores.max(axis=1) > 1.0, scores.argmax(axis=1), 0)
    return pred.astype(int)


# ===========================================================================
# Metrics
# ===========================================================================
def fatal_ranking(y_code, grave_p, target_fatal=3):
    n = len(y_code)
    n_fatal = int((y_code == target_fatal).sum())
    order = np.argsort(-grave_p)
    out = {"n_fatal": n_fatal, "by_k": {}}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int((y_code[order[:k]] == target_fatal).sum())
        out["by_k"][f"k={int(pct*100)}%"] = {
            "k": k, "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_fatal), 4),
            "lift": round((tp / k) / max(1e-12, n_fatal / n), 3),
        }
    return out


def full_metrics_3class(y_true, y_pred, y_proba):
    return {
        "kappa_q": float(cohen_kappa_score(y_true, y_pred,
                                             weights="quadratic")),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS_3,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_true) - y_pred))),
        "pr_auc_grave": float(average_precision_score(
            (y_true == 2).astype(int), y_proba[:, 2])),
        "recall": {n: float(r) for n, r in zip(
            CLASS_NAMES, recall_score(y_true, y_pred, labels=LABELS_3,
                                       average=None, zero_division=0))},
        "confusion": {"labels": CLASS_NAMES,
                      "matrix": confusion_matrix(y_true, y_pred,
                                                  labels=LABELS_3).tolist()},
    }


def bootstrap_recall_grave(y_true, y_pred, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    recalls = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if (yt == 2).sum() < 2:
            continue
        recalls.append(float(recall_score(
            yt, yp, labels=[2], average="macro", zero_division=0)))
    if not recalls:
        return None
    arr = np.asarray(recalls)
    return {"median": float(np.median(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5))}


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading data...")
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
    print(f"  {len(df):,} accidents, splits={df['split'].value_counts().to_dict()}")

    y_all = df["y3"].astype(int).to_numpy()
    y_code = df["severity_code"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    y_code_test = y_code[test_mask]

    print("\nBuilding features (with scaler)...")
    X = build_features(df)
    add_te_columns(X, df, train_mask, y_code[train_mask])
    pre = make_preprocessor_with_scaler()
    Xt_train = pre.fit_transform(X.loc[train_mask])
    Xt_val = pre.transform(X.loc[val_mask])
    Xt_test = pre.transform(X.loc[test_mask])
    print(f"  features: {Xt_train.shape[1]}")

    results = {}

    # ---- LR baseline (con scaler, debería funcionar) -------------------
    print("\n--- LR baseline (C=10, con scaler) ---")
    t0 = time.time()
    lr = LogisticRegression(C=10.0, max_iter=3000, random_state=42)
    lr.fit(Xt_train, y_train)
    print(f"  fit {time.time()-t0:.1f}s")
    p_val_lr = lr.predict_proba(Xt_val)
    p_test_lr = lr.predict_proba(Xt_test)
    # Argmax
    pred_val_lr_argmax = p_val_lr.argmax(1)
    pred_test_lr_argmax = p_test_lr.argmax(1)
    # Threshold tuned
    thr_lr = tune_thresholds_3class(y_val, p_val_lr)
    pred_val_lr_thr = apply_thresholds_3class(p_val_lr, thr_lr)
    pred_test_lr_thr = apply_thresholds_3class(p_test_lr, thr_lr)
    results["lr"] = {
        "thresholds": {str(k): round(v, 3) for k, v in thr_lr.items()},
        "val_argmax": full_metrics_3class(y_val, pred_val_lr_argmax, p_val_lr),
        "test_argmax": full_metrics_3class(y_test, pred_test_lr_argmax, p_test_lr),
        "val_thresholded": full_metrics_3class(y_val, pred_val_lr_thr, p_val_lr),
        "test_thresholded": full_metrics_3class(y_test, pred_test_lr_thr, p_test_lr),
        "fatal_ranking_test": fatal_ranking(y_code_test, p_test_lr[:, 2]),
        "bootstrap_recall_grave_test": bootstrap_recall_grave(
            y_test, pred_test_lr_thr),
    }
    m = results["lr"]["test_thresholded"]
    print(f"  TEST thresholded: kappa={m['kappa_q']:.3f}  "
          f"rec_grave={m['recall']['grave']:.3f}  "
          f"PR-AUC(grave)={m['pr_auc_grave']:.4f}")

    # ---- Modelos boosting ganadores de Opción B ------------------------
    for key, cfg in BEST_MODELS.items():
        print(f"\n--- {key} (reentrenado con best_params) ---")
        X_aug, y_aug = apply_augmentation(cfg["augmentation"],
                                            Xt_train, y_train)
        if cfg["model"] == "catboost":
            m_obj = CatBoostClassifier(
                loss_function="MultiClass", classes_count=N_CLASSES,
                random_seed=42, verbose=0, thread_count=8,
                allow_writing_files=False, **cfg["params"],
            )
        else:
            m_obj = LGBMClassifier(
                objective="multiclass", num_class=N_CLASSES,
                random_state=42, n_jobs=8, verbose=-1, **cfg["params"],
            )
        t0 = time.time()
        m_obj.fit(X_aug, y_aug)
        print(f"  fit {time.time()-t0:.1f}s on {len(y_aug)} samples")

        p_val = m_obj.predict_proba(Xt_val)
        p_test = m_obj.predict_proba(Xt_test)
        pred_val_argmax = p_val.argmax(1)
        pred_test_argmax = p_test.argmax(1)
        thr = tune_thresholds_3class(y_val, p_val)
        pred_val_thr = apply_thresholds_3class(p_val, thr)
        pred_test_thr = apply_thresholds_3class(p_test, thr)

        results[key] = {
            "thresholds": {str(k): round(v, 3) for k, v in thr.items()},
            "val_argmax": full_metrics_3class(y_val, pred_val_argmax, p_val),
            "test_argmax": full_metrics_3class(y_test, pred_test_argmax, p_test),
            "val_thresholded": full_metrics_3class(y_val, pred_val_thr, p_val),
            "test_thresholded": full_metrics_3class(y_test, pred_test_thr, p_test),
            "fatal_ranking_test": fatal_ranking(y_code_test, p_test[:, 2]),
            "bootstrap_recall_grave_test": bootstrap_recall_grave(
                y_test, pred_test_thr),
        }
        m = results[key]["test_thresholded"]
        print(f"  TEST thresholded: kappa={m['kappa_q']:.3f}  "
              f"rec_grave={m['recall']['grave']:.3f}  "
              f"PR-AUC(grave)={m['pr_auc_grave']:.4f}")

    save_json(results, OUT_TABLES / "27_opcion_b_thresholds.json")

    # ---- Comparison table ---------------------------------------------
    def _row(key):
        r = results[key]
        argmax = r["test_argmax"]
        thr = r["test_thresholded"]
        fk = r["fatal_ranking_test"]["by_k"]
        return {
            "model": key,
            "kappa_argmax": round(argmax["kappa_q"], 4),
            "kappa_thr": round(thr["kappa_q"], 4),
            "rec_grave_argmax": round(argmax["recall"]["grave"], 3),
            "rec_grave_thr": round(thr["recall"]["grave"], 3),
            "rec_minor_thr": round(thr["recall"]["minor"], 3),
            "rec_noinj_thr": round(thr["recall"]["no_injury"], 3),
            "pr_auc_grave": round(thr["pr_auc_grave"], 4),
            "fatal_Lift@2": fk["k=2%"]["lift"],
            "fatal_Lift@5": fk["k=5%"]["lift"],
            "fatal_Recall@10": fk["k=10%"]["recall"],
        }
    comp = pd.DataFrame([_row(k) for k in results])

    def _md(df_):
        h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
        s = "| " + " | ".join("---" for _ in df_.columns) + " |"
        rs = ["| " + " | ".join(str(v) for v in r) + " |"
              for r in df_.itertuples(index=False)]
        return "\n".join([h, s] + rs)

    sections = {
        "Opción B — argmax vs thresholded (test=2025)": _md(comp),
        "Thresholds seleccionados en val": dumps_jsonable(
            {k: results[k]["thresholds"] for k in results}, indent=2),
        "Bootstrap recall_grave (test, thresholded)": dumps_jsonable(
            {k: results[k]["bootstrap_recall_grave_test"] for k in results},
            indent=2),
        "Confusion matrices — best model (test, thresholded)": _md(
            pd.DataFrame(
                results["catboost__adasyn"]["test_thresholded"]
                ["confusion"]["matrix"],
                index=CLASS_NAMES, columns=CLASS_NAMES,
            ).reset_index().rename(columns={"index": "true"})
        ),
        "Notes": (
            "Target: 0=no_injury, 1=minor, 2=grave (serious ∪ fatal).\n"
            "Preprocesado con StandardScaler (necesario para LR y estable "
            "para árboles).\n"
            "Thresholds por clase ajustados en val para maximizar F1 OvR.\n"
            "Regla de decisión: si max(p_k/thr_k) > 1 → argmax(p_k/thr_k), "
            "si no → no_injury.\n"
            "fatal evaluado como ranking usando P(grave)."
        ),
    }
    save_markdown_report(sections, Path("reports/27_opcion_b_thresholds.md"))
    print("\nDone. See reports/27_opcion_b_thresholds.md")


if __name__ == "__main__":
    main()