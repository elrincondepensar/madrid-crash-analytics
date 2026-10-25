"""
Phase 3 · Iteration 5 — Ranking metrics + data augmentation.

Answer two questions with data:
  1. Does the M2 model rank fatal accidents usefully? (Recall@k, Lift@k)
  2. Does SMOTE / Borderline-SMOTE / ADASYN improve the operational metrics
     beyond what we already have?

Evaluated on val and test with:
  - Classification: F1-macro, kappa_q, MAE ordinal
  - Ranking: Recall@k, Precision@k, Lift@k for k ∈ {1,2,5,10,20}%
  - Thresholds per class optimized on val, applied to test

Run:
    pip install imbalanced-learn
    python scripts/12_phase3_ranking_and_augmentation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

ACCIDENTS = Path("data/processed/accidents_clean.parquet")
SPLITS = Path("data/processed/splits.parquet")
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]
LABELS = [0, 1, 2, 3]

NUMERIC_FULL = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "is_weekend",
    "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
    "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
    "coordenada_x_utm", "coordenada_y_utm",
]
CAT_FULL = ["tipo_accidente", "estado_meteorologico", "distrito"]


# ---------------------------------------------------------------------------
# Features (same as iter 3 — M4)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    for c in ["has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian"]:
        X[c] = df[c].astype(float)
    X["coordenada_x_utm"] = df["coordenada_x_utm"].astype(float)
    X["coordenada_y_utm"] = df["coordenada_y_utm"].astype(float)
    X["tipo_accidente"] = df["tipo_accidente"].fillna("Desconocido").astype(str)
    X["estado_meteorologico"] = df["estado_meteorologico"].fillna("Se desconoce").astype(str)
    X["distrito"] = df["distrito"].fillna("DESCONOCIDO").astype(str)
    return X


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), NUMERIC_FULL),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FULL),
    ])


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def ranking_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                    target_class: int,
                    k_percents: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20),
                    ) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    n_target = int((y_true == target_class).sum())
    base_rate = n_target / max(1, n)
    order = np.argsort(-y_proba[:, target_class])
    out = {}
    for pct in k_percents:
        k = max(1, int(np.ceil(n * pct)))
        top = order[:k]
        tp = int((y_true[top] == target_class).sum())
        precision = tp / k
        recall = tp / max(1, n_target)
        lift = precision / max(1e-12, base_rate)
        out[f"k={int(pct * 100)}%"] = {
            "k": int(k),
            "tp": tp,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "lift": round(float(lift), 3),
        }
    return {"base_rate": round(float(base_rate), 5),
            "n_target": n_target,
            "by_k": out}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred) -> dict:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred,
                                                     weights="quadratic")),
        "mae_ordinal": float(np.mean(np.abs(np.asarray(y_true)
                                             - np.asarray(y_pred)))),
        "recall_per_class": {
            n: float(r) for n, r in zip(
                LABEL_NAMES,
                recall_score(y_true, y_pred, labels=LABELS,
                             average=None, zero_division=0),
            )
        },
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Threshold tuning per class on val
# ---------------------------------------------------------------------------
def tune_thresholds_per_class(y_true: np.ndarray, y_proba: np.ndarray,
                              classes: tuple[int, ...] = (1, 2, 3),
                              grid: int = 200) -> dict[int, float]:
    """
    For each class k, find threshold t maximizing F1(k, one-vs-rest) on val.
    """
    thresholds = {0: 0.0}  # default for majority class
    for k in classes:
        y_k = (y_true == k).astype(int)
        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.99, grid):
            y_pred_k = (y_proba[:, k] >= t).astype(int)
            tp = ((y_pred_k == 1) & (y_k == 1)).sum()
            fp = ((y_pred_k == 1) & (y_k == 0)).sum()
            fn = ((y_pred_k == 0) & (y_k == 1)).sum()
            if tp + fp == 0:
                f1 = 0.0
            else:
                prec = tp / (tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * prec * rec / max(1e-12, prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[k] = best_t
    return thresholds


def apply_thresholds(y_proba: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    """
    For each sample:
      - normalized score_k = p_k / threshold_k for k in {1,2,3}
      - if max(score) > 1: predict argmax_k score_k
      - else: predict 0 (no_injury)
    """
    n = len(y_proba)
    pred = np.zeros(n, dtype=int)
    scores = np.zeros((n, 4))
    for k in (1, 2, 3):
        scores[:, k] = y_proba[:, k] / max(1e-9, thresholds[k])
    best_k = scores.argmax(axis=1)
    best_score = scores.max(axis=1)
    pred = np.where(best_score > 1.0, best_k, 0)
    return pred.astype(int)


# ---------------------------------------------------------------------------
# Training variants
# ---------------------------------------------------------------------------
def train_variant(name: str, variant: str,
                  X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Fit LR(C=10) with or without imblearn augmentation; evaluate on val/test."""
    t0 = time.time()
    pre = make_preprocessor()
    if variant == "baseline":
        pipe = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(C=10.0, max_iter=3000, random_state=42)),
        ])
    else:
        # Need ratio-limited oversampling so 116 fatal don't become 39k
        if variant == "smote":
            sampler = SMOTE(sampling_strategy={
                0: 39_157, 1: 27_209, 2: 7_788, 3: 1_160,
            }, random_state=42, k_neighbors=5)
        elif variant == "borderline":
            sampler = BorderlineSMOTE(sampling_strategy={
                0: 39_157, 1: 27_209, 2: 7_788, 3: 1_160,
            }, random_state=42, kind="borderline-1")
        elif variant == "adasyn":
            sampler = ADASYN(sampling_strategy={
                0: 39_157, 1: 27_209, 2: 7_788, 3: 1_160,
            }, random_state=42, n_neighbors=5)
        else:
            raise ValueError(variant)
        pipe = ImbPipeline([
            ("pre", pre),
            ("sampler", sampler),
            ("clf", LogisticRegression(C=10.0, max_iter=3000, random_state=42)),
        ])
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"  {name}: fit in {fit_time:.1f}s")

    proba_val = pipe.predict_proba(X_val)
    proba_test = pipe.predict_proba(X_test)

    # Default argmax
    pred_val_argmax = proba_val.argmax(1)
    pred_test_argmax = proba_test.argmax(1)

    # Threshold tuning on val, applied to test
    thresholds = tune_thresholds_per_class(y_val, proba_val)
    pred_val_thresh = apply_thresholds(proba_val, thresholds)
    pred_test_thresh = apply_thresholds(proba_test, thresholds)

    return {
        "variant": variant,
        "fit_time_s": round(fit_time, 1),
        "thresholds": {str(k): round(v, 3) for k, v in thresholds.items()},
        "val": {
            "argmax": classification_metrics(y_val, pred_val_argmax),
            "thresholded": classification_metrics(y_val, pred_val_thresh),
        },
        "test": {
            "argmax": classification_metrics(y_test, pred_test_argmax),
            "thresholded": classification_metrics(y_test, pred_test_thresh),
        },
        "ranking_test": {
            "fatal": ranking_metrics(y_test, proba_test, target_class=3),
            "serious": ranking_metrics(y_test, proba_test, target_class=2),
        },
        "ranking_val": {
            "fatal": ranking_metrics(y_val, proba_val, target_class=3),
            "serious": ranking_metrics(y_val, proba_val, target_class=2),
        },
        "pr_auc_fatal_test": float(
            average_precision_score((y_test == 3).astype(int), proba_test[:, 3])),
    }


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def _cm_to_md(cm: dict) -> str:
    df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
    header = "| | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + str(idx) + " | " + " | ".join(str(v) for v in row) + " |"
            for idx, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {ACCIDENTS}...")
    df = pd.read_parquet(ACCIDENTS)
    splits = pd.read_parquet(SPLITS)
    df = df.merge(splits[["num_expediente", "split"]],
                  on="num_expediente", how="left")
    X = build_features(df)
    y = df["severity_code"].astype(int).to_numpy()

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    X_train, y_train = X.loc[train_mask], y[train_mask]
    X_val, y_val = X.loc[val_mask], y[val_mask]
    X_test, y_test = X.loc[test_mask], y[test_mask]
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    results = {}
    for variant in ("baseline", "smote", "borderline", "adasyn"):
        print(f"\nTraining variant: {variant}")
        results[variant] = train_variant(
            name=variant, variant=variant,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )

    save_json({"results": results}, OUT / "12_ranking_and_augmentation.json")

    # --- Comparison table (test, thresholded) ---
    rows = []
    for v in ("baseline", "smote", "borderline", "adasyn"):
        r = results[v]
        m = r["test"]["thresholded"]
        rk_f = r["ranking_test"]["fatal"]["by_k"]
        rows.append({
            "variant": v,
            "kappa_q": round(m["kappa_quadratic"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "mae_ord": round(m["mae_ordinal"], 4),
            "rec_minor": round(m["recall_per_class"]["minor"], 3),
            "rec_serious": round(m["recall_per_class"]["serious"], 3),
            "rec_fatal": round(m["recall_per_class"]["fatal"], 3),
            "pr_auc_fatal": round(r["pr_auc_fatal_test"], 5),
            "lift@1": rk_f["k=1%"]["lift"],
            "lift@5": rk_f["k=5%"]["lift"],
            "recall@5": rk_f["k=5%"]["recall"],
            "recall@10": rk_f["k=10%"]["recall"],
        })
    comparison = pd.DataFrame(rows)

    # --- Ranking table per variant (fatal) ---
    ranking_rows = []
    for v in ("baseline", "smote", "borderline", "adasyn"):
        rk = results[v]["ranking_test"]["fatal"]["by_k"]
        for k_label, vals in rk.items():
            ranking_rows.append({
                "variant": v, "k": k_label,
                "tp": vals["tp"],
                "precision": vals["precision"],
                "recall": vals["recall"],
                "lift": vals["lift"],
            })
    ranking_df = pd.DataFrame(ranking_rows)

    # --- Confusion matrices (thresholded, test) ---
    cm_md = []
    for v in ("baseline", "smote", "borderline", "adasyn"):
        cm = results[v]["test"]["thresholded"]["confusion_matrix"]
        cm_md.append(f"**{v}** (test, thresholded)\n\n{_cm_to_md(cm)}\n")

    sections = {
        "Comparison — test (thresholded per class on val)":
            _df_to_md(comparison),
        "Fatal ranking on test (per variant)":
            _df_to_md(ranking_df),
        "Thresholds selected on val": dumps_jsonable(
            {v: results[v]["thresholds"] for v in results}, indent=2),
        "Confusion matrices (test, thresholded)": "\n".join(cm_md),
        "Notes": (
            "- Threshold per class tuned on val to maximize per-class F1 "
            "(one-vs-rest).\n"
            "- Lift@k = (precision in top k%) / base_rate(fatal).\n"
            "- Base fatal rate in test = 62 / 42386 = 0.00146.\n"
            "- SMOTE sampling_strategy caps class fatal at 1.160 (1:10 vs "
            "majority) to avoid overfit from 116 real samples.\n"
            "- SMOTE interpolates UTM coordinates linearly; documented as "
            "limitation (D054)."
        ),
    }
    save_markdown_report(sections, Path("reports/12_ranking_and_augmentation.md"))
    print("\nDone. See reports/12_ranking_and_augmentation.md.")


if __name__ == "__main__":
    main()