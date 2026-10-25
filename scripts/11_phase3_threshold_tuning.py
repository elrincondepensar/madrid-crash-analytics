"""
Phase 3 · Iteration 4 — Threshold tuning on M4 probabilities.

Take the M4 fitted LR (best of iter 3) but replace the argmax decision with
a cost-sensitive decision:
    for each accident i, pick argmax_k ( cost_k * p_k(i) - penalty_k )
where penalty_k is the cost of predicting class k when the true class is
not k. Calibrate probabilities (isotonic) on val first.

Under a triage cost matrix:
    miss fatal   → cost 100
    miss serious → cost 10
    miss minor   → cost 1
    miss no_injury → cost 0.1

Report confusion matrix, kappa, and per-class recall under:
    - default argmax (baseline from iter 3)
    - calibrated + argmax
    - calibrated + cost-sensitive

Run:
    python scripts/11_phase3_threshold_tuning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.frozen import FrozenEstimator

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

# Cost of predicting class k when the true class is not k.
# Note: cost is a per-class severity-of-misclassification. It approximates
# the harm of failing to flag a real event of that severity.
COST_MATRIX = np.array([0.1, 1.0, 10.0, 100.0])  # per row: no_injury, minor, serious, fatal


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


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), NUMERIC_FULL),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FULL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=10.0, max_iter=3000, random_state=42)),
    ])


def decision_from_proba(proba: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """
    Cost-sensitive decision: pick class k that minimizes expected cost.
    Expected cost of predicting k = sum_j cost_j * p_j  - cost_k * p_k
    (cost_k is the cost we AVOID by predicting k correctly).
    Simplification: expected_loss_k = cost @ p - cost_k * p_k.
    Equivalent to argmax_k (cost_k * p_k - (cost @ p) * 0) minus (cost @ p).
    Because cost @ p is the same for all k, argmax_k (cost_k * p_k).
    """
    expected_loss = proba @ cost  # (n,)
    score = cost[None, :] * proba - expected_loss[:, None]
    return np.argmax(score, axis=1).astype(int)


def evaluate(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae_ordinal": float(np.mean(np.abs(y_true - y_pred))),
        "recall_per_class": {
            n: float(r) for n, r in zip(
                LABEL_NAMES,
                recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
            )
        },
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def _cm_to_md(cm: dict) -> str:
    df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
    header = "| | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + str(idx) + " | " + " | ".join(str(v) for v in row) + " |"
            for idx, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


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

    print("Fitting LR(C=10) on train...")
    pipe = build_pipeline()
    pipe.fit(X.loc[train_mask], y[train_mask])

    print("Calibrating (isotonic) using val...")
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(X.loc[val_mask], y[val_mask])

    proba_val = cal.predict_proba(X.loc[val_mask])
    proba_test = cal.predict_proba(X.loc[test_mask])
    y_val = y[val_mask]
    y_test = y[test_mask]

    print("Computing decisions...")
    pred_argmax_val = proba_val.argmax(1).astype(int)
    pred_argmax_test = proba_test.argmax(1).astype(int)
    pred_cost_val = decision_from_proba(proba_val, COST_MATRIX)
    pred_cost_test = decision_from_proba(proba_test, COST_MATRIX)

    # Threshold sweep on fatal (binary) — reference for Fase 5
    fatal_sweep = []
    for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        p_fatal = proba_test[:, 3]
        pred_fatal = (p_fatal >= t).astype(int)
        tp = int(((pred_fatal == 1) & (y_test == 3)).sum())
        fp = int(((pred_fatal == 1) & (y_test != 3)).sum())
        fn = int(((pred_fatal == 0) & (y_test == 3)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fatal_sweep.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "n_positive_predictions": tp + fp,
        })

    results = {
        "argmax_calibrated": {
            "val": evaluate(y_val, pred_argmax_val, proba_val),
            "test": evaluate(y_test, pred_argmax_test, proba_test),
        },
        "cost_sensitive": {
            "cost_matrix": COST_MATRIX.tolist(),
            "val": evaluate(y_val, pred_cost_val, proba_val),
            "test": evaluate(y_test, pred_cost_test, proba_test),
        },
        "fatal_threshold_sweep_test": fatal_sweep,
        "pr_auc_fatal_test_calibrated": float(
            average_precision_score((y_test == 3).astype(int), proba_test[:, 3])),
    }
    save_json(results, OUT / "11_threshold_tuning.json")

    def _row(label, split, d):
        return {
            "decisions": label, "split": split,
            "f1_macro": round(d["f1_macro"], 4),
            "kappa_q": round(d["kappa_quadratic"], 4),
            "mae_ord": round(d["mae_ordinal"], 4),
            "rec_no_inj": round(d["recall_per_class"]["no_injury"], 3),
            "rec_minor": round(d["recall_per_class"]["minor"], 3),
            "rec_serious": round(d["recall_per_class"]["serious"], 3),
            "rec_fatal": round(d["recall_per_class"]["fatal"], 3),
        }

    comp = pd.DataFrame([
        _row("argmax", "val", results["argmax_calibrated"]["val"]),
        _row("argmax", "test", results["argmax_calibrated"]["test"]),
        _row("cost-sensitive", "val", results["cost_sensitive"]["val"]),
        _row("cost-sensitive", "test", results["cost_sensitive"]["test"]),
    ])

    def _df_to_md(df_):
        header = "| " + " | ".join(str(c) for c in df_.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df_.columns) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in df_.itertuples(index=False)]
        return "\n".join([header, sep] + rows)

    sections = {
        "Comparison": _df_to_md(comp),
        "Confusion matrix — argmax (test)": _cm_to_md(results["argmax_calibrated"]["test"]["confusion_matrix"]),
        "Confusion matrix — cost-sensitive (test)": _cm_to_md(results["cost_sensitive"]["test"]["confusion_matrix"]),
        "Fatal threshold sweep (test)": _df_to_md(pd.DataFrame(fatal_sweep)),
        "PR-AUC(fatal) on test (calibrated)": str(results["pr_auc_fatal_test_calibrated"]),
        "Notes": (
            "Cost matrix (per class): no_injury=0.1, minor=1, serious=10, fatal=100.\n"
            "Cost-sensitive decision minimizes expected cost.\n"
            "Threshold sweep shows precision-recall trade-off for the fatal class alone."
        ),
    }
    save_markdown_report(sections, Path("reports/11_threshold_tuning.md"))
    print("Done. See reports/11_threshold_tuning.md.")


if __name__ == "__main__":
    main()