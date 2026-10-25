"""
Phase 3 · Iteration 2b — Baselines.

Baseline 0 (majority) and baseline 1 (multinomial LR) on M2 features per D031.
Reports F1-macro, balanced accuracy, kappa quadratic, MAE ordinal, per-class
recall, PR-AUC(fatal), confusion matrices and 500-iteration bootstrap CIs.

Run from the repository root:
    python scripts/09_phase3_baselines.py

Outputs:
    reports/tables/09_baselines.json
    reports/09_baselines.md
    data/processed/baseline_test_predictions.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

ACCIDENTS = Path("data/processed/accidents_clean.parquet")
SPLITS = Path("data/processed/splits.parquet")
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]

NUMERIC_FEATURES = ["hour_sin", "hour_cos", "n_people",
                    "has_bicycle", "has_motorcycle",
                    "has_heavy_vehicle", "has_pedestrian"]
CATEGORICAL_FEATURES = ["tipo_accidente", "estado_meteorologico"]


# ---------------------------------------------------------------------------
# Feature building (M2, minimal for baseline)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    hour = df["hour"].astype(float)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["n_people"] = df["n_people"].astype(float)
    X["has_bicycle"] = df["has_bicycle"].astype(bool)
    X["has_motorcycle"] = df["has_motorcycle"].astype(bool)
    X["has_heavy_vehicle"] = df["has_heavy_vehicle"].astype(bool)
    X["has_pedestrian"] = df["has_pedestrian"].astype(bool)
    X["tipo_accidente"] = df["tipo_accidente"].fillna("Desconocido").astype(str)
    X["estado_meteorologico"] = (df["estado_meteorologico"]
                                   .fillna("Se desconoce").astype(str))
    y = df["severity_code"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    result = {
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred,
                                                     weights="quadratic")),
        "mae_ordinal": float(np.mean(np.abs(y_true - y_pred))),
        "recall_per_class": {
            name: float(r)
            for name, r in zip(
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
    n_fatal = int((y_true == 3).sum())
    if n_fatal > 0:
        result["pr_auc_fatal"] = float(
            average_precision_score((y_true == 3).astype(int), y_proba[:, 3])
        )
    else:
        result["pr_auc_fatal"] = None
    result["n_samples"] = int(len(y_true))
    result["n_fatal_true"] = n_fatal
    return result


def bootstrap_ci(y_true: np.ndarray, pred_or_proba: np.ndarray,
                 metric_fn, n_iter: int = 500, seed: int = 42,
                 require_positive_label: int | None = None) -> dict | None:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if require_positive_label is not None and \
           (yt == require_positive_label).sum() < 2:
            continue
        s = pred_or_proba[idx]
        try:
            v = metric_fn(yt, s)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            values.append(float(v))
    if not values:
        return None
    return {
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "n_valid_resamples": len(values),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {ACCIDENTS}...")
    df = pd.read_parquet(ACCIDENTS)
    splits = pd.read_parquet(SPLITS)
    df = df.merge(splits[["num_expediente", "split"]],
                  on="num_expediente", how="left")
    if df["split"].isna().any():
        raise RuntimeError("Some accidents lack a split assignment.")
    print(f"  {len(df):,} accidents joined with splits.")

    print("Building M2 features...")
    X, y = build_features(df)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    print(f"  train={train_mask.sum():,}  val={val_mask.sum():,}  "
          f"test={test_mask.sum():,}")

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask].to_numpy()
    y_val = y.loc[val_mask].to_numpy()
    y_test = y.loc[test_mask].to_numpy()

    # -----------------------------------------------------------------------
    # Baseline 0 — majority class from train
    # -----------------------------------------------------------------------
    print("\nBaseline 0: majority class...")
    majority = int(np.bincount(y_train, minlength=4).argmax())
    print(f"  majority = {LABEL_NAMES[majority]} ({majority})")

    def _constant(y_true, proba, cls=majority):
        pred = np.full(len(y_true), cls, dtype=int)
        p = np.zeros((len(y_true), 4))
        p[:, cls] = 1.0
        return pred, p

    y_pred_0_val, y_proba_0_val = _constant(y_val, None)
    y_pred_0_test, y_proba_0_test = _constant(y_test, None)

    # -----------------------------------------------------------------------
    # Baseline 1 — LR multinomial
    # -----------------------------------------------------------------------
    print("\nFitting ColumnTransformer on train (only)...")
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    t0 = time.time()
    pre.fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_val = pre.transform(X_val)
    Xt_test = pre.transform(X_test)
    print(f"  preprocessor fitted in {time.time() - t0:.1f}s, "
          f"feature dim = {Xt_train.shape[1]}.")

    print("\nBaseline 1a: LR multinomial (class_weight='balanced')...")
    t0 = time.time()
    lr_bal = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=1.0,
        class_weight="balanced", random_state=42,
    )
    lr_bal.fit(Xt_train, y_train)
    print(f"  fit in {time.time() - t0:.1f}s.")
    y_pred_1_val = lr_bal.predict(Xt_val).astype(int)
    y_proba_1_val = lr_bal.predict_proba(Xt_val)
    y_pred_1_test = lr_bal.predict(Xt_test).astype(int)
    y_proba_1_test = lr_bal.predict_proba(Xt_test)

    print("\nBaseline 1b: LR multinomial (no reweighting)...")
    t0 = time.time()
    lr_raw = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=1.0, random_state=42,
    )
    lr_raw.fit(Xt_train, y_train)
    print(f"  fit in {time.time() - t0:.1f}s.")
    y_pred_1b_val = lr_raw.predict(Xt_val).astype(int)
    y_proba_1b_val = lr_raw.predict_proba(Xt_val)
    y_pred_1b_test = lr_raw.predict(Xt_test).astype(int)
    y_proba_1b_test = lr_raw.predict_proba(Xt_test)

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    print("\nEvaluating on val and test...")
    results = {
        "splits": {"train": int(train_mask.sum()),
                   "val": int(val_mask.sum()),
                   "test": int(test_mask.sum())},
        "majority_class": {"code": majority, "label": LABEL_NAMES[majority]},
        "baseline_0_majority": {
            "val": evaluate(y_val, y_pred_0_val, y_proba_0_val),
            "test": evaluate(y_test, y_pred_0_test, y_proba_0_test),
        },
        "baseline_1a_lr_balanced": {
            "val": evaluate(y_val, y_pred_1_val, y_proba_1_val),
            "test": evaluate(y_test, y_pred_1_test, y_proba_1_test),
        },
        "baseline_1b_lr_unweighted": {
            "val": evaluate(y_val, y_pred_1b_val, y_proba_1b_val),
            "test": evaluate(y_test, y_pred_1b_test, y_proba_1b_test),
        },
    }

    print("Bootstrapping (500 iterations) on test...")
    f1_macro_fn = lambda yt, yp: f1_score(  # noqa: E731
        yt, yp, average="macro", zero_division=0)
    pr_auc_fn = lambda yt, p: (  # noqa: E731
        average_precision_score((yt == 3).astype(int), p)
        if (yt == 3).sum() > 0 else None)

    bootstrap = {}
    for name, pred, proba in [
        ("baseline_0_majority", y_pred_0_test, y_proba_0_test),
        ("baseline_1a_lr_balanced", y_pred_1_test, y_proba_1_test),
        ("baseline_1b_lr_unweighted", y_pred_1b_test, y_proba_1b_test),
    ]:
        bootstrap[name] = {
            "f1_macro": bootstrap_ci(y_test, pred, f1_macro_fn),
            "pr_auc_fatal": bootstrap_ci(
                y_test, proba[:, 3], pr_auc_fn,
                require_positive_label=3,
            ),
        }
    results["bootstrap_ci_test"] = bootstrap

    save_json(results, OUT / "09_baselines.json")

    # -----------------------------------------------------------------------
    # Persist test predictions for later error analysis
    # -----------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "num_expediente": df.loc[test_mask, "num_expediente"].to_numpy(),
        "year": df.loc[test_mask, "year"].astype(int).to_numpy(),
        "y_true": y_test,
        "pred_majority": y_pred_0_test,
        "pred_lr_balanced": y_pred_1_test,
        "pred_lr_unweighted": y_pred_1b_test,
        "p_lr_balanced_no_injury": y_proba_1_test[:, 0],
        "p_lr_balanced_minor": y_proba_1_test[:, 1],
        "p_lr_balanced_serious": y_proba_1_test[:, 2],
        "p_lr_balanced_fatal": y_proba_1_test[:, 3],
    })
    pred_df.to_parquet("data/processed/baseline_test_predictions.parquet",
                       index=False)

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    def _fmt_metrics(split_dict):
        return {
            "n": split_dict["n_samples"],
            "f1_macro": round(split_dict["f1_macro"], 4),
            "balanced_acc": round(split_dict["balanced_accuracy"], 4),
            "kappa_q": round(split_dict["kappa_quadratic"], 4),
            "mae_ord": round(split_dict["mae_ordinal"], 4),
            "pr_auc_fatal": (round(split_dict["pr_auc_fatal"], 5)
                             if split_dict["pr_auc_fatal"] is not None
                             else None),
            "recall_per_class": {
                k: round(v, 3)
                for k, v in split_dict["recall_per_class"].items()
            },
        }

    comparison_rows = []
    for name in ["baseline_0_majority", "baseline_1a_lr_balanced",
                 "baseline_1b_lr_unweighted"]:
        for split in ("val", "test"):
            m = results[name][split]
            comparison_rows.append({
                "model": name, "split": split,
                "f1_macro": round(m["f1_macro"], 4),
                "balanced_acc": round(m["balanced_accuracy"], 4),
                "kappa_q": round(m["kappa_quadratic"], 4),
                "mae_ord": round(m["mae_ordinal"], 4),
                "pr_auc_fatal": (round(m["pr_auc_fatal"], 5)
                                 if m["pr_auc_fatal"] is not None else None),
                "recall_fatal": round(m["recall_per_class"]["fatal"], 3),
            })
    comparison_df = pd.DataFrame(comparison_rows)

    confusion_md = []
    for name in ["baseline_0_majority", "baseline_1a_lr_balanced",
                 "baseline_1b_lr_unweighted"]:
        cm = results[name]["test"]["confusion_matrix"]
        cm_df = pd.DataFrame(cm["matrix"], index=cm["labels"],
                              columns=cm["labels"])
        confusion_md.append(f"**{name}** (test)\n\n{cm_df.to_markdown()}\n")

    sections = {
        "Splits": dumps_jsonable(results["splits"], indent=2),
        "Majority class": dumps_jsonable(results["majority_class"], indent=2),
        "Comparison — val and test": comparison_df.to_markdown(index=False),
        "Bootstrap 95% CI (test)": dumps_jsonable(bootstrap, indent=2),
        "Confusion matrices (test)": "\n".join(confusion_md),
    }
    save_markdown_report(sections, Path("reports/09_baselines.md"))
    print("\nDone. See reports/09_baselines.md and "
          "data/processed/baseline_test_predictions.parquet.")


if __name__ == "__main__":
    main()