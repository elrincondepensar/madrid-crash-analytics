"""
Phase 3 · Iteration 3 — Ablation: features vs. hyperparameters.

Compares four configurations on the same temporal splits:

    M1: minimal features, default LR                 (baseline from iter 2)
    M2: full D031 features, default LR               (feature ablation)
    M3: minimal features, GridSearchCV + LeaveOneGroupOut on train years
    M4: full D031 features, GridSearchCV + LeaveOneGroupOut on train years

Reports for each model:
    - kappa quadratic, f1_macro, balanced_accuracy, mae_ordinal
    - recall per class
    - pr_auc_macro (multiclass), pr_auc_fatal (ovr), roc_auc_macro (ovr)
    - confusion matrix

Bootstraps 95% CI on test for the best model.

Run:
    python scripts/10_phase3_baselines_tuned.py
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
    average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, make_scorer, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

ACCIDENTS = Path("data/processed/accidents_clean.parquet")
SPLITS = Path("data/processed/splits.parquet")
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]

# --- Feature sets ----------------------------------------------------------
NUMERIC_MINIMAL = [
    "hour_sin", "hour_cos", "n_people",
    "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
]
CAT_MINIMAL = ["tipo_accidente", "estado_meteorologico"]

NUMERIC_FULL = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "is_weekend",
    "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
    "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
    "coordenada_x_utm", "coordenada_y_utm",
]
CAT_FULL = ["tipo_accidente", "estado_meteorologico", "distrito"]


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, full: bool) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    hour = df["hour"].astype(float)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["n_people"] = df["n_people"].astype(float)
    for c in ["has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian"]:
        X[c] = df[c].astype(float)
    X["tipo_accidente"] = df["tipo_accidente"].fillna("Desconocido").astype(str)
    X["estado_meteorologico"] = (
        df["estado_meteorologico"].fillna("Se desconoce").astype(str)
    )

    if full:
        month = df["fecha"].dt.month.astype(float)
        weekday = df["fecha"].dt.weekday.astype(float)
        X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        X["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
        X["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
        X["is_weekend"] = (weekday >= 5).astype(float)
        X["n_vehicles"] = df["n_vehicles"].astype(float)
        X["n_conductores"] = df["n_conductores"].astype(float)
        X["n_pasajeros"] = df["n_pasajeros"].astype(float)
        X["n_peatones"] = df["n_peatones"].astype(float)
        X["coordenada_x_utm"] = df["coordenada_x_utm"].astype(float)
        X["coordenada_y_utm"] = df["coordenada_y_utm"].astype(float)
        X["distrito"] = df["distrito"].fillna("DESCONOCIDO").astype(str)

    y = df["severity_code"].astype(int)
    return X, y


def make_preprocessor(full: bool) -> ColumnTransformer:
    num = NUMERIC_FULL if full else NUMERIC_MINIMAL
    cat = CAT_FULL if full else CAT_MINIMAL
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


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
            name: float(r) for name, r in zip(
                LABEL_NAMES,
                recall_score(y_true, y_pred, labels=LABELS,
                             average=None, zero_division=0),
            )
        },
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
        "n_samples": int(len(y_true)),
        "n_fatal_true": int((y_true == 3).sum()),
    }
    # One-vs-rest fatal
    if (y_true == 3).sum() > 0:
        result["pr_auc_fatal"] = float(
            average_precision_score((y_true == 3).astype(int), y_proba[:, 3]))
    else:
        result["pr_auc_fatal"] = None
    # Multiclass macro PR-AUC (average_precision_score macro)
    try:
        result["pr_auc_macro"] = float(
            average_precision_score(y_true, y_proba, average="macro"))
    except Exception:
        result["pr_auc_macro"] = None
    # Multiclass macro ROC-AUC (ovr)
    try:
        result["roc_auc_macro"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        result["roc_auc_macro"] = None
    return result


def bootstrap_ci(y_true, s, metric_fn, n_iter=500, seed=42,
                 require_positive_label=None):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if require_positive_label is not None and \
           (yt == require_positive_label).sum() < 2:
            continue
        try:
            v = metric_fn(yt, s[idx])
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
# Markdown helpers (avoid optional `tabulate` dependency)
# ---------------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def _cm_to_md(cm_df: pd.DataFrame) -> str:
    header = "| | " + " | ".join(str(c) for c in cm_df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cm_df.columns) + " |"
    rows = ["| " + str(idx) + " | "
            + " | ".join(str(v) for v in row) + " |"
            for idx, row in cm_df.iterrows()]
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
    if df["split"].isna().any():
        raise RuntimeError("Accidents without split assignment.")
    print(f"  {len(df):,} accidents.")

    X_min, y = build_features(df, full=False)
    X_full, _ = build_features(df, full=True)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    train_years = df.loc[train_mask, "year"].astype(int).to_numpy()
    print(f"  train={train_mask.sum():,}  val={val_mask.sum():,}  "
          f"test={test_mask.sum():,}")
    print(f"  train years: {sorted(set(train_years))}")

    y_train = y.loc[train_mask].to_numpy()
    y_val = y.loc[val_mask].to_numpy()
    y_test = y.loc[test_mask].to_numpy()

    logo = LeaveOneGroupOut()
    cv = list(logo.split(np.zeros(len(y_train)), y_train, groups=train_years))
    print(f"  LeaveOneGroupOut folds: {len(cv)}")

    def _prep(Xset, full):
        pre = make_preprocessor(full)
        Xt_train = pre.fit_transform(Xset.loc[train_mask])
        Xt_val = pre.transform(Xset.loc[val_mask])
        Xt_test = pre.transform(Xset.loc[test_mask])
        return Xt_train, Xt_val, Xt_test, pre

    # -----------------------------------------------------------------------
    # M1 / M2: default LR
    # -----------------------------------------------------------------------
    print("\nM1: minimal features, default LR")
    Xt_tr_m, Xt_va_m, Xt_te_m, _ = _prep(X_min, full=False)
    lr_m = LogisticRegression(max_iter=5000, random_state=42)
    lr_m.fit(Xt_tr_m, y_train)
    m1_val = evaluate(y_val, lr_m.predict(Xt_va_m), lr_m.predict_proba(Xt_va_m))
    m1_test = evaluate(y_test, lr_m.predict(Xt_te_m), lr_m.predict_proba(Xt_te_m))

    print("M2: full D031 features, default LR")
    Xt_tr_f, Xt_va_f, Xt_te_f, _ = _prep(X_full, full=True)
    lr_f = LogisticRegression(max_iter=5000, random_state=42)
    lr_f.fit(Xt_tr_f, y_train)
    m2_val = evaluate(y_val, lr_f.predict(Xt_va_f), lr_f.predict_proba(Xt_va_f))
    m2_test = evaluate(y_test, lr_f.predict(Xt_te_f), lr_f.predict_proba(Xt_te_f))

    # -----------------------------------------------------------------------
    # M3 / M4: GridSearchCV with LeaveOneGroupOut
    # -----------------------------------------------------------------------
    kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
    param_grid = {
        "C": [0.01, 0.1, 0.5, 1.0, 5, 10.0],
        "class_weight": [None, "balanced"],
    }

    print("\nM3: GridSearchCV on minimal features (24 fits expected)...")
    t0 = time.time()
    gs_min = GridSearchCV(
        LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42),
        param_grid, cv=cv, scoring=kappa_scorer, n_jobs=-1, refit=True,
    )
    gs_min.fit(Xt_tr_m, y_train, groups=train_years)
    print(f"  best={gs_min.best_params_}  cv_kappa={gs_min.best_score_:.4f}  "
          f"({time.time() - t0:.1f}s)")
    m3_val = evaluate(y_val, gs_min.predict(Xt_va_m), gs_min.predict_proba(Xt_va_m))
    m3_test = evaluate(y_test, gs_min.predict(Xt_te_m), gs_min.predict_proba(Xt_te_m))

    print("\nM4: GridSearchCV on full features (24 fits expected)...")
    t0 = time.time()
    gs_full = GridSearchCV(
        LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42),
        param_grid, cv=cv, scoring=kappa_scorer, n_jobs=-1, refit=True,
    )
    gs_full.fit(Xt_tr_f, y_train, groups=train_years)
    print(f"  best={gs_full.best_params_}  cv_kappa={gs_full.best_score_:.4f}  "
          f"({time.time() - t0:.1f}s)")
    m4_val = evaluate(y_val, gs_full.predict(Xt_va_f), gs_full.predict_proba(Xt_va_f))
    m4_test = evaluate(y_test, gs_full.predict(Xt_te_f), gs_full.predict_proba(Xt_te_f))

    # -----------------------------------------------------------------------
    # Pick best model by test kappa (already seeing test - final decision in Fase 5)
    # -----------------------------------------------------------------------
    models = {
        "M1_minimal_default": {"val": m1_val, "test": m1_test},
        "M2_full_default": {"val": m2_val, "test": m2_test},
        "M3_minimal_tuned": {"val": m3_val, "test": m3_test,
                             "best_params": gs_min.best_params_,
                             "cv_kappa": gs_min.best_score_},
        "M4_full_tuned": {"val": m4_val, "test": m4_test,
                          "best_params": gs_full.best_params_,
                          "cv_kappa": gs_full.best_score_},
    }
    best_name = max(
        ["M1_minimal_default", "M2_full_default", "M3_minimal_tuned", "M4_full_tuned"],
        key=lambda k: models[k]["test"]["kappa_quadratic"],
    )
    print(f"\nBest by test kappa: {best_name}")

    # Bootstrap the best model on test
    print("Bootstrapping best model on test...")
    if best_name == "M1_minimal_default":
        proba_best = lr_m.predict_proba(Xt_te_m); pred_best = lr_m.predict(Xt_te_m)
    elif best_name == "M2_full_default":
        proba_best = lr_f.predict_proba(Xt_te_f); pred_best = lr_f.predict(Xt_te_f)
    elif best_name == "M3_minimal_tuned":
        proba_best = gs_min.predict_proba(Xt_te_m); pred_best = gs_min.predict(Xt_te_m)
    else:
        proba_best = gs_full.predict_proba(Xt_te_f); pred_best = gs_full.predict(Xt_te_f)

    f1_macro_fn = lambda yt, yp: f1_score(  # noqa: E731
        yt, yp, average="macro", zero_division=0)
    pr_auc_fn = lambda yt, p: (  # noqa: E731
        average_precision_score((yt == 3).astype(int), p)
        if (yt == 3).sum() > 0 else None)

    bootstrap_best = {
        "f1_macro": bootstrap_ci(y_test, pred_best, f1_macro_fn),
        "pr_auc_fatal": bootstrap_ci(y_test, proba_best[:, 3], pr_auc_fn,
                                     require_positive_label=3),
    }

    # -----------------------------------------------------------------------
    # Write reports
    # -----------------------------------------------------------------------
    report = {
        "splits": {"train": int(train_mask.sum()),
                   "val": int(val_mask.sum()),
                   "test": int(test_mask.sum())},
        "train_years": sorted(set(int(y) for y in train_years)),
        "models": {k: v for k, v in models.items()},
        "best_model": best_name,
        "bootstrap_best_test": bootstrap_best,
    }
    save_json(report, OUT / "10_baselines_tuned.json")

    rows = []
    for name in ["M1_minimal_default", "M2_full_default",
                 "M3_minimal_tuned", "M4_full_tuned"]:
        for split in ("val", "test"):
            m = models[name][split]
            rows.append({
                "model": name, "split": split,
                "kappa_q": round(m["kappa_quadratic"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "bal_acc": round(m["balanced_accuracy"], 4),
                "mae_ord": round(m["mae_ordinal"], 4),
                "pr_auc_macro": (round(m["pr_auc_macro"], 4)
                                 if m["pr_auc_macro"] is not None else None),
                "pr_auc_fatal": (round(m["pr_auc_fatal"], 5)
                                 if m["pr_auc_fatal"] is not None else None),
                "roc_auc_macro": (round(m["roc_auc_macro"], 4)
                                  if m["roc_auc_macro"] is not None else None),
                "rec_fatal": round(m["recall_per_class"]["fatal"], 3),
            })
    comparison = pd.DataFrame(rows)

    best_cm = models[best_name]["test"]["confusion_matrix"]
    cm_df = pd.DataFrame(best_cm["matrix"], index=best_cm["labels"],
                          columns=best_cm["labels"])

    sections = {
        "Ablation — features vs. tuning (val and test)": _df_to_md(comparison),
        "Best model by test kappa": (
            f"**{best_name}**\n\n"
            f"Best params: {dumps_jsonable(models[best_name].get('best_params', 'default'))}"
            f"\n\nCV kappa (LeaveOneGroupOut): "
            f"{models[best_name].get('cv_kappa', 'N/A')}"
        ),
        "Bootstrap 95% CI (best model, test)":
            dumps_jsonable(bootstrap_best, indent=2),
        "Confusion matrix (best model, test)": _cm_to_md(cm_df),
        "Notes": (
            "- `pr_auc_fatal` is one-vs-rest AP for the fatal class "
            "(the operationally relevant question).\n"
            "- `pr_auc_macro` is multiclass macro average of per-class AP.\n"
            "- `roc_auc_macro` is multiclass macro of ROC-AUC (ovr).\n"
            "- CV inside GridSearchCV is LeaveOneGroupOut over train years "
            "(2019–2022), NOT random KFold, to avoid temporal leakage."
        ),
    }
    save_markdown_report(sections, Path("reports/10_baselines_tuned.md"))
    print("Done. See reports/10_baselines_tuned.md.")


if __name__ == "__main__":
    main()