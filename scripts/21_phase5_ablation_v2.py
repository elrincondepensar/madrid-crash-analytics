"""
Phase 5 · Iteration 0b — Re-run the ablation with:
  - Fixed bootstrap (proper CIs for Lift@2, Recall@5, Recall@10 on fatal)
  - Larger val (2023+2024, 52 fatals) per D081
  - Test = 2025 (34 fatals)
  - Also: target-encode edge_name (high cardinality) to test if specific
    dangerous locations add signal beyond highway/maxspeed.

Run:
    python scripts/21_phase5_ablation_v2.py
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
    f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]
LABELS = [0, 1, 2, 3]

NUMERIC_M4 = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "is_weekend",
    "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
    "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
    "coordenada_x_utm", "coordenada_y_utm",
]
CAT_M4 = ["tipo_accidente", "estado_meteorologico", "distrito"]

NUMERIC_NETWORK = ["edge_maxspeed", "edge_lanes", "edge_length_m",
                   "edge_distance_m", "edge_highway_ordinal"]
CAT_NETWORK = ["edge_highway", "tipo_via"]


# ---------------------------------------------------------------------------
# Custom splits (D081)
# ---------------------------------------------------------------------------
def build_splits_v2(df: pd.DataFrame) -> pd.Series:
    year = df["year"].astype(int)
    return pd.Series(
        np.where(year <= 2022, "train",
                 np.where(year <= 2024, "val", "test")),
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Target encoding (out-of-fold)
# ---------------------------------------------------------------------------
def target_encode_series(series: pd.Series, target: pd.Series,
                         folds: list, prior_weight: float = 20.0) -> pd.Series:
    """Out-of-fold target encoding with smoothing."""
    global_mean = float(target.mean())
    out = pd.Series(index=series.index, dtype=float)
    for tr_idx, va_idx in folds:
        s_tr = series.iloc[tr_idx]
        y_tr = target.iloc[tr_idx]
        stats = y_tr.groupby(s_tr).agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"]
                    + global_mean * prior_weight) / (stats["count"] + prior_weight)
        out.iloc[va_idx] = s_tr.iloc[va_idx].map(smoothed).fillna(global_mean) \
            if False else series.iloc[va_idx].map(smoothed).fillna(global_mean)
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, with_network: bool,
                   with_target_enc: bool = False,
                   folds: list | None = None) -> pd.DataFrame:
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

    if with_target_enc and folds is not None:
        edge_name = df["edge_name"].fillna("unknown").astype(str)
        target = df["severity_code"].astype(float)
        X["edge_name_te"] = target_encode_series(edge_name, target, folds)
        via = df["nombre_via"].fillna("DESCONOCIDO").astype(str)
        X["nombre_via_te"] = target_encode_series(via, target, folds)

    return X


def make_preprocessor(with_network: bool, with_te: bool) -> ColumnTransformer:
    num = NUMERIC_M4 + (NUMERIC_NETWORK if with_network else [])
    cat = CAT_M4 + (CAT_NETWORK if with_network else [])
    if with_te:
        num = num + ["edge_name_te", "nombre_via_te"]
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), cat),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ranking_metrics(y_true, y_proba, target_class=3):
    n = len(y_true)
    n_pos = int((y_true == target_class).sum())
    order = np.argsort(-y_proba[:, target_class])
    out = {}
    for pct in (0.02, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        tp = int((y_true[order[:k]] == target_class).sum())
        out[f"k={int(pct*100)}%"] = {
            "k": int(k), "tp": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / max(1, n_pos), 4),
            "lift": round((tp / k) / max(1e-12, n_pos / n), 3),
        }
    return {"n_target": n_pos, "by_k": out}


def threshold_per_class(y_true, y_proba, classes=(1, 2, 3), grid=200):
    thresholds = {}
    for k in classes:
        y_k = (y_true == k).astype(int)
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.01, 0.99, grid):
            yp = (y_proba[:, k] >= t).astype(int)
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


def apply_thresholds(y_proba, thresholds):
    n = len(y_proba)
    scores = np.zeros((n, 4))
    for k in (1, 2, 3):
        scores[:, k] = y_proba[:, k] / max(1e-9, thresholds[k])
    return np.where(scores.max(axis=1) > 1.0, scores.argmax(axis=1), 0).astype(int)


def full_metrics(y_t, y_proba):
    """Ranking metrics + basic cls metrics (using top-class predictions)."""
    pred = y_proba.argmax(axis=1)
    return {
        "kappa_q": float(cohen_kappa_score(y_t, pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_t, pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "mae_ord": float(np.mean(np.abs(np.asarray(y_t) - pred))),
        "pr_auc_fatal": float(average_precision_score(
            (y_t == 3).astype(int), y_proba[:, 3])),
        "pr_auc_macro": float(average_precision_score(
            y_t, y_proba, average="macro")),
        "roc_auc_macro": float(roc_auc_score(
            y_t, y_proba, multi_class="ovr", average="macro")),
        "ranking_fatal": ranking_metrics(y_t, y_proba, 3),
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_metrics(y_true, y_proba, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {k: [] for k in ("Lift@2", "Lift@5", "Lift@10",
                              "Recall@2", "Recall@5", "Recall@10",
                              "PR_AUC_fatal")}
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if (yt == 3).sum() < 2:
            continue
        p = y_proba[idx]
        m = ranking_metrics(yt, p, 3)
        for label, key in [("Lift@2", "k=2%"), ("Lift@5", "k=5%"),
                            ("Lift@10", "k=10%")]:
            stats[label].append(m["by_k"][key]["lift"])
        for label, key in [("Recall@2", "k=2%"), ("Recall@5", "k=5%"),
                            ("Recall@10", "k=10%")]:
            stats[label].append(m["by_k"][key]["recall"])
        stats["PR_AUC_fatal"].append(
            average_precision_score((yt == 3).astype(int), p[:, 3])
        )
    out = {}
    for k, vals in stats.items():
        if not vals:
            out[k] = None
            continue
        arr = np.asarray(vals)
        out[k] = {
            "median": float(np.median(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading accidents + network features...")
    acc = pd.read_parquet(OUT_DATA / "accidents_clean.parquet")
    net = pd.read_parquet(OUT_DATA / "accidents_with_network_features.parquet")
    base_cols = set(acc.columns)
    net_extra = [c for c in net.columns
                 if c not in base_cols and c != "num_expediente"]
    df = acc.merge(net[["num_expediente"] + net_extra],
                    on="num_expediente", how="left")
    df = df.loc[df["edge_distance_m"].notna()
                & (df["edge_distance_m"] < 1000)].copy()
    df["split"] = build_splits_v2(df)
    print(f"  {len(df):,} accidents.")
    print(f"  splits: {df['split'].value_counts().to_dict()}")
    print(f"  fatal by split: "
          f"{df.groupby('split')['severity_code'].apply(lambda s: (s==3).sum()).to_dict()}")

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y = df["severity_code"].astype(int).to_numpy()
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    # Folds for target encoding (within train only, 5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx = np.where(train_mask)[0]
    folds_local = [(train_idx[tr], train_idx[va])
                    for tr, va in kf.split(train_idx)]
    # For val/test, use global mean (no leakage). We'll build TE only on train.

    configs = [
        ("M4_baseline", False, False),
        ("M4_plus_network", True, False),
        ("M4_plus_network_te", True, True),
    ]
    results = {}
    fitted = {}

    for name, with_net, with_te in configs:
        print(f"\n=== {name} (network={with_net}, target_enc={with_te}) ===")
        if with_te:
            X = build_features(df, with_network=with_net, with_target_enc=True,
                               folds=folds_local)
            # For val/test rows (not in any train fold), fill TE with global
            # mean of train.
            edge_name = df["edge_name"].fillna("unknown").astype(str)
            via = df["nombre_via"].fillna("DESCONOCIDO").astype(str)
            train_mean = float(y_train.mean())
            te_edge = pd.Series(train_mean, index=df.index)
            te_via = pd.Series(train_mean, index=df.index)
            for tr_local, va_local in folds_local:
                s_tr = edge_name.iloc[tr_local]
                y_tr = pd.Series(y[tr_local], index=s_tr.index)
                stats = y_tr.groupby(s_tr).agg(["mean", "count"])
                smoothed = (stats["mean"] * stats["count"]
                            + train_mean * 20.0) / (stats["count"] + 20.0)
                te_edge.iloc[va_local] = (edge_name.iloc[va_local]
                                           .map(smoothed).fillna(train_mean))
            for tr_local, va_local in folds_local:
                s_tr = via.iloc[tr_local]
                y_tr = pd.Series(y[tr_local], index=s_tr.index)
                stats = y_tr.groupby(s_tr).agg(["mean", "count"])
                smoothed = (stats["mean"] * stats["count"]
                            + train_mean * 20.0) / (stats["count"] + 20.0)
                te_via.iloc[va_local] = (via.iloc[va_local]
                                          .map(smoothed).fillna(train_mean))
            X["edge_name_te"] = te_edge
            X["nombre_via_te"] = te_via
        else:
            X = build_features(df, with_network=with_net)

        X_train = X.loc[train_mask]
        X_val = X.loc[val_mask]
        X_test = X.loc[test_mask]
        pre = make_preprocessor(with_network=with_net, with_te=with_te)
        Xt_train = pre.fit_transform(X_train)
        Xt_val = pre.transform(X_val)
        Xt_test = pre.transform(X_test)

        t0 = time.time()
        lr = LogisticRegression(C=10.0, max_iter=3000, random_state=42)
        lr.fit(Xt_train, y_train)
        print(f"  fit {time.time() - t0:.1f}s, features={Xt_train.shape[1]}")

        proba_val = lr.predict_proba(Xt_val)
        proba_test = lr.predict_proba(Xt_test)
        m_val = full_metrics(y_val, proba_val)
        m_test = full_metrics(y_test, proba_test)
        print(f"  test kappa={m_test['kappa_q']:.3f}, "
              f"PR-AUC(fatal)={m_test['pr_auc_fatal']:.4f}, "
              f"Lift@2={m_test['ranking_fatal']['by_k']['k=2%']['lift']:.2f}")

        print("  bootstrapping...")
        ci = bootstrap_metrics(y_test, proba_test, n_iter=1000)
        results[name] = {"val": m_val, "test": m_test,
                          "bootstrap_test": ci}
        fitted[name] = proba_test

    # --- Comparison table ---
    def _row(name):
        m = results[name]["test"]
        rk = m["ranking_fatal"]["by_k"]
        return {
            "model": name,
            "kappa_q": round(m["kappa_q"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "pr_auc_fatal": round(m["pr_auc_fatal"], 5),
            "pr_auc_macro": round(m["pr_auc_macro"], 4),
            "Lift@2": rk["k=2%"]["lift"],
            "Lift@5": rk["k=5%"]["lift"],
            "Lift@10": rk["k=10%"]["lift"],
            "Recall@5": rk["k=5%"]["recall"],
            "Recall@10": rk["k=10%"]["recall"],
        }
    comp = pd.DataFrame([_row(n) for n in results])

    # --- Delta table with CIs ---
    deltas = {}
    base_ci = results["M4_baseline"]["bootstrap_test"]
    for name in ("M4_plus_network", "M4_plus_network_te"):
        te_ci = results[name]["bootstrap_test"]
        delta = {}
        for k in ("Lift@2", "Lift@5", "Lift@10", "Recall@5", "Recall@10",
                  "PR_AUC_fatal"):
            if te_ci[k] is None or base_ci[k] is None:
                continue
            delta[k] = {
                "point": round(float(results[name]["test"]["pr_auc_fatal"]
                                       if k == "PR_AUC_fatal"
                                       else results[name]["test"]["ranking_fatal"]
                                            ["by_k"][
                                                {"Lift@2": "k=2%",
                                                 "Lift@5": "k=5%",
                                                 "Lift@10": "k=10%",
                                                 "Recall@5": "k=5%",
                                                 "Recall@10": "k=10%",
                                                 }[k]][
                                                "lift" if k.startswith("Lift") else "recall"]
                                      ) - float(
                                          results["M4_baseline"]["test"]["pr_auc_fatal"]
                                          if k == "PR_AUC_fatal"
                                          else results["M4_baseline"]["test"]["ranking_fatal"]
                                               ["by_k"][
                                                   {"Lift@2": "k=2%",
                                                    "Lift@5": "k=5%",
                                                    "Lift@10": "k=10%",
                                                    "Recall@5": "k=5%",
                                                    "Recall@10": "k=10%",
                                                    }[k]][
                                                   "lift" if k.startswith("Lift") else "recall"]
                                       ), 4),
                "ci95_delta_low": round(te_ci[k]["ci95_low"]
                                        - base_ci[k]["ci95_high"], 4),
                "ci95_delta_high": round(te_ci[k]["ci95_high"]
                                          - base_ci[k]["ci95_low"], 4),
            }
        deltas[name] = delta
    results["_deltas_vs_baseline"] = deltas

    save_json(results, OUT_TABLES / "21_ablation_v2.json")

    def _df_to_md(df_):
        header = "| " + " | ".join(str(c) for c in df_.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df_.columns) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |"
                for row in df_.itertuples(index=False)]
        return "\n".join([header, sep] + rows)

    sections = {
        "Comparison (test = 2025)": _df_to_md(comp),
        "Bootstrap 95% CI (test, 1000 iters)": dumps_jsonable(
            {n: results[n]["bootstrap_test"] for n in
             ("M4_baseline", "M4_plus_network", "M4_plus_network_te")},
            indent=2),
        "Deltas vs baseline (with CI)": dumps_jsonable(deltas, indent=2),
        "Notes": (
            "- Splits (D081): train 2019-22, val 2023-24, test 2025.\n"
            "- Test has ~34 fatal events (vs ~60 in original).\n"
            "- Bootstrap 1000 iterations, resampling with replacement.\n"
            "- `M4_plus_network_te` adds out-of-fold target encoding of "
            "edge_name and nombre_via."
        ),
    }
    save_markdown_report(sections, Path("reports/21_ablation_v2.md"))
    print("\nDone. See reports/21_ablation_v2.md.")


if __name__ == "__main__":
    main()