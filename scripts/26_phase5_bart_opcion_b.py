"""
Phase 5 · Iteration 6 — Opción B: BART + LR baseline.

Target: 3 clases
    0 = no_injury
    1 = minor
    2 = grave (serious ∪ fatal)

Modelos:
    - LR(C=10) sobre features completas (preprocesado M4)
    - BART one-vs-rest (3 BARTs binarios), mismas features que LR

Ambos modelos:
    - Mismo train/val/test (D081: train 2019-22, val 2023-24, test 2025)
    - Mismas features (M2 + numero + red + TE)
    - Misma imputación (median + indicator)
    - Mismas métricas (kappa_q, F1-macro, MAE ordinal, PR-AUC(grave),
      ROC-AUC(grave), recall por clase, fatal como ranking con P(grave))

Configuración:
    FAST_SMOKE_TEST=True  → BART solo grave (clase 2), top-40 features,
                            20 trees, 50 draws, 1 chain, ~3 min.
                            LR se entrena igual (todas las features).
    FAST_SMOKE_TEST=False → BART 3 clases, features completas,
                            50 trees, 300 draws, 2 chains, ~45 min.
                            LR se entrena igual.

Run:
    pip install pymc-bart pymc arviz lightgbm
    python scripts/26_phase5_bart_opcion_b.py
"""
from __future__ import annotations

import sys
import time
import warnings
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")

# ===========================================================================
# Configuración
# ===========================================================================
FAST_SMOKE_TEST = True  # True → validación rápida (~3 min)

if FAST_SMOKE_TEST:
    BART_TARGET_CLASSES = [2]     # solo grave
    BART_N_TREES = 20
    BART_N_SAMPLES = 50
    BART_N_TUNE = 100
    BART_N_CHAINS = 1
    BART_N_CORES = 1
    BART_TOP_FEATURES = 40
else:
    BART_TARGET_CLASSES = [0, 1, 2]  # 3 clases
    BART_N_TREES = 50
    BART_N_SAMPLES = 300
    BART_N_TUNE = 300
    BART_N_CHAINS = 2
    BART_N_CORES = 4
    BART_TOP_FEATURES = None         # todas

LR_C = 10.0
N_CLASSES = 3
LABELS_3 = [0, 1, 2]
CLASS_NAMES = ["no_injury", "minor", "grave"]
CLASS_LABELS = {0: "no_injury", 1: "minor", 2: "grave"}

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Features (idénticas a scripts 23 y 25)
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
                    train_mask: np.ndarray, y_train_code: np.ndarray,
                    smoothing: float = 20.0) -> None:
    """Target encoding fit-on-train sobre P(grave). Igual que scripts 22/23/25."""
    y_grave_train = (y_train_code >= 2).astype(float)
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
# Feature selection (solo para BART, en smoke test)
# ===========================================================================
def select_top_features(Xt_train: np.ndarray, y_train: np.ndarray,
                         top_k: int | None
                         ) -> tuple[np.ndarray, np.ndarray | None]:
    if top_k is None:
        return Xt_train, None
    from lightgbm import LGBMClassifier
    print(f"  [feature selection] LGBM rápido para top-{top_k}...")
    m = LGBMClassifier(
        n_estimators=200, max_depth=8, num_leaves=31,
        n_jobs=-1, verbose=-1, random_state=42,
        objective="multiclass", num_class=N_CLASSES,
    )
    m.fit(Xt_train, y_train)
    imp = m.feature_importances_
    idx = np.sort(np.argsort(-imp)[:top_k])
    print(f"  [feature selection] keep {len(idx)} features.")
    return Xt_train[:, idx], idx


# ===========================================================================
# Metrics (3 clases + fatal ranking)
# ===========================================================================
def fatal_ranking_from_grave_proba(y_true_code: np.ndarray,
                                     grave_proba: np.ndarray,
                                     target_fatal: int = 3) -> dict:
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
        "precision_grave": _ci(prec_grave),
    }


# ===========================================================================
# LR baseline
# ===========================================================================
def evaluate_lr_3class(Xt_train, Xt_val, Xt_test,
                        y_train, y_val, y_test,
                        y_code_val, y_code_test,
                        C: float = 10.0) -> dict:
    print("\n--- LR baseline (Opción B, C=10) ---")
    t0 = time.time()
    lr = LogisticRegression(C=C, max_iter=3000, random_state=42)
    lr.fit(Xt_train, y_train)
    fit_s = time.time() - t0
    print(f"  fit {fit_s:.1f}s")

    p_val = lr.predict_proba(Xt_val)
    p_test = lr.predict_proba(Xt_test)
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
        "model": "lr", "C": C, "fit_seconds": round(fit_s, 1),
        "val": m_val, "test": m_test,
        "fatal_ranking_val": fatal_val,
        "fatal_ranking_test": fatal_test,
        "bootstrap_test": boot,
    }


# ===========================================================================
# BART binary (pm.Data + pm.set_data — API estable)
# ===========================================================================
def fit_bart_binary(X_train, y_bin, X_test,
                     n_trees, n_samples, n_tune, n_chains, n_cores,
                     seed=42, label=""):
    import pymc as pm
    import pymc_bart as pmb

    print(f"  [BART {label}] fitting on {len(X_train)} samples, "
          f"{n_trees} trees, {n_samples} draws, {n_chains} chains...")
    t0 = time.time()
    with pm.Model() as model:
        X_shared = pm.Data("X", X_train)
        mu = pmb.BART("mu", X=X_shared, Y=y_bin, m=n_trees)
        p = pm.Deterministic("p", pm.math.sigmoid(mu))
        pm.Bernoulli("y", p=p, observed=y_bin)
        idata = pm.sample(
            draws=n_samples, tune=n_tune, chains=n_chains, cores=n_cores,
            random_seed=seed, progressbar=False,
            compute_convergence_checks=False,
        )
    elapsed = time.time() - t0
    print(f"  [BART {label}] sampling done in {elapsed:.0f}s")

    with model:
        pm.set_data({"X": X_test})
        pp = pm.sample_posterior_predictive(
            idata, var_names=["p"], random_seed=seed, progressbar=False,
        )
    p_test = (pp.posterior_predictive["p"]
                .mean(dim=("chain", "draw"))
                .values)
    return p_test, elapsed


def fit_bart_ovr(X_train, y_train, X_test, target_classes,
                  n_trees, n_samples, n_tune, n_chains, n_cores, seed=42):
    n_test = X_test.shape[0]
    probas = np.zeros((n_test, N_CLASSES), dtype=float)
    fits = {}
    for k in target_classes:
        y_bin = (y_train == k).astype(int)
        if y_bin.sum() < 10:
            print(f"  [BART {CLASS_LABELS[k]}] only {y_bin.sum()} positives, skip")
            continue
        p_test, elapsed = fit_bart_binary(
            X_train, y_bin, X_test,
            n_trees=n_trees, n_samples=n_samples, n_tune=n_tune,
            n_chains=n_chains, n_cores=n_cores,
            seed=seed, label=CLASS_LABELS[k],
        )
        probas[:, k] = p_test
        fits[CLASS_LABELS[k]] = {
            "n_positive_train": int(y_bin.sum()),
            "fit_seconds": round(elapsed, 1),
        }
    row_sum = probas.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return probas / row_sum, probas, fits


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
    y_code_val, y_code_test = y_code_all[val_mask], y_code_all[test_mask]

    print("\nBuilding features...")
    X = build_features(df, with_network=True)
    add_te_columns(X, df, train_mask, y_code_all[train_mask])
    pre = make_preprocessor(with_network=True)
    Xt_train_full = pre.fit_transform(X.loc[train_mask])
    Xt_val = pre.transform(X.loc[val_mask])
    Xt_test = pre.transform(X.loc[test_mask])
    print(f"  features: {Xt_train_full.shape[1]}")

    results = {}

    # ---------------- LR baseline (usa TODAS las features) --------------
    results["lr"] = evaluate_lr_3class(
        Xt_train_full, Xt_val, Xt_test,
        y_train, y_val, y_test,
        y_code_val, y_code_test,
        C=LR_C,
    )
    save_json(results, OUT_TABLES / "26_bart_opcion_b.json")

    # ---------------- BART one-vs-rest -------------------------------
    # En smoke test se reduce a top-K features para rapidez.
    Xt_train_bart, sel_idx = select_top_features(
        Xt_train_full, y_train, top_k=BART_TOP_FEATURES,
    )
    if sel_idx is not None:
        Xt_val_bart = Xt_val[:, sel_idx]
        Xt_test_bart = Xt_test[:, sel_idx]
    else:
        Xt_val_bart = Xt_val
        Xt_test_bart = Xt_test
    print(f"  BART features: {Xt_train_bart.shape[1]}")

    print(f"\nBART config: classes={BART_TARGET_CLASSES}, "
          f"trees={BART_N_TREES}, samples={BART_N_SAMPLES}, "
          f"chains={BART_N_CHAINS}")
    if FAST_SMOKE_TEST:
        print("  *** FAST_SMOKE_TEST — BART solo grave ***")

    print("\nFitting BART (one-vs-rest)...")
    probas_test, probas_raw_test, bart_fits = fit_bart_ovr(
        Xt_train_bart, y_train, Xt_test_bart,
        target_classes=BART_TARGET_CLASSES,
        n_trees=BART_N_TREES, n_samples=BART_N_SAMPLES, n_tune=BART_N_TUNE,
        n_chains=BART_N_CHAINS, n_cores=BART_N_CORES, seed=42,
    )

    if FAST_SMOKE_TEST:
        # Solo clase grave entrenada: evaluar PR-AUC(grave) + fatal ranking
        p_grave = probas_raw_test[:, 2]
        pr_auc = float(average_precision_score(
            (y_test == 2).astype(int), p_grave))
        fatal_rk = fatal_ranking_from_grave_proba(y_code_test, p_grave)
        results["bart"] = {
            "model": "bart", "mode": "smoke",
            "n_trees": BART_N_TREES, "n_samples": BART_N_SAMPLES,
            "n_chains": BART_N_CHAINS,
            "features_used": Xt_train_bart.shape[1],
            "fits": bart_fits,
            "pr_auc_grave": round(pr_auc, 4),
            "fatal_ranking_test": fatal_rk,
        }
        print(f"\n  BART smoke: PR-AUC(grave)={pr_auc:.4f}  "
              f"fatal_Lift@2={fatal_rk['by_k']['k=2%']['lift']:.2f}  "
              f"fatal_Recall@10={fatal_rk['by_k']['k=10%']['recall']:.3f}")
    else:
        m_test = full_metrics_3class(y_test, probas_test)
        boot = bootstrap_3class(y_test, probas_test, n_iter=500)
        fatal_rk = fatal_ranking_from_grave_proba(y_code_test, probas_test[:, 2])
        results["bart"] = {
            "model": "bart", "mode": "full",
            "n_trees": BART_N_TREES, "n_samples": BART_N_SAMPLES,
            "n_chains": BART_N_CHAINS,
            "features_used": Xt_train_bart.shape[1],
            "fits": bart_fits,
            "test": m_test,
            "bootstrap_test": boot,
            "fatal_ranking_test": fatal_rk,
        }
        print(f"\n  BART full: kappa={m_test['kappa_q']:.3f}  "
              f"PR-AUC(grave)={m_test['pr_auc_grave']:.4f}  "
              f"rec_grave={m_test['recall']['grave']:.3f}  "
              f"fatal_Lift@2={fatal_rk['by_k']['k=2%']['lift']:.2f}")

    save_json(results, OUT_TABLES / "26_bart_opcion_b.json")

    # ---------------- Comparison table ---------------------------------
    rows = []
    # LR
    lr_test = results["lr"]["test"]
    lr_fk = results["lr"]["fatal_ranking_test"]["by_k"]
    rows.append({
        "model": "LR (C=10)",
        "kappa_q": round(lr_test["kappa_q"], 4),
        "f1_macro": round(lr_test["f1_macro"], 4),
        "mae_ord": round(lr_test["mae_ord"], 4),
        "pr_auc_grave": round(lr_test["pr_auc_grave"], 4),
        "roc_auc_grave": round(lr_test["roc_auc_grave"], 4),
        "rec_grave": round(lr_test["recall"]["grave"], 3),
        "rec_minor": round(lr_test["recall"]["minor"], 3),
        "rec_no_inj": round(lr_test["recall"]["no_injury"], 3),
        "fatal_Lift@2": lr_fk["k=2%"]["lift"],
        "fatal_Lift@5": lr_fk["k=5%"]["lift"],
        "fatal_Recall@10": lr_fk["k=10%"]["recall"],
    })
    # BART (puede ser smoke o full)
    bart = results["bart"]
    if bart["mode"] == "full":
        bt = bart["test"]
        bfk = bart["fatal_ranking_test"]["by_k"]
        rows.append({
            "model": "BART (OvR)",
            "kappa_q": round(bt["kappa_q"], 4),
            "f1_macro": round(bt["f1_macro"], 4),
            "mae_ord": round(bt["mae_ord"], 4),
            "pr_auc_grave": round(bt["pr_auc_grave"], 4),
            "roc_auc_grave": round(bt["roc_auc_grave"], 4),
            "rec_grave": round(bt["recall"]["grave"], 3),
            "rec_minor": round(bt["recall"]["minor"], 3),
            "rec_no_inj": round(bt["recall"]["no_injury"], 3),
            "fatal_Lift@2": bfk["k=2%"]["lift"],
            "fatal_Lift@5": bfk["k=5%"]["lift"],
            "fatal_Recall@10": bfk["k=10%"]["recall"],
        })
    else:
        # smoke: solo métricas que BART ha calculado
        bfk = bart["fatal_ranking_test"]["by_k"]
        rows.append({
            "model": "BART (smoke, solo grave)",
            "kappa_q": None, "f1_macro": None, "mae_ord": None,
            "pr_auc_grave": bart["pr_auc_grave"],
            "roc_auc_grave": None,
            "rec_grave": None, "rec_minor": None, "rec_no_inj": None,
            "fatal_Lift@2": bfk["k=2%"]["lift"],
            "fatal_Lift@5": bfk["k=5%"]["lift"],
            "fatal_Recall@10": bfk["k=10%"]["recall"],
        })
    comp = pd.DataFrame(rows)

    sections = {
        "Comparación LR vs BART (test=2025)": _md(comp),
        "Bootstrap CI — LR": dumps_jsonable(
            results["lr"]["bootstrap_test"], indent=2),
        "Bootstrap CI — BART": (
            dumps_jsonable(results["bart"].get("bootstrap_test"), indent=2)
            if bart["mode"] == "full" else "N/A (smoke)"
        ),
        "Fatal ranking — LR": dumps_jsonable(
            results["lr"]["fatal_ranking_test"], indent=2),
        "Fatal ranking — BART": dumps_jsonable(
            bart["fatal_ranking_test"], indent=2),
        "BART fits (por clase)": dumps_jsonable(bart["fits"], indent=2),
        "Notes": (
            "Target: 0=no_injury, 1=minor, 2=grave (serious ∪ fatal).\n"
            "Mismo preprocesado (M4: imputer + OHE), fit en train.\n"
            "Mismas features (M2 + numero + red + TE edge_name/nombre_via).\n"
            "LR usa todas las features. BART usa todas en full, "
            "top-K en smoke.\n"
            "fatal se evalúa como ranking usando P(grave) como score."
        ),
    }
    save_markdown_report(sections, Path("reports/26_bart_opcion_b.md"))
    print("\nDone. See reports/26_bart_opcion_b.md")


if __name__ == "__main__":
    main()