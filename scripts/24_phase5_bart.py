"""
Phase 5 · Iteration 5 — BART for Opción A (4 clases).

pymc-bart solo soporta binario, así que se entrena **un BART por clase**
(4 BARTs one-vs-rest) y se normalizan las 4 probabilidades para obtener
P(clase | X). Estrategia estándar de one-vs-rest multiclass.

Configuración:
  FAST_SMOKE_TEST = True  → solo fatal, 40 top features, 50 samples,
                             1 chain, m=20 trees. Tiempo: ~5-10 min.
  FAST_SMOKE_TEST = False → 4 clases, 102 features, 300 samples,
                             2 chains, m=50 trees. Tiempo: ~1-3 h.

Reduce dimensionalidad a top-K via LGBM feature importance (rápido) para
acelerar el sampling MCMC. K configurable.

Run:
    pip install pymc-bart pymc arviz
    python scripts/24_phase5_bart.py
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
FAST_SMOKE_TEST = False   # True → solo fatal, top-40, 50 samples. False → full.

N_SAMPLES = 50 if FAST_SMOKE_TEST else 300
N_TUNE = 100 if FAST_SMOKE_TEST else 300
N_CHAINS = 1 if FAST_SMOKE_TEST else 2
N_CORES = 1 if FAST_SMOKE_TEST else 4
N_TREES = 20 if FAST_SMOKE_TEST else 50
TOP_FEATURES = 40 if FAST_SMOKE_TEST else None  # None → todas

# Clases a predecir: si smoke test, solo fatal
TARGET_CLASSES = [3] if FAST_SMOKE_TEST else [0, 1, 2, 3]
CLASS_LABELS = {0: "no_injury", 1: "minor", 2: "serious", 3: "fatal"}

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

LABELS_4 = [0, 1, 2, 3]
LABEL_NAMES = ["no_injury", "minor", "serious", "fatal"]

# Features (idénticas a script 23)
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
# Data prep (copia literal de script 23)
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
                    train_mask: np.ndarray, y_train_code: np.ndarray,
                    smoothing: float = 20.0) -> None:
    """Fit-on-train TE (target = P(grave)), igual que scripts 22/23/25."""
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
# Feature selection rápida (LGBM importance) para acelerar BART
# ===========================================================================
def select_top_features(Xt_train: np.ndarray, y_train: np.ndarray,
                         feature_names: list[str], top_k: int | None
                         ) -> tuple[np.ndarray, list[str]]:
    if top_k is None:
        return Xt_train, feature_names
    from lightgbm import LGBMClassifier
    print(f"  [feature selection] LGBM rápido para top-{top_k}...")
    m = LGBMClassifier(n_estimators=200, max_depth=8, num_leaves=31,
                        n_jobs=-1, verbose=-1, random_state=42,
                        objective="multiclass", num_class=4)
    m.fit(Xt_train, y_train)
    imp = m.feature_importances_
    idx = np.argsort(-imp)[:top_k]
    idx = np.sort(idx)
    print(f"  [feature selection] keep {len(idx)} features.")
    return Xt_train[:, idx], [feature_names[i] for i in idx]


# ===========================================================================
# Metrics (idénticas a script 23)
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
# BART one-vs-rest
# ===========================================================================
def fit_bart_binary(X_train, y_bin, X_test, n_trees, n_samples, n_tune,
                     n_chains, n_cores, seed=42, label=""):
    """
    Entrena un BART binario con pm.Data (contenedor mutable) y predice
    out-of-sample con pm.set_data. Esta API es estable en PyMC 4.x y 5.x
    y no depende de helpers específicos de pymc-bart que cambian entre
    versiones.

    Devuelve:
        p_test  : (n_test,) probabilidad media P(clase=1 | X_test)
        elapsed : tiempo de sampling en segundos
    """
    import pymc as pm
    import pymc_bart as pmb

    print(f"  [BART {label}] fitting on {len(X_train)} samples, "
          f"{n_trees} trees, {n_samples} draws, {n_chains} chains...")
    t0 = time.time()

    with pm.Model() as model:
        # pm.Data es un tensor mutable que BART puede usar como X.
        X_shared = pm.Data("X", X_train)
        # BART sobre la función latente mu (escala real)
        mu = pmb.BART("mu", X=X_shared, Y=y_bin, m=n_trees)
        # Link logit → probabilidad
        p = pm.Deterministic("p", pm.math.sigmoid(mu))
        pm.Bernoulli("y", p=p, observed=y_bin)

        idata = pm.sample(
            draws=n_samples, tune=n_tune, chains=n_chains, cores=n_cores,
            random_seed=seed, progressbar=False,
            compute_convergence_checks=False,
        )
    elapsed = time.time() - t0
    print(f"  [BART {label}] sampling done in {elapsed:.0f}s")

    # --- Predicción out-of-sample con pm.set_data ---------------------
    with model:
        pm.set_data({"X": X_test})
        pp = pm.sample_posterior_predictive(
            idata,
            var_names=["p"],
            random_seed=seed,
            progressbar=False,
        )
    # pp.posterior_predictive["p"]: shape (chain, draw, n_test)
    p_test = (pp.posterior_predictive["p"]
                .mean(dim=("chain", "draw"))
                .values)
    return p_test, elapsed


def fit_bart_ovr(X_train, y_train, X_test, target_classes,
                  n_trees, n_samples, n_tune, n_chains, n_cores, seed=42):
    """
    Ajusta un BART por cada clase en target_classes.
    Devuelve y_proba_test (n_test, 4) normalizada.
    """
    n_test = X_test.shape[0]
    probas = np.zeros((n_test, 4), dtype=float)
    fits = {}
    for k in target_classes:
        y_bin = (y_train == k).astype(int)
        # Sanity: si no hay positivos, saltamos
        if y_bin.sum() < 10:
            print(f"  [BART {CLASS_LABELS[k]}] only {y_bin.sum()} positives, "
                  f"skipping (proba=0)")
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

    # Normalizar (suma = 1 por fila) con suavizado para evitar 0/0
    row_sum = probas.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probas_norm = probas / row_sum
    return probas_norm, probas, fits


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
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    print("\nBuilding features (M2 + red + TE)...")
    X = build_features(df, with_network=True)
    add_te_columns(X, df, train_mask, y_train)
    pre = make_preprocessor(with_network=True)
    Xt_train_full = pre.fit_transform(X.loc[train_mask])
    Xt_val = pre.transform(X.loc[val_mask])
    Xt_test = pre.transform(X.loc[test_mask])
    print(f"  features: {Xt_train_full.shape[1]}")

    # Nombres de columna generados por ColumnTransformer (para importance)
    def _get_feature_names(ct, num_cols, cat_cols):
        names = list(num_cols)
        # Para OHE, los nombres reales vienen del encoder
        ohe = ct.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        # Indicadores del imputer (add_indicator=True)
        imp = ct.named_transformers_["num"].named_steps["imp"]
        if hasattr(imp, "indicator_"):
            idx_ = np.asarray(imp.indicator_.features_)
            ind_names = [f"{num_cols[i]}_missing" for i in idx_]
            names = names + ind_names
        return names + cat_names

    num = NUMERIC_M2 + NUMERIC_NETWORK
    cat = CAT_M2 + CAT_NETWORK
    feat_names = _get_feature_names(pre, num, cat)
    if len(feat_names) != Xt_train_full.shape[1]:
        # Fallback: nombres genéricos
        feat_names = [f"f{i}" for i in range(Xt_train_full.shape[1])]

    # Reducción dimensional
    Xt_train, feat_used = select_top_features(
        Xt_train_full, y_train, feat_names, top_k=TOP_FEATURES
    )
    if TOP_FEATURES is not None:
        # Aplicar la misma selección a val/test
        mask = np.isin(np.arange(Xt_train_full.shape[1]),
                        [feat_names.index(f) for f in feat_used])
        Xt_val = Xt_val[:, mask]
        Xt_test = Xt_test[:, mask]
    print(f"  final features: {Xt_train.shape[1]}")

    print(f"\nBART config: classes={TARGET_CLASSES}, trees={N_TREES}, "
          f"samples={N_SAMPLES}, tune={N_TUNE}, chains={N_CHAINS}")
    if FAST_SMOKE_TEST:
        print("  *** FAST_SMOKE_TEST activo — solo fatal, rápido ***")

    print("\nFitting BART (one-vs-rest)...")
    probas_test, probas_raw_test, fits = fit_bart_ovr(
        Xt_train, y_train, Xt_test,
        target_classes=TARGET_CLASSES,
        n_trees=N_TREES, n_samples=N_SAMPLES, n_tune=N_TUNE,
        n_chains=N_CHAINS, n_cores=N_CORES, seed=42,
    )

    # Evaluación: solo sobre clase fatal si smoke test
    if FAST_SMOKE_TEST:
        # Solo la columna fatal tiene señal real
        p_fatal = probas_raw_test[:, 3]
        pr_auc = float(average_precision_score(
            (y_test == 3).astype(int), p_fatal))
        rk = ranking_metrics(y_test, np.column_stack(
            [np.zeros((len(y_test), 3)), p_fatal]), target=3)
        boot = bootstrap_ranking(
            y_test, np.column_stack([np.zeros((len(y_test), 3)), p_fatal]),
            n_iter=500)
        result = {
            "mode": "smoke",
            "pr_auc_fatal": round(pr_auc, 4),
            "ranking_fatal": rk,
            "bootstrap": boot,
            "fits": fits,
        }
        print(f"\n  BART smoke: PR-AUC(fatal)={pr_auc:.4f}  "
              f"Lift@2={rk['by_k']['k=2%']['lift']:.2f}  "
              f"Recall@10={rk['by_k']['k=10%']['recall']:.3f}")
    else:
        m_test = full_metrics(y_test, probas_test)
        boot = bootstrap_ranking(y_test, probas_test, n_iter=500)
        result = {
            "mode": "full",
            "test": m_test,
            "bootstrap": boot,
            "fits": fits,
        }
        rk = m_test["ranking_fatal"]["by_k"]
        print(f"\n  BART full: kappa={m_test['kappa_q']:.3f}  "
              f"PR-AUC(fatal)={m_test['pr_auc_fatal']:.4f}  "
              f"Lift@2={rk['k=2%']['lift']:.2f}  "
              f"Recall@10={rk['k=10%']['recall']:.3f}")

    save_json(result, OUT_TABLES / "24_bart.json")

    # Markdown
    sections = {
        "BART result": dumps_jsonable(result, indent=2),
        "Reference: Opción A (script 23, test=2025)": (
            "LR baseline: PR-AUC(fatal)=0.0089, Lift@2=10.9, Recall@10=0.50\n"
            "lgbm__borderline: PR-AUC(fatal)=0.0086, Lift@2=6.24, Recall@10=0.44\n"
            "catboost__smote: PR-AUC(fatal)=0.0075, Lift@2=9.37, Recall@10=0.50"
        ),
        "Notes": (
            "BART one-vs-rest (4 BARTs binarios, probabilidades normalizadas).\n"
            f"Config: trees={N_TREES}, samples={N_SAMPLES}, "
            f"chains={N_CHAINS}, top_features={TOP_FEATURES}.\n"
            "Si FAST_SMOKE_TEST=True, solo se entrena fatal (clase 3) y "
            "la evaluación es binaria (PR-AUC + Lift@k sobre fatal).\n"
            "Feature selection: LGBM importance, top-K reducido."
        ),
    }
    save_markdown_report(sections, Path("reports/24_bart.md"))
    print("\nDone. See reports/24_bart.md.")


if __name__ == "__main__":
    main()