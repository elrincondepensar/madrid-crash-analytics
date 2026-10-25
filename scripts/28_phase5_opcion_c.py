"""
Phase 5 · Iteration 8 — Opción C modificada (binario injured vs no_injury)
con búsqueda de hiperparámetros para los 4 modelos y log de progreso
trial-a-trial.

Modelos:
    LR       : RandomizedSearchCV (30 trials) + LeaveOneGroupOut(año)
    LGBM     : RandomizedSearchCV (20 trials)
    CatBoost : RandomizedSearchCV (12 trials)
    BART     : grid manual (m, alpha, beta) sobre subconjunto de train,
               refit en train completo con best

Log de progreso:
    Cada trial imprime:
      [i/N] best=<f1_so_far>  elapsed=<s>  ETA=<s>
    BART imprime "trying config i/N" con ETA similar.
    Al final de cada modelo se imprime la evaluación en val y test.

Run:
    python scripts/28_phase5_opcion_c.py
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
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut, ParameterGrid, ParameterSampler, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import check_random_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")

# ===========================================================================
# Configuración
# ===========================================================================
FAST_SMOKE_TEST = False

N_ITER = {
    "lr":       30 if not FAST_SMOKE_TEST else 4,
    "lgbm":     20 if not FAST_SMOKE_TEST else 3,
    "catboost": 12 if not FAST_SMOKE_TEST else 2,
}
BART_GRID = ([(50, 0.95, 2.0), (100, 0.95, 2.0)]
              if not FAST_SMOKE_TEST else [(50, 0.95, 2.0)])
BART_SEARCH_SAMPLES = 10000
BART_SEARCH_DRAWS = 200
BART_SEARCH_CHAINS = 1
BART_REFIT_DRAWS = 300
BART_REFIT_CHAINS = 2
BART_REFIT_CORES = 4
INCLUDE_BART = False

N_JOBS = 8
RANDOM_SEED = 42

CLASS_NAMES = ["no_injury", "injured"]
LABELS = [0, 1]

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# ProgressiveRandomizedSearchCV — log trial a trial
# ===========================================================================
class ProgressiveRandomizedSearchCV(RandomizedSearchCV):
    """
    Subclase de RandomizedSearchCV que evalúa un trial a la vez y loguea
    [i/N] con ETA.

    Captura los groups en fit() para que _run_search() pueda calcular
    el número de folds sin depender del estado interno de sklearn.
    """
    def fit(self, X, y=None, **fit_params):
        # Capturamos groups antes de delegar en el RandomizedSearchCV padre
        self._log_groups = fit_params.get("groups", None)
        return super().fit(X, y, **fit_params)

    def _run_search(self, evaluate_candidates):
        rng = check_random_state(self.random_state)
        candidates = list(ParameterSampler(
            self.param_distributions, self.n_iter, random_state=rng,
        ))
        total = len(candidates)

        # Número de folds: pasa groups si lo tenemos (LOGO lo exige)
        try:
            n_splits = (self.cv.get_n_splits(self._log_groups)
                        if self._log_groups is not None
                        else self.cv.get_n_splits())
        except Exception:
            n_splits = "?"

        start = time.time()
        best_score = -np.inf
        print(f"  Searching {total} candidates × "
              f"{n_splits} folds = "
              f"{total * n_splits if isinstance(n_splits, int) else '?'} fits total.",
              flush=True)

        for i, params in enumerate(candidates):
            t0 = time.time()
            evaluate_candidates([params])
            try:
                scores = self.cv_results_["mean_test_score"][:i + 1]
                scores = scores[~np.isnan(scores)]
                best_score = float(np.max(scores)) if len(scores) else -np.inf
            except Exception:
                pass
            elapsed = time.time() - start
            trial_s = time.time() - t0
            remaining = total - (i + 1)
            eta = elapsed / (i + 1) * remaining if (i + 1) > 0 else 0
            print(f"  [{i+1:>2}/{total}] best_f1={best_score:.4f} "
                  f"| trial={trial_s:>5.1f}s elapsed={elapsed:>6.0f}s "
                  f"ETA={eta:>6.0f}s", flush=True)


# ===========================================================================
# Features
# ===========================================================================
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


def build_splits_v2(year):
    return pd.Series(
        np.where(year <= 2022, "train",
                 np.where(year <= 2024, "val", "test")),
        index=year.index,
    )


def severity_to_binary(s):
    return (s >= 1).astype(int)


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


def add_te_columns(X, df, train_mask, y_train_bin, smoothing=20.0):
    global_mean = float(y_train_bin.mean())

    def _fit_apply(col_src, col_te):
        train_vals = (df.loc[train_mask, col_src]
                        .fillna("__unknown__").astype(str))
        stats = (pd.DataFrame({"s": train_vals.values, "y": y_train_bin})
                   .groupby("s")["y"].agg(["mean", "count"]))
        smoothed = ((stats["mean"] * stats["count"]
                     + global_mean * smoothing)
                    / (stats["count"] + smoothing))
        all_vals = df[col_src].fillna("__unknown__").astype(str)
        X[col_te] = all_vals.map(smoothed).fillna(global_mean).values

    _fit_apply("edge_name", "edge_name_te")
    _fit_apply("nombre_via", "nombre_via_te")


def make_column_transformer(with_scaler):
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
# Metrics
# ===========================================================================
def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "f1_injured": float(f1_score(y_true, y_pred, pos_label=1,
                                       zero_division=0)),
        "precision_injured": float(precision_score(y_true, y_pred,
                                                     pos_label=1,
                                                     zero_division=0)),
        "recall_injured": float(recall_score(y_true, y_pred, pos_label=1,
                                               zero_division=0)),
        "pr_auc_injured": float(average_precision_score(y_true, y_prob)),
        "roc_auc_injured": float(roc_auc_score(y_true, y_prob)),
        "confusion": {"labels": CLASS_NAMES,
                      "matrix": confusion_matrix(
                          y_true, y_pred, labels=LABELS).tolist()},
    }


def lift_recall_at_k(y_true, y_prob, ks=(0.05, 0.10, 0.20, 0.30, 0.50)):
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    base_rate = n_pos / n
    order = np.argsort(-y_prob)
    out = {"base_rate": round(base_rate, 4), "n_positive": n_pos, "by_k": {}}
    for k_frac in ks:
        k = max(1, int(np.ceil(n * k_frac)))
        top = order[:k]
        tp = int((y_true[top] == 1).sum())
        out["by_k"][f"k={int(k_frac*100)}%"] = {
            "k": k, "tp": tp,
            "injury_rate": round(tp / k, 4),
            "lift": round((tp / k) / base_rate, 3),
            "recall": round(tp / n_pos, 4),
        }
    return out


def tune_threshold(y_val, p_val):
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p_val >= t).astype(int)
        f1 = f1_score(y_val, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def bootstrap_ci(y_true, y_prob, threshold, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s, pr_aucs, rocs = [], [], []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, pp = y_true[idx], y_prob[idx]
        if (yt == 1).sum() < 2 or (yt == 0).sum() < 2:
            continue
        pred = (pp >= threshold).astype(int)
        f1s.append(f1_score(yt, pred, pos_label=1, zero_division=0))
        pr_aucs.append(average_precision_score(yt, pp))
        rocs.append(roc_auc_score(yt, pp))
    def _ci(vals):
        if not vals:
            return None
        arr = np.asarray(vals)
        return {"median": float(np.median(arr)),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5))}
    return {"f1_injured": _ci(f1s),
            "pr_auc_injured": _ci(pr_aucs),
            "roc_auc_injured": _ci(rocs)}


# ===========================================================================
# Tuning con progreso
# ===========================================================================
SCORER = make_scorer(f1_score, pos_label=1)


def tune_lr(X_train_df, y_train, groups):
    print(f"\n=== Tuning LR ({N_ITER['lr']} trials, LOGO {len(set(groups))}-fold) ===", flush=True)
    pre = make_column_transformer(with_scaler=True)
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=3000,
                                                  random_state=RANDOM_SEED,
                                                  n_jobs=1))])
    param_dist = {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
        "clf__class_weight": [None, "balanced"],
    }
    search = ProgressiveRandomizedSearchCV(
        pipe, param_dist, n_iter=N_ITER["lr"],
        cv=LeaveOneGroupOut(), scoring=SCORER,
        random_state=RANDOM_SEED, n_jobs=N_JOBS, refit=True, verbose=0,
    )
    search.fit(X_train_df, y_train, groups=groups)
    print(f"  → best f1(inj)={search.best_score_:.4f}", flush=True)
    return search


def tune_lgbm(X_train_df, y_train, groups):
    print(f"\n=== Tuning LGBM ({N_ITER['lgbm']} trials, LOGO {len(set(groups))}-fold) ===", flush=True)
    pre = make_column_transformer(with_scaler=False)
    pipe = Pipeline([("pre", pre),
                     ("clf", LGBMClassifier(
                         objective="binary", random_state=RANDOM_SEED,
                         n_jobs=1, verbose=-1))])
    param_dist = {
        "clf__n_estimators": randint(200, 1500),
        "clf__learning_rate": loguniform(1e-2, 0.2),
        "clf__num_leaves": randint(15, 200),
        "clf__max_depth": randint(4, 14),
        "clf__min_child_samples": randint(5, 100),
        "clf__reg_alpha": loguniform(1e-3, 10.0),
        "clf__reg_lambda": loguniform(1e-3, 10.0),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__subsample_freq": [1],
    }
    search = ProgressiveRandomizedSearchCV(
        pipe, param_dist, n_iter=N_ITER["lgbm"],
        cv=LeaveOneGroupOut(), scoring=SCORER,
        random_state=RANDOM_SEED, n_jobs=N_JOBS, refit=True, verbose=0,
    )
    search.fit(X_train_df, y_train, groups=groups)
    print(f"  → best f1(inj)={search.best_score_:.4f}", flush=True)
    return search


def tune_catboost(X_train_df, y_train, groups):
    print(f"\n=== Tuning CatBoost ({N_ITER['catboost']} trials, LOGO {len(set(groups))}-fold) ===", flush=True)
    pre = make_column_transformer(with_scaler=False)
    pipe = Pipeline([("pre", pre),
                     ("clf", CatBoostClassifier(
                         loss_function="Logloss", random_seed=RANDOM_SEED,
                         verbose=0, thread_count=1,
                         allow_writing_files=False))])
    param_dist = {
        "clf__iterations": randint(500, 2500),
        "clf__learning_rate": loguniform(1e-2, 0.2),
        "clf__depth": randint(4, 10),
        "clf__l2_leaf_reg": loguniform(1e-2, 30.0),
        "clf__random_strength": loguniform(1e-3, 10.0),
        "clf__bagging_temperature": uniform(0.0, 1.0),
    }
    search = ProgressiveRandomizedSearchCV(
        pipe, param_dist, n_iter=N_ITER["catboost"],
        cv=LeaveOneGroupOut(), scoring=SCORER,
        random_state=RANDOM_SEED, n_jobs=1, refit=True, verbose=0,
    )
    search.fit(X_train_df, y_train, groups=groups)
    print(f"  → best f1(inj)={search.best_score_:.4f}", flush=True)
    return search


def tune_bart(Xt_train, y_train, Xt_val, y_val):
    import pymc as pm
    import pymc_bart as pmb

    total = len(BART_GRID)
    print(f"\n=== Tuning BART (grid={total} configs, "
          f"search_on={BART_SEARCH_SAMPLES} samples) ===", flush=True)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(y_train) > BART_SEARCH_SAMPLES:
        idx = rng.choice(len(y_train), BART_SEARCH_SAMPLES, replace=False)
        X_tr_sub, y_tr_sub = Xt_train[idx], y_train[idx]
    else:
        X_tr_sub, y_tr_sub = Xt_train, y_train

    results = []
    best_score = -np.inf
    best_cfg = None
    start = time.time()
    for i, (m_trees, alpha, beta) in enumerate(BART_GRID):
        print(f"  [{i+1}/{total}] m={m_trees} alpha={alpha} beta={beta}",
              end="", flush=True)
        t0 = time.time()
        try:
            with pm.Model() as model:
                X_shared = pm.Data("X", X_tr_sub.astype(float))
                mu = pmb.BART("mu", X=X_shared, Y=y_tr_sub, m=m_trees,
                               alpha=alpha, beta=beta)
                p = pm.Deterministic("p", pm.math.sigmoid(mu))
                pm.Bernoulli("y", p=p, observed=y_tr_sub)
                idata = pm.sample(
                    draws=BART_SEARCH_DRAWS, tune=100,
                    chains=BART_SEARCH_CHAINS, cores=1,
                    random_seed=RANDOM_SEED, progressbar=False,
                    compute_convergence_checks=False,
                )
            with model:
                pm.set_data({"X": Xt_val.astype(float)})
                pp = pm.sample_posterior_predictive(
                    idata, var_names=["p"],
                    random_seed=RANDOM_SEED, progressbar=False,
                )
            p_val = (pp.posterior_predictive["p"]
                       .mean(dim=("chain", "draw")).values)
            f1 = f1_score(y_val, (p_val >= 0.5).astype(int),
                            pos_label=1, zero_division=0)
            elapsed = time.time() - t0
            remaining = total - (i + 1)
            eta = (time.time() - start) / (i + 1) * remaining
            print(f" → f1(inj)={f1:.4f} in {elapsed:.0f}s "
                  f"(ETA {eta:.0f}s)", flush=True)
            results.append({"m": m_trees, "alpha": alpha, "beta": beta,
                            "f1_val": round(float(f1), 4),
                            "seconds": round(elapsed, 1)})
            if f1 > best_score:
                best_score = f1
                best_cfg = (m_trees, alpha, beta)
        except Exception as exc:
            print(f" → error: {exc}", flush=True)
            results.append({"m": m_trees, "alpha": alpha, "beta": beta,
                            "error": str(exc)})
    print(f"  → best BART: {best_cfg} (val f1={best_score:.4f})", flush=True)
    return best_cfg, results


# ===========================================================================
# Helpers
# ===========================================================================
def _clean_params(params):
    out = {}
    for k, v in params.items():
        kk = k.replace("clf__", "").replace("pre__", "")
        if hasattr(v, "item"):
            v = v.item()
        out[kk] = v
    return out


def _md(df_):
    h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
    s = "| " + " | ".join("---" for _ in df_.columns) + " |"
    rs = ["| " + " | ".join(str(v) for v in r) + " |"
          for r in df_.itertuples(index=False)]
    return "\n".join([h, s] + rs)


def _cm_to_md(cm):
    df = pd.DataFrame(cm["matrix"], index=cm["labels"],
                       columns=cm["labels"])
    header = "| | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + str(idx) + " | " +
            " | ".join(str(v) for v in row) + " |"
            for idx, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def _eval_and_log(key, search_or_none, X_val_df, y_val, X_test_df, y_test,
                  pipeline=None):
    """Helper para evaluar cualquier modelo pipeline-tuneado."""
    if search_or_none is not None:
        p_val = search_or_none.predict_proba(X_val_df)[:, 1]
        p_test = search_or_none.predict_proba(X_test_df)[:, 1]
    else:
        p_val = pipeline["val"]
        p_test = pipeline["test"]

    thr = tune_threshold(y_val, p_val)
    return {
        "threshold_tuned": round(thr, 3),
        "val_at_0.5": binary_metrics(y_val, p_val, 0.5),
        "test_at_0.5": binary_metrics(y_test, p_test, 0.5),
        "val_at_tuned": binary_metrics(y_val, p_val, thr),
        "test_at_tuned": binary_metrics(y_test, p_test, thr),
        "lift_recall_test": lift_recall_at_k(y_test, p_test),
        "bootstrap_test": bootstrap_ci(y_test, p_test, thr),
    }, p_val, p_test


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading accidents + network...", flush=True)
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
    df["y_bin"] = severity_to_binary(df["severity_code"].astype(int))
    print(f"  {len(df):,} accidents.")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        counts = sub["y_bin"].value_counts().sort_index().to_dict()
        rate = counts.get(1, 0) / len(sub) if len(sub) else 0
        print(f"  {split}: {len(sub):>6,} | injured_rate={rate:.3f} | {counts}",
              flush=True)

    y_all = df["y_bin"].astype(int).to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]
    groups_train = df.loc[train_mask, "year"].astype(int).to_numpy()
    print(f"  train years: {sorted(set(groups_train))}\n", flush=True)

    print("Building features (M2 + red + TE injured)...", flush=True)
    X = build_features(df)
    add_te_columns(X, df, train_mask, y_train)
    X_train_df = X.loc[train_mask]
    X_val_df = X.loc[val_mask]
    X_test_df = X.loc[test_mask]
    print(f"  features: {X.shape[1]} cols\n", flush=True)

    results = {}
    t_global = time.time()

    # -------- LR --------
    search_lr = tune_lr(X_train_df, y_train, groups_train)
    res_lr, p_val_lr, p_test_lr = _eval_and_log(
        "lr", search_lr, X_val_df, y_val, X_test_df, y_test)
    res_lr["best_params"] = _clean_params(search_lr.best_params_)
    res_lr["cv_f1_injured"] = round(float(search_lr.best_score_), 4)
    results["lr"] = res_lr
    m = res_lr["test_at_tuned"]
    print(f"  LR test @thr={res_lr['threshold_tuned']}: kappa={m['kappa']:.3f} "
          f"F1(inj)={m['f1_injured']:.3f} PR-AUC={m['pr_auc_injured']:.4f} "
          f"rec={m['recall_injured']:.3f}\n", flush=True)

    # -------- LGBM --------
    search_lgbm = tune_lgbm(X_train_df, y_train, groups_train)
    res_lgbm, _, _ = _eval_and_log(
        "lgbm", search_lgbm, X_val_df, y_val, X_test_df, y_test)
    res_lgbm["best_params"] = _clean_params(search_lgbm.best_params_)
    res_lgbm["cv_f1_injured"] = round(float(search_lgbm.best_score_), 4)
    results["lgbm"] = res_lgbm
    m = res_lgbm["test_at_tuned"]
    print(f"  LGBM test @thr={res_lgbm['threshold_tuned']}: kappa={m['kappa']:.3f} "
          f"F1(inj)={m['f1_injured']:.3f} PR-AUC={m['pr_auc_injured']:.4f} "
          f"rec={m['recall_injured']:.3f}\n", flush=True)

    # -------- CatBoost --------
    search_cb = tune_catboost(X_train_df, y_train, groups_train)
    res_cb, _, _ = _eval_and_log(
        "catboost", search_cb, X_val_df, y_val, X_test_df, y_test)
    res_cb["best_params"] = _clean_params(search_cb.best_params_)
    res_cb["cv_f1_injured"] = round(float(search_cb.best_score_), 4)
    results["catboost"] = res_cb
    m = res_cb["test_at_tuned"]
    print(f"  CatBoost test @thr={res_cb['threshold_tuned']}: kappa={m['kappa']:.3f} "
          f"F1(inj)={m['f1_injured']:.3f} PR-AUC={m['pr_auc_injured']:.4f} "
          f"rec={m['recall_injured']:.3f}\n", flush=True)

    # -------- BART (búsqueda manual + refit) --------
    bart_search_log = []
    if INCLUDE_BART:
        # Para la búsqueda usamos el preprocesado sin scaler (árboles)
        pre_tree = make_column_transformer(with_scaler=False)
        Xt_tree_train = pre_tree.fit_transform(X_train_df)
        Xt_tree_val = pre_tree.transform(X_val_df)
        Xt_tree_test = pre_tree.transform(X_test_df)

        best_cfg, bart_search_log = tune_bart(
            Xt_tree_train, y_train, Xt_tree_val, y_val,
        )

        if best_cfg is not None:
            import pymc as pm
            import pymc_bart as pmb
            m_trees, alpha, beta = best_cfg
            print(f"\n=== Refit BART (m={m_trees}) en train completo "
                  f"({len(y_train)} samples) ===", flush=True)
            t0 = time.time()
            with pm.Model() as model:
                X_shared = pm.Data("X", Xt_tree_train.astype(float))
                mu = pmb.BART("mu", X=X_shared, Y=y_train, m=m_trees,
                               alpha=alpha, beta=beta)
                p = pm.Deterministic("p", pm.math.sigmoid(mu))
                pm.Bernoulli("y", p=p, observed=y_train)
                idata = pm.sample(
                    draws=BART_REFIT_DRAWS, tune=300,
                    chains=BART_REFIT_CHAINS, cores=BART_REFIT_CORES,
                    random_seed=RANDOM_SEED, progressbar=False,
                    compute_convergence_checks=False,
                )
            print(f"  sampling {time.time()-t0:.0f}s", flush=True)

            with model:
                pm.set_data({"X": Xt_tree_val.astype(float)})
                pp_val = pm.sample_posterior_predictive(
                    idata, var_names=["p"], random_seed=RANDOM_SEED,
                    progressbar=False,
                )
            p_val_b = (pp_val.posterior_predictive["p"]
                         .mean(dim=("chain", "draw")).values)

            with model:
                pm.set_data({"X": Xt_tree_test.astype(float)})
                pp_test = pm.sample_posterior_predictive(
                    idata, var_names=["p"], random_seed=RANDOM_SEED,
                    progressbar=False,
                )
            p_test_b = (pp_test.posterior_predictive["p"]
                          .mean(dim=("chain", "draw")).values)

            thr_b = tune_threshold(y_val, p_val_b)
            results["bart"] = {
                "best_params": {"m": m_trees, "alpha": alpha, "beta": beta},
                "search_log": bart_search_log,
                "threshold_tuned": round(thr_b, 3),
                "val_at_0.5": binary_metrics(y_val, p_val_b, 0.5),
                "test_at_0.5": binary_metrics(y_test, p_test_b, 0.5),
                "val_at_tuned": binary_metrics(y_val, p_val_b, thr_b),
                "test_at_tuned": binary_metrics(y_test, p_test_b, thr_b),
                "lift_recall_test": lift_recall_at_k(y_test, p_test_b),
                "bootstrap_test": bootstrap_ci(y_test, p_test_b, thr_b),
            }
            m = results["bart"]["test_at_tuned"]
            print(f"  BART test @thr={thr_b:.3f}: kappa={m['kappa']:.3f} "
                  f"F1(inj)={m['f1_injured']:.3f} "
                  f"PR-AUC={m['pr_auc_injured']:.4f}\n", flush=True)

    print(f"Tiempo total: {time.time()-t_global:.0f}s\n", flush=True)

    save_json(results, OUT_TABLES / "28_opcion_c.json")

    # -------- Resumen --------
    rows = []
    for key, r in results.items():
        m = r["test_at_tuned"]
        rows.append({
            "model": key,
            "cv_f1": r.get("cv_f1_injured", "N/A"),
            "threshold": r["threshold_tuned"],
            "kappa": round(m["kappa"], 4),
            "f1_injured": round(m["f1_injured"], 4),
            "precision": round(m["precision_injured"], 4),
            "recall": round(m["recall_injured"], 4),
            "pr_auc": round(m["pr_auc_injured"], 4),
            "roc_auc": round(m["roc_auc_injured"], 4),
        })
    comp = pd.DataFrame(rows)

    best_key = max(results.keys(),
                    key=lambda k: results[k]["test_at_tuned"]["pr_auc_injured"])
    lift_df = pd.DataFrame([
        {"k": k_label,
         "count": vals["k"],
         "injury_rate": vals["injury_rate"],
         "lift": vals["lift"],
         "recall": vals["recall"]}
        for k_label, vals in results[best_key]["lift_recall_test"]["by_k"].items()
    ])

    sections = {
        "Comparación (test=2025, threshold tuned en val)": _md(comp),
        "Lift@k — best model (" + best_key + ")": _md(lift_df),
        "Bootstrap CI — best model": dumps_jsonable(
            results[best_key]["bootstrap_test"], indent=2),
        "Confusion matrix — best (test, threshold tuned)": _cm_to_md(
            results[best_key]["test_at_tuned"]["confusion"]),
        "Confusion matrix — best (test, threshold 0.5)": _cm_to_md(
            results[best_key]["test_at_0.5"]["confusion"]),
        "Best params por modelo": dumps_jsonable(
            {k: r.get("best_params") for k, r in results.items()}, indent=2),
        "BART search log": dumps_jsonable(bart_search_log, indent=2),
        "Notes": (
            "Búsqueda de hiperparámetros:\n"
            "- LR/LGBM/CatBoost: RandomizedSearchCV + LeaveOneGroupOut(año) "
            "sobre train. Log trial-a-trial con ETA.\n"
            "- BART: grid manual (m, alpha, beta) sobre subconjunto de train "
            "(10000 samples), refit en train completo con best.\n"
            "Target: 0=no_injury, 1=any_injury (minor ∪ serious ∪ fatal).\n"
            "Threshold tuned en val (max F1(injured)).\n"
            "Preprocesado: imputer + OHE (árboles) / + scaler (LR).\n"
            "TE fit-on-train sobre P(injured)."
        ),
    }
    save_markdown_report(sections, Path("reports/28_opcion_c.md"))
    print(f"Best by PR-AUC(injured): {best_key}")
    print("Done. See reports/28_opcion_c.md.")


if __name__ == "__main__":
    main()