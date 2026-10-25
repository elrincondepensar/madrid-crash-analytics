"""
Script 32 — Phase 6 iter 2 (v3): temporal persistence of HDBSCAN clusters.

Changes vs v1 (same script number, same outputs):
- Assignment rule: dist < min(p90_radius, RADIUS_CAP_M) with factor 1.0
  (was: dist < 1.5 * max_radius). The previous rule was tautologically
  satisfied: share_persisted ≈ 1.0 in both splits.
- New stability metrics independent of the assignment threshold:
  - Spearman rho between n_train and n_future (all clusters and top-100).
  - Top-k persistence: share_ratio and rank overlap for k in {10, 20, 30}.
- Generosity gap diagnostic: coverage_new - train_coverage.
- Acceptance criteria updated: share_persisted loses its upper bound
  (moved to WARN flag), coverage_new gains an upper bound (0.80), new
  hard criterion on spearman_top100 >= 0.60.

Design notes
------------
- Coordinates: UTM 30N (EPSG:25830). Distances in meters.
- Cluster radius effective: min(p90_radius, 500 m). Purpose: exclude
  outliers that stretch elongated clusters into multi-km envelopes.
- Assignment: nearest train centroid if dist < radius_eff (factor 1.0).
- Persistence definition (unchanged): a train cluster is "persisted" in
  a future split if at least MIN_PERSISTED_ACCIDENTS of that split's
  accidents are assigned to it.
- Injured variant: fit and assignment restricted to injured accidents.

Outputs (overwritten):
    data/processed/hdbscan_persistence_all.parquet
    data/processed/hdbscan_persistence_injured.parquet
    reports/tables/32_hdbscan_persistence.json
    reports/32_persistence.md
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_TABLES = REPORTS / "tables"

ACCIDENTS_PATH = DATA_PROCESSED / "accidents_clean.parquet"
SPLITS_PATH = DATA_PROCESSED / "splits.parquet"

OUT_PARQUET_ALL = DATA_PROCESSED / "hdbscan_persistence_all.parquet"
OUT_PARQUET_INJURED = DATA_PROCESSED / "hdbscan_persistence_injured.parquet"
OUT_JSON = REPORTS_TABLES / "32_hdbscan_persistence.json"
OUT_MD = REPORTS / "32_persistence.md"

PARAMS = {
    "all":     {"mcs": 50, "ms": 10},
    "injured": {"mcs": 30, "ms": 10},
}

ASSIGNMENT_FACTOR = 1.0
RADIUS_CAP_M = 500.0
MIN_PERSISTED_ACCIDENTS = 10
METRIC = "euclidean"
COORDS = ["coordenada_x_utm", "coordenada_y_utm"]

TOP_K_VALUES = (10, 20, 30)
TOP_N_FOR_SPEARMAN = 100

ACCEPTANCE = {
    "share_persisted_min": 0.30,
    "share_persisted_warn_at": 1.00,
    "coverage_new_min": 0.40,
    "coverage_new_max": 0.80,
    "share_persistent_in_both_min": 0.30,
    "spearman_top100_min": 0.60,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase6_iter2")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_inputs() -> pd.DataFrame:
    log.info("Loading accidents_clean.parquet and splits.parquet ...")
    if not ACCIDENTS_PATH.exists():
        raise FileNotFoundError(f"missing {ACCIDENTS_PATH}")
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"missing {SPLITS_PATH}")

    acc = pd.read_parquet(ACCIDENTS_PATH)
    spl = pd.read_parquet(SPLITS_PATH)

    required_acc = {"num_expediente", "year", "severity_code", *COORDS}
    missing_acc = required_acc - set(acc.columns)
    if missing_acc:
        raise RuntimeError(f"accidents_clean missing columns: {missing_acc}")
    if "split" not in spl.columns or "num_expediente" not in spl.columns:
        raise RuntimeError("splits.parquet must have 'num_expediente' and 'split'")

    df = acc.merge(
        spl[["num_expediente", "split"]],
        on="num_expediente",
        how="inner",
        validate="one_to_one",
    )
    log.info(f"  accidents_clean: {len(acc):,}  |  after merge: {len(df):,}")
    log.info(f"  split counts: {df['split'].value_counts().to_dict()}")

    n_null = int(df[COORDS].isna().any(axis=1).sum())
    if n_null:
        log.warning(f"  {n_null} rows with missing coordinates will be dropped")
    df = df.dropna(subset=COORDS).copy()

    log.info(f"  severity_code counts: "
             f"{df['severity_code'].value_counts().sort_index().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_train_hdbscan(df_train: pd.DataFrame, mcs: int, ms: int, variant: str):
    log.info(f"[{variant}] fitting HDBSCAN on train: n={len(df_train):,}, "
             f"mcs={mcs}, ms={ms}, metric={METRIC}")
    if len(df_train) == 0:
        raise RuntimeError(f"[{variant}] empty train split")

    X = df_train[COORDS].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric=METRIC,
        cluster_selection_epsilon=0.0,
        algorithm="best",
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(X)
    fit_s = float(time.perf_counter() - t0)

    n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
    noise_share = float((labels == -1).mean())
    log.info(f"[{variant}] fit done in {fit_s:.1f}s — "
             f"{n_clusters} clusters, noise_share={noise_share:.3f}")

    if n_clusters == 0:
        raise RuntimeError(
            f"[{variant}] HDBSCAN produced 0 clusters on train — "
            f"parameters incompatible with this split"
        )
    return labels, n_clusters, noise_share, fit_s


def compute_centroids_and_radii(labels: np.ndarray,
                                coords: np.ndarray,
                                n_clusters: int) -> pd.DataFrame:
    log.info(f"Computing centroids and radii for {n_clusters} clusters ...")
    rows = []
    for cid in range(n_clusters):
        mask = labels == cid
        pts = coords[mask]
        cx, cy = pts.mean(axis=0)
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        p90 = float(np.percentile(dists, 90))
        rows.append({
            "cluster_train": cid,
            "cx": float(cx),
            "cy": float(cy),
            "radius_m": float(dists.max()),
            "radius_p90_m": p90,
            "radius_eff_m": float(min(p90, RADIUS_CAP_M)),
            "n_train": int(mask.sum()),
            "median_dist_m": float(np.median(dists)),
        })
    centroids = (pd.DataFrame(rows)
                 .sort_values("cluster_train")
                 .reset_index(drop=True))
    log.info(f"  radius_m (max): median={centroids['radius_m'].median():.1f} m, "
             f"max={centroids['radius_m'].max():.1f} m")
    log.info(f"  radius_p90_m: median={centroids['radius_p90_m'].median():.1f} m, "
             f"p90={centroids['radius_p90_m'].quantile(0.9):.1f} m, "
             f"max={centroids['radius_p90_m'].max():.1f} m")
    log.info(f"  radius_eff_m: median={centroids['radius_eff_m'].median():.1f} m, "
             f"p90={centroids['radius_eff_m'].quantile(0.9):.1f} m, "
             f"max={centroids['radius_eff_m'].max():.1f} m "
             f"(capped at {RADIUS_CAP_M:.0f} m)")
    n_capped = int((centroids['radius_p90_m'] > RADIUS_CAP_M).sum())
    log.info(f"  clusters affected by cap: {n_capped}/{n_clusters}")
    return centroids


def compute_train_dists(labels: np.ndarray, coords: np.ndarray,
                        centroids: pd.DataFrame) -> np.ndarray:
    cxy = centroids.set_index("cluster_train")[["cx", "cy"]]
    out = np.full(len(labels), np.nan, dtype=np.float64)
    mask = labels >= 0
    if mask.any():
        cc = cxy.reindex(labels[mask]).to_numpy()
        out[mask] = np.hypot(coords[mask, 0] - cc[:, 0],
                             coords[mask, 1] - cc[:, 1])
    return out


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------

def assign_points(df_split: pd.DataFrame,
                  centroids: pd.DataFrame,
                  factor: float = ASSIGNMENT_FACTOR) -> pd.DataFrame:
    if len(df_split) == 0:
        out = df_split.copy()
        out["cluster_train"] = np.array([], dtype=np.int64)
        out["dist_to_centroid_m"] = np.array([], dtype=np.float64)
        return out

    xy = df_split[COORDS].to_numpy(dtype=np.float64)
    cxy = centroids[["cx", "cy"]].to_numpy(dtype=np.float64)
    radii = centroids["radius_eff_m"].to_numpy(dtype=np.float64)
    cids = centroids["cluster_train"].to_numpy(dtype=np.int64)

    dx = xy[:, None, 0] - cxy[None, :, 0]
    dy = xy[:, None, 1] - cxy[None, :, 1]
    dist = np.hypot(dx, dy)

    nearest_idx = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(len(xy)), nearest_idx]
    nearest_radius = radii[nearest_idx]
    nearest_cid = cids[nearest_idx]

    assigned = nearest_dist < factor * nearest_radius
    cluster_out = np.where(assigned, nearest_cid, -1).astype(np.int64)
    dist_out = np.where(assigned, nearest_dist, np.nan).astype(np.float64)

    out = df_split.copy()
    out["cluster_train"] = cluster_out
    out["dist_to_centroid_m"] = dist_out
    return out


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def _cluster_counts(df: pd.DataFrame) -> pd.Series:
    s = df.loc[df["cluster_train"] >= 0, "cluster_train"]
    return s.value_counts().astype(int)


def _spearman_safe(a: pd.Series, b: pd.Series) -> dict:
    if len(a) < 5:
        return {"rho": None, "p_value": None, "n_clusters": int(len(a))}
    try:
        rho, p = spearmanr(a.values, b.values)
        return {"rho": float(rho), "p_value": float(p),
                "n_clusters": int(len(a))}
    except Exception as e:  # noqa: BLE001
        return {"rho": None, "p_value": None, "n_clusters": int(len(a)),
                "error": str(e)}


def compute_stability_metrics(df_train: pd.DataFrame,
                              df_future: pd.DataFrame,
                              centroids: pd.DataFrame,
                              top_n_for_spearman: int = TOP_N_FOR_SPEARMAN,
                              top_k_values: tuple = TOP_K_VALUES) -> dict:
    """Stability of cluster sizes between train and a future split.

    Both metrics are independent of the assignment rule's sharpness:
    - Spearman on counts: does the ranking of cluster sizes persist?
    - Top-k share ratio: do the k largest train clusters retain future volume?
    """
    train_counts = _cluster_counts(df_train)
    future_counts = _cluster_counts(df_future)

    merged = pd.DataFrame({
        "cluster_train": train_counts.index,
        "n_train": train_counts.values,
    }).merge(
        pd.DataFrame({
            "cluster_train": future_counts.index,
            "n_future": future_counts.values,
        }),
        on="cluster_train",
        how="inner",
    )

    if len(merged) == 0:
        return {"spearman": {"all": None, "top_n_by_train": None},
                "top_k_persistence": {}}

    # Spearman: all clusters present in both
    sp_all = _spearman_safe(merged["n_train"], merged["n_future"])

    # Spearman: top-N by n_train (the operationally meaningful ones)
    top_n = min(top_n_for_spearman, len(merged))
    merged_top = merged.nlargest(top_n, "n_train").reset_index(drop=True)
    sp_top = _spearman_safe(merged_top["n_train"], merged_top["n_future"])

    # Top-k persistence
    train_total = int(train_counts.sum())
    future_total = int(future_counts.sum())
    top_k_out = {}
    for k in top_k_values:
        if len(train_counts) < k:
            continue
        top_k_ids = set(train_counts.nlargest(k).index.astype(int).tolist())
        top_k_future_ids = set(future_counts.nlargest(k).index.astype(int).tolist())

        share_train = int(train_counts.loc[list(top_k_ids)].sum()) / max(train_total, 1)
        share_future_in_k = int(future_counts.reindex(list(top_k_ids)).fillna(0).sum()) / max(future_total, 1)
        ratio = (share_future_in_k / share_train) if share_train > 0 else None

        overlap = len(top_k_ids & top_k_future_ids)
        top_k_out[f"k_{k}"] = {
            "n_train_in_top_k": int(train_counts.loc[list(top_k_ids)].sum()),
            "share_train_in_top_k": float(share_train),
            "n_future_in_top_k": int(future_counts.reindex(list(top_k_ids)).fillna(0).sum()),
            "share_future_in_top_k": float(share_future_in_k),
            "share_ratio": float(ratio) if ratio is not None else None,
            "top_k_overlap": int(overlap),
            "top_k_overlap_share": float(overlap / k),
            "top_k_ids_by_train": sorted(int(x) for x in top_k_ids),
        }

    return {
        "spearman": {"all": sp_all, "top_n_by_train": sp_top,
                     "top_n": int(top_n)},
        "top_k_persistence": top_k_out,
    }


# ---------------------------------------------------------------------------
# Persistence metrics
# ---------------------------------------------------------------------------

def _per_split_metrics(df: pd.DataFrame, min_persisted: int,
                       train_coverage: float) -> dict:
    assigned = df["cluster_train"] >= 0
    coverage = float(assigned.mean()) if len(df) else 0.0
    counts = df.loc[assigned, "cluster_train"].value_counts()
    persisted_ids = sorted(counts[counts >= min_persisted].index.astype(int).tolist())

    if assigned.any():
        med = float(df.loc[assigned, "dist_to_centroid_m"].median())
        p90 = float(df.loc[assigned, "dist_to_centroid_m"].quantile(0.9))
    else:
        med, p90 = None, None

    return {
        "n_accidents": int(len(df)),
        "coverage_new": coverage,
        "generosity_gap": float(coverage - train_coverage),
        "n_clusters_with_any": int((counts > 0).sum()),
        "n_clusters_persisted": len(persisted_ids),
        "persisted_ids": persisted_ids,
        "dist_median": med,
        "dist_p90": p90,
    }


# ---------------------------------------------------------------------------
# Orchestration per variant
# ---------------------------------------------------------------------------

def run_variant(df: pd.DataFrame, variant: str):
    params = PARAMS[variant]
    mcs, ms = params["mcs"], params["ms"]

    df_train = df[df["split"] == "train"].copy()
    if variant == "injured":
        df_train = df_train[df_train["severity_code"] >= 1].copy()
    log.info(f"[{variant}] train subset: n={len(df_train):,}")

    labels, n_clusters, noise_share, fit_s = fit_train_hdbscan(
        df_train, mcs, ms, variant)
    train_coverage = float(1.0 - noise_share)

    train_coords = df_train[COORDS].to_numpy(dtype=np.float64)
    centroids = compute_centroids_and_radii(labels, train_coords, n_clusters)

    df_train["cluster_train"] = labels.astype(np.int64)
    df_train["dist_to_centroid_m"] = compute_train_dists(labels, train_coords, centroids)

    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()
    if variant == "injured":
        df_val = df_val[df_val["severity_code"] >= 1].copy()
        df_test = df_test[df_test["severity_code"] >= 1].copy()
    log.info(f"[{variant}] val subset: n={len(df_val):,}   "
             f"test subset: n={len(df_test):,}")

    df_val = assign_points(df_val, centroids)
    df_test = assign_points(df_test, centroids)

    # Stability metrics on assigned points
    stability_val = compute_stability_metrics(df_train, df_val, centroids)
    stability_test = compute_stability_metrics(df_train, df_test, centroids)

    # Assemble parquet
    out_cols = ["num_expediente", "year", "split", "severity_code",
                "coordenada_x_utm", "coordenada_y_utm",
                "cluster_train", "dist_to_centroid_m"]
    df_out = pd.concat(
        [df_train[out_cols], df_val[out_cols], df_test[out_cols]],
        ignore_index=True,
    ).rename(columns={"coordenada_x_utm": "x", "coordenada_y_utm": "y"})
    df_out = df_out[["num_expediente", "year", "split", "severity_code",
                     "x", "y", "cluster_train", "dist_to_centroid_m"]]

    # Per-split persistence metrics
    val_m = _per_split_metrics(df_val, MIN_PERSISTED_ACCIDENTS, train_coverage)
    test_m = _per_split_metrics(df_test, MIN_PERSISTED_ACCIDENTS, train_coverage)

    val_out = {k: v for k, v in val_m.items() if k != "persisted_ids"}
    test_out = {k: v for k, v in test_m.items() if k != "persisted_ids"}
    val_out["stability"] = stability_val
    test_out["stability"] = stability_test

    both = sorted(set(val_m["persisted_ids"]) & set(test_m["persisted_ids"]))

    metrics = {
        "variant": variant,
        "params": params,
        "metric": METRIC,
        "assignment_rule": {
            "radius_definition": f"min(p90_radius, {RADIUS_CAP_M:.0f} m)",
            "radius_cap_m": RADIUS_CAP_M,
            "factor": ASSIGNMENT_FACTOR,
        },
        "n_train": int(len(df_train)),
        "n_clusters_train": int(n_clusters),
        "train_noise_share": float(noise_share),
        "train_coverage": train_coverage,
        "fit_seconds": float(fit_s),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "val": val_out,
        "test": test_out,
        "n_clusters_persisted_val": int(val_m["n_clusters_persisted"]),
        "n_clusters_persisted_test": int(test_m["n_clusters_persisted"]),
        "n_clusters_persistent_in_both": int(len(both)),
        "persistent_ids_both": both,
        "share_persisted_val":  val_m["n_clusters_persisted"] / n_clusters,
        "share_persisted_test": test_m["n_clusters_persisted"] / n_clusters,
        "share_persistent_in_both": len(both) / n_clusters,
    }

    metrics["acceptance"] = acceptance_check(metrics)
    metrics["centroids"] = centroids.to_dict(orient="records")
    return metrics, df_out


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def acceptance_check(m: dict) -> dict:
    checks = {}

    # share_persisted: lower bound only (>= 0.30). Value == 1.00 → WARN.
    for split in ("val", "test"):
        v = m[f"share_persisted_{split}"]
        lo = ACCEPTANCE["share_persisted_min"]
        warn_val = ACCEPTANCE["share_persisted_warn_at"]
        checks[f"share_persisted_{split}"] = {
            "value": float(v),
            "lower": lo,
            "ok": bool(v >= lo),
            "warn_laxity": bool(abs(v - warn_val) < 1e-9),
        }

    # coverage_new: [0.40, 0.80]
    lo = ACCEPTANCE["coverage_new_min"]
    hi = ACCEPTANCE["coverage_new_max"]
    for split in ("val", "test"):
        v = m[split]["coverage_new"]
        checks[f"coverage_new_{split}"] = {
            "value": float(v), "bounds": [lo, hi],
            "ok": bool(lo <= v <= hi),
        }

    # share_persistent_in_both >= 0.30
    lo = ACCEPTANCE["share_persistent_in_both_min"]
    v = m["share_persistent_in_both"]
    checks["share_persistent_in_both"] = {
        "value": float(v), "lower": lo, "ok": bool(v >= lo),
    }

    # spearman_top100 >= 0.60 (per split)
    lo = ACCEPTANCE["spearman_top100_min"]
    for split in ("val", "test"):
        sp = m[split]["stability"]["spearman"]["top_n_by_train"]
        rho = sp.get("rho") if sp else None
        checks[f"spearman_top100_{split}"] = {
            "value": float(rho) if rho is not None else None,
            "lower": lo,
            "ok": bool(rho is not None and rho >= lo),
        }

    checks["_all_ok"] = all(
        c["ok"] for k, c in checks.items() if not k.startswith("_"))
    checks["_any_laxity_warn"] = any(
        c.get("warn_laxity", False)
        for k, c in checks.items() if not k.startswith("_"))
    return checks


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(v, digits=3, suffix=""):
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}{suffix}"


def render_markdown(results: dict) -> str:
    lines = ["# Phase 6 · Iter 2 (v3) — Temporal persistence of HDBSCAN clusters\n"]
    lines.append(
        f"_Assignment rule: nearest train centroid if "
        f"`dist < {ASSIGNMENT_FACTOR} × min(p90_radius, {RADIUS_CAP_M:.0f} m)`. "
        f"Persistence threshold: `>= {MIN_PERSISTED_ACCIDENTS}` accidents "
        f"from the future split._\n"
    )
    lines.append(
        "_No refit on val/test. Coordinates UTM 30N (EPSG:25830), distances in meters._\n"
    )
    lines.append(
        "_Iter-2 v1 used `1.5 × max_radius`; that rule was tautologically satisfied. "
        "v3 replaces `max` by `p90` with a hard cap to remove outlier-driven envelopes, "
        "and adds two rule-independent stability metrics (Spearman ρ and top-k share ratio)._\n"
    )

    for variant in ("all", "injured"):
        m = results[variant]
        c = m["acceptance"]
        p = m["params"]
        lines.append(f"## Variant `{variant}` — mcs={p['mcs']}, ms={p['ms']}\n")

        lines.append(
            f"- **Train**: {m['n_train']:,} accidents → "
            f"**{m['n_clusters_train']} clusters**, "
            f"noise_share {m['train_noise_share']:.3f} "
            f"(train coverage in-cluster {m['train_coverage']:.3f}), "
            f"fit {m['fit_seconds']:.1f}s"
        )
        lines.append(
            f"- **radius_eff_m**: median {m['centroids'][0].get('radius_eff_m', 0):.0f} m "
            f"… (per-cluster, capped at {RADIUS_CAP_M:.0f} m)"
        )

        for split_label, split_key, n_key in (
            ("Val (2023-24)", "val", "n_val"),
            ("Test (2025)",   "test", "n_test"),
        ):
            s = m[split_key]
            lines.append(
                f"- **{split_label}**: n={m[n_key]:,} → "
                f"coverage **{s['coverage_new']:.3f}** "
                f"(generosity gap {s['generosity_gap']:+.3f}), "
                f"persisted **{s['n_clusters_persisted']}** "
                f"({s['n_clusters_persisted']/m['n_clusters_train']:.3f})"
            )
            if s["dist_median"] is not None:
                lines.append(
                    f"  - dist to assigned centroid: median "
                    f"{s['dist_median']:.1f} m, p90 {s['dist_p90']:.1f} m"
                )

            stab = s["stability"]
            sp_all = stab["spearman"]["all"]
            sp_top = stab["spearman"]["top_n_by_train"]
            lines.append(
                f"  - **Spearman ρ** "
                f"(all clusters, n={sp_all['n_clusters'] if sp_all else 0}): "
                f"{_fmt(sp_all['rho'] if sp_all else None)}"
                + (f" [p={sp_all['p_value']:.2e}]" if sp_all and sp_all.get('p_value') is not None else "")
                + f" | "
                f"(top-{stab['spearman']['top_n']} by train): "
                f"{_fmt(sp_top['rho'] if sp_top else None)}"
            )

            tk = stab["top_k_persistence"]
            for kk, vv in tk.items():
                lines.append(
                    f"  - **Top-{kk[2:]}**: "
                    f"share_train={vv['share_train_in_top_k']:.3f}, "
                    f"share_future={vv['share_future_in_top_k']:.3f}, "
                    f"ratio={_fmt(vv['share_ratio'])}, "
                    f"overlap={vv['top_k_overlap']}/{kk[2:]} "
                    f"({vv['top_k_overlap_share']:.2f})"
                )

        lines.append(
            f"- **Persisted in BOTH** val and test: "
            f"**{m['n_clusters_persistent_in_both']}** "
            f"({m['share_persistent_in_both']:.3f})\n"
        )

        lines.append("### Acceptance criteria\n")
        lines.append("| criterion | value | bounds | status |")
        lines.append("| --- | ---: | :---: | :---: |")
        for k, v in c.items():
            if k.startswith("_"):
                continue
            val = v.get("value")
            if "bounds" in v:
                bounds = f"[{v['bounds'][0]:.2f}, {v['bounds'][1]:.2f}]"
            elif "lower" in v:
                bounds = f">= {v['lower']:.2f}"
            else:
                bounds = "n/a"
            status = "PASS" if v["ok"] else "FAIL"
            if v.get("warn_laxity"):
                status += " ⚠"
            lines.append(f"| `{k}` | {_fmt(val)} | {bounds} | {status} |")
        lines.append(
            f"\n**Overall**: "
            f"{'ALL PASS' if c['_all_ok'] else 'SOME FAIL'}"
            + ("  ·  ⚠ laxity warning" if c['_any_laxity_warn'] else "")
            + "\n"
        )
        lines.append("---\n")

    lines.append("## Notes\n")
    lines.append(
        "- Cluster radius effective = **min(p90_radius, "
        f"{RADIUS_CAP_M:.0f} m)**. Purpose: remove outlier-driven envelopes "
        "that stretch elongated clusters to multi-km radii."
    )
    lines.append(
        "- `generosity_gap = coverage_new - train_coverage`. Values close to 0 "
        "indicate the assignment rule is not more generous than HDBSCAN itself. "
        "Iter-2 v1 (max_radius × 1.5) gave +0.25 pp — tautological."
    )
    lines.append(
        "- **Spearman ρ** between `n_train` and `n_future` (cluster sizes) is "
        "invariant to the assignment threshold. Reported for all clusters "
        "present in both splits and for the top-100 by train."
    )
    lines.append(
        "- **Top-k share ratio** = `share_future(top-k-by-train) / "
        "share_train(top-k-by-train)`. Values near 1.0 indicate the k largest "
        "train clusters retain their proportional future volume. "
        "`overlap` counts how many of the top-k by train remain top-k by future."
    )
    lines.append(
        "- For the `injured` variant, val/test coverage is measured over "
        "**injured-only** future accidents, matching the fit population."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_inputs()

    results = {}
    for variant in ("all", "injured"):
        log.info(f"=== Variant: {variant} ===")
        metrics, df_out = run_variant(df, variant)

        out_path = OUT_PARQUET_ALL if variant == "all" else OUT_PARQUET_INJURED
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path, index=False)
        log.info(f"  saved {out_path.relative_to(ROOT)} ({len(df_out):,} rows)")

        results[variant] = metrics

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(results), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    log.info("=" * 60)
    for v in ("all", "injured"):
        m = results[v]
        c = m["acceptance"]
        log.info(
            f"[{v}] val persisted={m['n_clusters_persisted_val']} "
            f"({m['share_persisted_val']:.3f})  "
            f"test={m['n_clusters_persisted_test']} "
            f"({m['share_persisted_test']:.3f})  "
            f"both={m['n_clusters_persistent_in_both']} "
            f"({m['share_persistent_in_both']:.3f})  "
            f"| cov val={m['val']['coverage_new']:.3f} "
            f"test={m['test']['coverage_new']:.3f}  "
            f"| ρ_top100 val={_fmt(c['spearman_top100_val']['value'])} "
            f"test={_fmt(c['spearman_top100_test']['value'])}  "
            f"| {'PASS' if c['_all_ok'] else 'FAIL'}"
            + ("  ⚠" if c['_any_laxity_warn'] else "")
        )


if __name__ == "__main__":
    main()