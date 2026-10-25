"""
Script 33 — Noise-ceiling validation for HDBSCAN stability (Phase 6 closure).

Purpose
-------
Script 32 measured Spearman rho between train-cluster sizes and sizes
obtained when assigning val/test accidents to train centroids using the
rule `dist < min(p90_radius, 500 m)`. Values (0.51-0.60 for top-100,
0.73-0.82 for all clusters) fell below an arbitrary threshold of 0.60
chosen a priori. This script asks: is the observed rho the *methodological
ceiling* imposed by sample size + Poisson noise, or evidence of real
temporal drift?

Approach
--------
Fit-assign controls. For each variant (all, injured):
    1. Sanity: fit on full-train, assign full-train. rho should be ~1.0.
    2. Reference: fit on full-train, assign test. Should match script 32.
    3. Reference: fit on full-train, assign val.
    4. Internal 50/50: random split of train into A + B. Fit A -> assign B.
       Same pipeline, no temporal separation. Compare to (2).
    5. Internal 75/25: same, mimicking the train/val ratio.
    6. Internal test 50/50: split test, fit one half, assign the other.

Decision rule
-------------
If rho(internal 50/50) ~ rho(train -> test), the temporal dimension adds
nothing beyond sampling noise: the observed rho is at the ceiling.
If rho(internal 50/50) >> rho(train -> test), temporal drift is real.

Outputs:
    reports/tables/33_noise_ceiling.json
    reports/33_noise_ceiling.md
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
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
SCRIPT32_JSON = REPORTS_TABLES / "32_hdbscan_persistence.json"

OUT_JSON = REPORTS_TABLES / "33_noise_ceiling.json"
OUT_MD = REPORTS / "33_noise_ceiling.md"

PARAMS = {
    "all":     {"mcs": 50, "ms": 10},
    "injured": {"mcs": 30, "ms": 10},
}
RADIUS_CAP_M = 500.0
ASSIGN_FACTOR = 1.0
SEED = 42
METRIC = "euclidean"
COORDS = ["coordenada_x_utm", "coordenada_y_utm"]

log = logging.getLogger("phase6_iter3")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_inputs() -> pd.DataFrame:
    log.info("Loading accidents_clean.parquet and splits.parquet ...")
    acc = pd.read_parquet(ACCIDENTS_PATH)
    spl = pd.read_parquet(SPLITS_PATH)
    df = acc.merge(
        spl[["num_expediente", "split"]],
        on="num_expediente", how="inner", validate="one_to_one",
    )
    df = df.dropna(subset=COORDS).copy()
    log.info(f"  rows: {len(df):,}  |  splits: {df['split'].value_counts().to_dict()}")
    log.info(f"  severity: {df['severity_code'].value_counts().sort_index().to_dict()}")
    return df


def load_script32_reference() -> dict:
    if not SCRIPT32_JSON.exists():
        log.warning("script 32 reference JSON not found; skipping cross-reference")
        return {}
    import json
    data = json.loads(SCRIPT32_JSON.read_text("utf-8"))
    ref = {}
    for v in ("all", "injured"):
        if v not in data:
            continue
        m = data[v]
        ref[v] = {
            "rho_train_vs_val_all":       m["val"]["stability"]["spearman"]["all"]["rho"],
            "rho_train_vs_test_all":      m["test"]["stability"]["spearman"]["all"]["rho"],
            "rho_train_vs_val_top100":    m["val"]["stability"]["spearman"]["top_n_by_train"]["rho"],
            "rho_train_vs_test_top100":   m["test"]["stability"]["spearman"]["top_n_by_train"]["rho"],
            "coverage_val":  m["val"]["coverage_new"],
            "coverage_test": m["test"]["coverage_new"],
        }
    return ref


# ---------------------------------------------------------------------------
# Fit + assign primitives
# ---------------------------------------------------------------------------

def fit_hdbscan(X: np.ndarray, mcs: int, ms: int) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric=METRIC,
        cluster_selection_epsilon=0.0,
        algorithm="best",
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(X)


def centroids_and_radii(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    n = int(labels.max() + 1) if labels.max() >= 0 else 0
    rows = []
    for cid in range(n):
        mask = labels == cid
        pts = X[mask]
        if len(pts) == 0:
            continue
        cx, cy = pts.mean(axis=0)
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        p90 = float(np.percentile(dists, 90))
        rows.append({
            "cluster_id":   cid,
            "cx":           float(cx),
            "cy":           float(cy),
            "radius_p90_m": p90,
            "radius_eff_m": float(min(p90, RADIUS_CAP_M)),
            "size":         int(len(pts)),
        })
    return pd.DataFrame(rows)


def assign_to_centroids(df_assign: pd.DataFrame,
                        cent: pd.DataFrame,
                        factor: float = ASSIGN_FACTOR) -> np.ndarray:
    xy = df_assign[COORDS].to_numpy(dtype=np.float64)
    cxy = cent[["cx", "cy"]].to_numpy(dtype=np.float64)
    radii = cent["radius_eff_m"].to_numpy(dtype=np.float64)
    cids = cent["cluster_id"].to_numpy(dtype=np.int64)

    dx = xy[:, None, 0] - cxy[None, :, 0]
    dy = xy[:, None, 1] - cxy[None, :, 1]
    D = np.hypot(dx, dy)

    nearest_idx = D.argmin(axis=1)
    nearest_dist = D[np.arange(len(xy)), nearest_idx]
    nearest_radius = radii[nearest_idx]
    nearest_cid = cids[nearest_idx]

    assigned = nearest_dist < factor * nearest_radius
    return np.where(assigned, nearest_cid, -1)


def assign_and_compare(df_fit: pd.DataFrame,
                       df_assign: pd.DataFrame,
                       mcs: int, ms: int,
                       label: str) -> dict:
    """Fit on df_fit, assign df_assign points, return rho between sizes."""
    t0 = time.perf_counter()
    X_fit = df_fit[COORDS].to_numpy(dtype=np.float64)
    labels_fit = fit_hdbscan(X_fit, mcs, ms)
    cent = centroids_and_radii(X_fit, labels_fit)
    fit_s = time.perf_counter() - t0

    if len(cent) == 0:
        return {"label": label, "rho": None, "n_matched": 0,
                "error": "no clusters on fit", "fit_seconds": float(fit_s)}

    cluster_assigned = assign_to_centroids(df_assign, cent)
    counts = Counter(cluster_assigned[cluster_assigned >= 0].tolist())

    n_fit = cent["size"].to_numpy(dtype=np.int64)
    n_assign = np.array(
        [counts.get(int(cid), 0) for cid in cent["cluster_id"]],
        dtype=np.int64,
    )
    keep = n_assign > 0
    n_matched = int(keep.sum())

    result = {
        "label": label,
        "n_fit": int(len(df_fit)),
        "n_assign": int(len(df_assign)),
        "n_clusters_fit": int(len(cent)),
        "coverage_assign": float((cluster_assigned >= 0).mean()),
        "n_matched": n_matched,
        "n_fit_median": int(np.median(n_fit[keep])) if keep.any() else 0,
        "n_assign_median": int(np.median(n_assign[keep])) if keep.any() else 0,
        "fit_seconds": float(fit_s),
    }

    if n_matched < 5:
        result.update({"rho": None, "p_value": None,
                       "message": "too few matched clusters"})
        return result

    rho, p = spearmanr(n_fit[keep], n_assign[keep])
    result.update({
        "rho": float(rho),
        "p_value": float(p),
    })
    return result


# ---------------------------------------------------------------------------
# Per-variant orchestration
# ---------------------------------------------------------------------------

def run_variant(df: pd.DataFrame, variant: str) -> dict:
    params = PARAMS[variant]
    mcs, ms = params["mcs"], params["ms"]

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()
    if variant == "injured":
        df_train = df_train[df_train["severity_code"] >= 1].copy()
        df_val = df_val[df_val["severity_code"] >= 1].copy()
        df_test = df_test[df_test["severity_code"] >= 1].copy()

    log.info(f"[{variant}] train: {len(df_train):,}  "
             f"val: {len(df_val):,}  test: {len(df_test):,}")

    comparisons = {}

    # 1. Sanity: fit train, assign train (rho should be ~1.0)
    log.info(f"[{variant}] sanity: train -> train")
    comparisons["sanity_train_to_train"] = assign_and_compare(
        df_train, df_train, mcs, ms, "train_full -> train_full (sanity)")

    # 2. Reference: train -> test (matches script 32)
    log.info(f"[{variant}] reference: train -> test")
    comparisons["train_full_to_test"] = assign_and_compare(
        df_train, df_test, mcs, ms, "train_full -> test (vs script 32)")

    # 3. Reference: train -> val
    log.info(f"[{variant}] reference: train -> val")
    comparisons["train_full_to_val"] = assign_and_compare(
        df_train, df_val, mcs, ms, "train_full -> val")

    # 4. Internal 50/50 of train (both directions)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)
    n = len(df_train)
    n_half = n // 2
    df_A = df_train.iloc[idx[:n_half]].copy()
    df_B = df_train.iloc[idx[n_half:]].copy()

    log.info(f"[{variant}] internal 50/50: A -> B  (A={len(df_A):,}, B={len(df_B):,})")
    comparisons["train_50_A_to_B"] = assign_and_compare(
        df_A, df_B, mcs, ms, "train_A -> train_B (50/50)")
    log.info(f"[{variant}] internal 50/50: B -> A")
    comparisons["train_50_B_to_A"] = assign_and_compare(
        df_B, df_A, mcs, ms, "train_B -> train_A (50/50)")

    # 5. Internal 75/25 of train (mimics train/val ratio)
    n_75 = int(0.75 * n)
    df_75 = df_train.iloc[idx[:n_75]].copy()
    df_25 = df_train.iloc[idx[n_75:]].copy()
    log.info(f"[{variant}] internal 75/25: 75 -> 25  "
             f"(75={len(df_75):,}, 25={len(df_25):,})")
    comparisons["train_75_to_25"] = assign_and_compare(
        df_75, df_25, mcs, ms, "train_75 -> train_25 (mimics val ratio)")

    # 6. Internal 50/50 of test
    rng2 = np.random.default_rng(SEED + 1)
    idx_test = np.arange(len(df_test))
    rng2.shuffle(idx_test)
    n_t = len(df_test)
    n_t_half = n_t // 2
    df_tA = df_test.iloc[idx_test[:n_t_half]].copy()
    df_tB = df_test.iloc[idx_test[n_t_half:]].copy()
    log.info(f"[{variant}] internal test 50/50: tA -> tB  "
             f"(tA={len(df_tA):,}, tB={len(df_tB):,})")
    comparisons["test_50_tA_to_tB"] = assign_and_compare(
        df_tA, df_tB, mcs, ms, "test_A -> test_B (50/50)")

    return {
        "variant": variant,
        "params": params,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v, digits=3):
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def render_markdown(results: dict, ref32: dict) -> str:
    lines = ["# Script 33 — Noise-ceiling validation for HDBSCAN stability\n"]
    lines.append(
        "**Question.** Is the Spearman rho from script 32 (0.51-0.60 for "
        "top-100, 0.73-0.82 for all clusters) at the methodological ceiling "
        "imposed by sample size and Poisson noise, or does it reveal real "
        "temporal drift?\n"
    )
    lines.append(
        "**Method.** The same fit-assign pipeline used in script 32 (`fit` "
        "HDBSCAN, `radius_eff = min(p90, 500 m)`, assign points within "
        "`radius_eff`, Spearman rho on matched cluster sizes), applied to "
        "random 50/50 and 75/25 splits of train and to a 50/50 split of "
        "test. If rho(internal split) ~ rho(train -> test), the temporal "
        "dimension adds nothing beyond sampling.\n"
    )
    lines.append(
        "**Decision rule.** if `rho(internal 50/50) ~= rho(train -> test)`: "
        "no temporal drift beyond sampling (ceiling reached). "
        "If `rho(internal) >> rho(train -> test)`: real temporal drift.\n"
    )

    for variant in ("all", "injured"):
        r = results[variant]
        p = r["params"]
        lines.append(f"## Variant `{variant}` (mcs={p['mcs']}, ms={p['ms']})\n")
        lines.append(
            f"Train: {r['n_train']:,}  ·  Val: {r['n_val']:,}  ·  "
            f"Test: {r['n_test']:,}\n"
        )
        lines.append(
            "| comparison | rho | n_matched | n_clusters_fit | coverage_assign |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key, comp in r["comparisons"].items():
            lines.append(
                f"| `{key}` | {_fmt(comp.get('rho'))} | "
                f"{comp['n_matched']} | {comp['n_clusters_fit']} | "
                f"{comp['coverage_assign']:.3f} |"
            )

        rho_50 = r["comparisons"]["train_50_A_to_B"].get("rho")
        rho_tt = r["comparisons"]["train_full_to_test"].get("rho")
        rho_75 = r["comparisons"]["train_75_to_25"].get("rho")
        rho_tv = r["comparisons"]["train_full_to_val"].get("rho")

        if rho_50 is not None and rho_tt is not None and rho_50 > 0:
            ratio_50 = rho_tt / rho_50
            lines.append(
                f"\n**Key comparison (vs test).** "
                f"rho(internal 50/50) = **{rho_50:.3f}**  ·  "
                f"rho(train -> test) = **{rho_tt:.3f}**  ·  "
                f"ratio = **{ratio_50:.3f}**"
            )
        if rho_75 is not None and rho_tv is not None and rho_75 > 0:
            ratio_75 = rho_tv / rho_75
            lines.append(
                f"\n**Key comparison (vs val).** "
                f"rho(internal 75/25) = **{rho_75:.3f}**  ·  "
                f"rho(train -> val) = **{rho_tv:.3f}**  ·  "
                f"ratio = **{ratio_75:.3f}**"
            )

        if rho_50 is not None and rho_tt is not None and rho_50 > 0:
            ratio = rho_tt / rho_50
            if ratio >= 0.95:
                lines.append(
                    "\n→ **Ratio ≈ 1.** The temporal split does *not* introduce "
                    "measurable stability loss beyond sampling noise. The rho "
                    "from script 32 is essentially what you would get from any "
                    "random 50/50 resample of the same data. **Ceiling reached.**"
                )
            elif ratio >= 0.85:
                lines.append(
                    "\n→ **Ratio ∈ [0.85, 0.95].** Mild temporal drift exists, "
                    "but the bulk of the observed rho is explained by sampling "
                    "noise. Consistent with 'stable structure, some reordering'."
                )
            else:
                lines.append(
                    "\n→ **Ratio < 0.85.** Substantial temporal drift beyond "
                    "sampling: the hotspot structure changed between 2019-2022 "
                    "and 2025."
                )

        # Cross-reference to script 32
        if variant in ref32:
            rv = ref32[variant]
            lines.append(
                f"\n_Reference (script 32, all-clusters rho): "
                f"train→val {_fmt(rv['rho_train_vs_val_all'])}, "
                f"train→test {_fmt(rv['rho_train_vs_test_all'])}._"
            )
        lines.append("\n---\n")

    lines.append("## Interpretation notes\n")
    lines.append(
        "- All comparisons use the **same fit-assign pipeline as script 32**: "
        "fit HDBSCAN on one side, compute centroids and `radius_eff = "
        "min(p90, 500m)`, assign the other side to the nearest centroid if "
        "within `radius_eff`, Spearman rho on clusters with at least one "
        "assigned point."
    )
    lines.append(
        "- **`sanity_train_to_train`**: fit and assign on the same data. rho "
        "should be ~1.0. Any deviation indicates assignment ambiguity from "
        "neighbouring clusters."
    )
    lines.append(
        "- **`train_50_A_to_B` / `train_50_B_to_A`**: two random halves of "
        "train. Same data distribution, different random sample, no temporal "
        "separation."
    )
    lines.append(
        "- **`train_75_to_25`**: 75% of train for fit, 25% for assign. Ratio "
        "close to train/val (74/21), so it is the closest internal analogue "
        "of the temporal comparison against val."
    )
    lines.append(
        "- **`train_full_to_test`**: the exact comparison of script 32, "
        "reproduced to verify that the pipeline is identical."
    )
    lines.append(
        "- The **key number** is `rho(train -> test) / rho(internal 50/50)`. "
        "Values around 1.0 mean sampling noise fully explains the observed "
        "correlation; values substantially below 1.0 mean temporal drift "
        "reduces stability beyond what random resampling of the same period "
        "would produce."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    df = load_inputs()
    ref32 = load_script32_reference()

    results = {}
    for variant in ("all", "injured"):
        log.info(f"=== Variant: {variant} ===")
        results[variant] = run_variant(df, variant)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(results, ref32), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    # Console summary
    log.info("=" * 72)
    for variant in ("all", "injured"):
        r = results[variant]
        log.info(f"[{variant}]")
        for key, comp in r["comparisons"].items():
            log.info(f"  {key:25s}  rho={_fmt(comp.get('rho'))}  "
                     f"n_matched={comp['n_matched']:4d}  "
                     f"coverage={comp['coverage_assign']:.3f}")
        rho_50 = r["comparisons"]["train_50_A_to_B"].get("rho")
        rho_tt = r["comparisons"]["train_full_to_test"].get("rho")
        if rho_50 and rho_tt:
            log.info(f"  → KEY  rho(train->test)/rho(50/50) = {rho_tt/rho_50:.3f}")


if __name__ == "__main__":
    main()