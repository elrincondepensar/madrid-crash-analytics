"""
Script 36 — Phase 7 iter 3: Option B (edge-weighted KDE) + C vs B comparison.

Fixes the accident→edge distance in `accidents_with_edges.parquet` using
sjoin_nearest over the LineString geometry, so `dist_to_line_m` is the
true distance to the nearest line (not to the midpoint).

Computes Option B: KDE 2D fitted on edge midpoints weighted by the
number of accidents assigned to each edge.

Compares C vs B per edge, per (bw, period, severity) with:
  - Spearman rho
  - Jaccard overlap of top-20% dense edges

Decision rule (D-36-3):
  - If mean rho >= 0.95 AND mean Jaccard >= 0.70: C wins (simpler).
  - Otherwise: B wins (network-aligned).

Produces `network_kde_final.parquet` with only the winning option at the
primary bandwidth (260 m), plus `network_kde_b.parquet` for audit.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.neighbors import KernelDensity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_TABLES = REPORTS / "tables"

ACCIDENTS_WITH_EDGES = DATA_PROCESSED / "accidents_with_edges.parquet"
NETWORK_PATH = DATA_PROCESSED / "network_osm.parquet"
KDE_C_PATH = DATA_PROCESSED / "network_kde_c.parquet"

OUT_B = DATA_PROCESSED / "network_kde_b.parquet"
OUT_FINAL = DATA_PROCESSED / "network_kde_final.parquet"
OUT_JSON = REPORTS_TABLES / "36_kde_compare.json"
OUT_MD = REPORTS / "36_kde_compare.md"

TARGET_CRS = "EPSG:25830"
BW_VALUES = (100, 150, 200, 260, 350, 500)
BW_PRIMARY = 260
PERIODS = ("train", "val", "test")
SEVERITIES = ("all", "injured")

N_JOBS = -1
log = logging.getLogger("phase7_compare_b")


# ---------------------------------------------------------------------------
# Robust loaders (same defensive pattern as 34/35, D076)
# ---------------------------------------------------------------------------

def _reconstruct_geometry(df: pd.DataFrame, geom_col: str) -> gpd.GeoDataFrame:
    sample = df[geom_col].dropna().iloc[0] if df[geom_col].notna().any() else None
    if sample is None:
        raise RuntimeError(f"geometry column '{geom_col}' is entirely null")
    if isinstance(sample, (bytes, bytearray)):
        from shapely import wkb
        geoms = df[geom_col].apply(
            lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else None)
    elif isinstance(sample, str):
        from shapely import wkt
        geoms = df[geom_col].apply(
            lambda g: wkt.loads(g) if isinstance(g, str) and g else None)
    else:
        geoms = df[geom_col]
    return gpd.GeoDataFrame(df.drop(columns=[geom_col]),
                            geometry=geoms, crs="EPSG:4326")


def load_network_robust(path: Path) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_parquet(path)
    except (ValueError, KeyError):
        df = pd.read_parquet(path)
        geom_col = next((c for c in ("geometry", "geometry_wkt", "geom",
                                     "wkt", "geometry_wkb")
                         if c in df.columns), None)
        if geom_col is None:
            raise RuntimeError(f"no geometry column in {path}")
        gdf = _reconstruct_geometry(df, geom_col)
    if "edge_id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["edge_id"] = np.arange(len(gdf), dtype=np.int64)
    gdf["edge_id"] = gdf["edge_id"].astype(np.int64)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def compute_edge_midpoints(net: gpd.GeoDataFrame) -> pd.DataFrame:
    midpoints = net.geometry.interpolate(0.5, normalized=True)
    out = pd.DataFrame({
        "edge_id": net["edge_id"].astype(np.int64).to_numpy(),
        "mid_x":   midpoints.x.to_numpy(),
        "mid_y":   midpoints.y.to_numpy(),
    })
    return out.dropna(subset=["mid_x", "mid_y"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load & fix distance
# ---------------------------------------------------------------------------

def load_and_fix_distance() -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    log.info("Loading accidents_with_edges + network ...")
    df = pd.read_parquet(ACCIDENTS_WITH_EDGES)
    net = load_network_robust(NETWORK_PATH)
    if net.crs.to_epsg() != 25830:
        net = net.to_crs(TARGET_CRS)

    log.info("Fixing distance to line via sjoin_nearest ...")
    t0 = time.perf_counter()
    gdf_acc = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["coordenada_x_utm"],
                                    df["coordenada_y_utm"]),
        crs=TARGET_CRS,
    )
    joined = gpd.sjoin_nearest(
        gdf_acc, net[["geometry"]], how="left", distance_col="dist_to_line_m",
    )
    joined = joined.drop_duplicates(subset="num_expediente", keep="first")
    df["dist_to_line_m"] = joined["dist_to_line_m"].to_numpy()

    log.info(f"  sjoin_nearest done in {time.perf_counter() - t0:.1f}s")
    log.info(f"  dist_to_line_m: median={np.median(df['dist_to_line_m']):.2f} m, "
             f"p90={np.percentile(df['dist_to_line_m'], 90):.2f} m, "
             f"max={df['dist_to_line_m'].max():.1f} m")
    log.info(f"  dist_to_edge_midpoint_m (old): "
             f"median={np.median(df['dist_to_edge_midpoint_m']):.2f} m")

    # Persist fixed distances back to disk
    df.to_parquet(ACCIDENTS_WITH_EDGES, index=False)
    log.info(f"saved {ACCIDENTS_WITH_EDGES.name} (with dist_to_line_m)")

    mp_by_edge = compute_edge_midpoints(net)
    return df, net, mp_by_edge


# ---------------------------------------------------------------------------
# Option B KDE
# ---------------------------------------------------------------------------

def kde_option_b(df_period: pd.DataFrame,
                 mp_by_edge: pd.DataFrame,
                 bw: float) -> np.ndarray:
    if len(df_period) < 5:
        return np.full(len(mp_by_edge), np.nan)

    counts = (
        df_period["edge_id"].astype(np.int64).value_counts()
    )
    if len(counts) == 0:
        return np.full(len(mp_by_edge), np.nan)

    src_edges = counts.index.to_numpy(dtype=np.int64)
    src_weights = counts.to_numpy(dtype=np.float64)
    src_xy = (
        mp_by_edge.set_index("edge_id")
        .reindex(src_edges)[["mid_x", "mid_y"]]
        .to_numpy(dtype=np.float64)
    )
    mask = ~np.isnan(src_xy).any(axis=1)
    src_xy = src_xy[mask]
    src_weights = src_weights[mask]

    if len(src_xy) < 5:
        return np.full(len(mp_by_edge), np.nan)

    kde = KernelDensity(
        bandwidth=float(bw),
        kernel="gaussian",
        algorithm="ball_tree",
        leaf_size=40,
        metric="euclidean",
    )
    kde.fit(src_xy, sample_weight=src_weights)

    eval_xy = mp_by_edge[["mid_x", "mid_y"]].to_numpy(dtype=np.float64)
    return np.exp(kde.score_samples(eval_xy))


def _one_job_b(df: pd.DataFrame, severity: str, period: str,
               bw: int, mp_by_edge: pd.DataFrame) -> tuple:
    t0 = time.perf_counter()
    sub = df if severity == "all" else df[df["severity_code"] >= 1]
    sub = sub[sub["split"] == period]
    col = f"kde_b_{period}_{severity}_bw{bw}"
    scores = kde_option_b(sub, mp_by_edge, bw)
    return severity, period, bw, col, scores, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# C vs B comparison
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() < 100:
        return None
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def _jaccard_top20(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() < 100:
        return None
    aa = a[mask]
    bb = b[mask]
    thr_a = np.percentile(aa, 80)
    thr_b = np.percentile(bb, 80)
    top_a = aa >= thr_a
    top_b = bb >= thr_b
    union = (top_a | top_b).sum()
    if union == 0:
        return None
    return float((top_a & top_b).sum() / union)


def compare_c_vs_b(kde_c: pd.DataFrame, kde_b: pd.DataFrame) -> dict:
    results = {}
    rhos, jaccards = [], []
    for bw in BW_VALUES:
        for period in PERIODS:
            for sev in SEVERITIES:
                col = f"kde_{period}_{sev}_bw{bw}"
                if col not in kde_c.columns or col not in kde_b.columns:
                    continue
                a = kde_c[col].to_numpy()
                b = kde_b[col].to_numpy()
                rho = _spearman(a, b)
                jac = _jaccard_top20(a, b)
                results[col] = {"rho": rho, "jaccard_top20": jac}
                if rho is not None:
                    rhos.append(rho)
                if jac is not None:
                    jaccards.append(jac)

    return {
        "per_combo": results,
        "n_comparisons": len(rhos),
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "min_rho": float(np.min(rhos)) if rhos else None,
        "mean_jaccard_top20": float(np.mean(jaccards)) if jaccards else None,
        "min_jaccard_top20": float(np.min(jaccards)) if jaccards else None,
    }


def decide(c_vs_b: dict) -> dict:
    mean_rho = c_vs_b["mean_rho"]
    mean_jac = c_vs_b["mean_jaccard_top20"]
    if mean_rho is None or mean_jac is None:
        return {"winner": "unknown",
                "reason": "missing metrics"}
    if mean_rho >= 0.95 and mean_jac >= 0.70:
        return {"winner": "C",
                "reason": f"C and B agree strongly "
                          f"(rho={mean_rho:.3f}, jaccard={mean_jac:.3f}); "
                          f"C is simpler."}
    return {"winner": "B",
            "reason": f"C and B diverge "
                      f"(rho={mean_rho:.3f}, jaccard={mean_jac:.3f}); "
                      f"B uses the network source and is preferred."}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(results: dict) -> str:
    lines = ["# Phase 7 · Iter 3 — Option B & C vs B comparison\n"]
    lines.append(
        "_Option B = 2D KDE fitted on edge midpoints weighted by accident "
        "counts per edge. Same evaluation grid (edge midpoints), same "
        "Gaussian kernel, same bandwidths as Option C._\n"
    )

    fix = results["distance_fix"]
    lines.append("## Distance fix (sjoin_nearest)\n")
    lines.append(
        f"- Old (`dist_to_edge_midpoint_m`): "
        f"median **{fix['old_median']:.2f} m**, p90 "
        f"{fix['old_p90']:.2f} m, max {fix['old_max']:.0f} m"
    )
    lines.append(
        f"- New (`dist_to_line_m`): "
        f"median **{fix['new_median']:.2f} m**, p90 "
        f"{fix['new_p90']:.2f} m, max {fix['new_max']:.0f} m"
    )
    lines.append(
        f"\n_Persisted to `accidents_with_edges.parquet`. "
        f"The old column is kept for audit._\n"
    )

    c = results["c_vs_b"]
    lines.append("## C vs B comparison\n")
    lines.append(
        f"Compared over **{c['n_comparisons']}** (bw, period, severity) "
        f"combinations. Per-edge Spearman rho and Jaccard overlap of the "
        f"top-20% dense edges.\n"
    )
    lines.append("| metric | mean | min |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| Spearman rho (C, B) | {_fmt(c['mean_rho'])} | "
                 f"{_fmt(c['min_rho'])} |")
    lines.append(f"| Jaccard (top-20% dense) | {_fmt(c['mean_jaccard_top20'])} | "
                 f"{_fmt(c['min_jaccard_top20'])} |")

    lines.append(f"\n**Winner: `{results['decision']['winner']}`** — "
                 f"{results['decision']['reason']}\n")

    lines.append("## Per-combination table\n")
    lines.append("| col | rho | jaccard_top20 |")
    lines.append("| --- | ---: | ---: |")
    for col, v in sorted(c["per_combo"].items()):
        lines.append(f"| `{col}` | {_fmt(v['rho'])} | "
                     f"{_fmt(v['jaccard_top20'])} |")
    lines.append("")

    lines.append("## Decision rule (D-36-3)\n")
    lines.append(
        "- `mean_rho ≥ 0.95` **AND** `mean_jaccard_top20 ≥ 0.70` → **C wins** "
        "(simpler, no information loss)."
    )
    lines.append(
        "- Otherwise → **B wins** (network-aligned source)."
    )
    lines.append(
        "- `network_kde_final.parquet` contains only the winning option at "
        f"bw = {BW_PRIMARY} m (3 periods × 2 severities = 6 columns)."
    )

    lines.append("\n## Notes\n")
    lines.append(
        "- **Option B source is different, not just weighted.** C uses "
        "137k raw accident points. B uses ~10-20k edge midpoints (the edges "
        "with at least one accident in the period), each weighted by its "
        "accident count. Total mass is comparable; within-edge dispersion "
        "is lost (each edge contributes a single point). This is the price "
        "of the approximation."
    )
    lines.append(
        "- **Jaccard** measures the fraction of top-20% dense edges that "
        "the two options agree on. High Jaccard = identical hotspot "
        "rankings at the top. Low Jaccard = the top hotspots differ."
    )
    lines.append(
        "- **If B wins**, the reason is *not* that C is wrong — C is the "
        "standard 2D KDE. B is preferred because the sources align with "
        "the network structure that the map represents, which is the "
        "conceptual target of Phase 7."
    )
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
    df, net, mp_by_edge = load_and_fix_distance()

    # Snapshot of the fix for the report
    fix = {
        "old_median": float(np.median(df["dist_to_edge_midpoint_m"])),
        "old_p90":    float(np.percentile(df["dist_to_edge_midpoint_m"], 90)),
        "old_max":    float(df["dist_to_edge_midpoint_m"].max()),
        "new_median": float(np.median(df["dist_to_line_m"])),
        "new_p90":    float(np.percentile(df["dist_to_line_m"], 90)),
        "new_max":    float(df["dist_to_line_m"].max()),
    }

    # Load Option C
    log.info("Loading Option C ...")
    kde_c = pd.read_parquet(KDE_C_PATH)
    if "edge_id" not in kde_c.columns:
        raise RuntimeError("network_kde_c.parquet must have edge_id")

    # Compute Option B
    kde_b = mp_by_edge[["edge_id"]].copy()
    n_jobs_total = len(SEVERITIES) * len(PERIODS) * len(BW_VALUES)
    log.info(f"Running {n_jobs_total} Option B jobs on {N_JOBS} workers ...")
    t0 = time.perf_counter()
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_one_job_b)(df, sev, period, bw, mp_by_edge)
        for sev in SEVERITIES
        for period in PERIODS
        for bw in BW_VALUES
    )
    log.info(f"  jobs done in {time.perf_counter() - t0:.1f}s")

    for _sev, _period, _bw, col, scores, _elapsed in results:
        kde_b[col] = scores

    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    kde_b.to_parquet(OUT_B, index=False)
    log.info(f"saved {OUT_B.relative_to(ROOT)} "
             f"({len(kde_b):,} rows × {len(kde_b.columns)} cols)")

    # Compare
    log.info("Comparing C vs B ...")
    c_vs_b = compare_c_vs_b(kde_c, kde_b)
    log.info(f"  mean_rho={_fmt(c_vs_b['mean_rho'])}  "
             f"min_rho={_fmt(c_vs_b['min_rho'])}  "
             f"mean_jaccard={_fmt(c_vs_b['mean_jaccard_top20'])}  "
             f"min_jaccard={_fmt(c_vs_b['min_jaccard_top20'])}")

    decision = decide(c_vs_b)
    log.info(f"  DECISION: {decision['winner']} — {decision['reason']}")

    # Produce final parquet at BW_PRIMARY with winning option
    log.info(f"Building network_kde_final.parquet (option {decision['winner']}, "
             f"bw={BW_PRIMARY}) ...")
    src = kde_c if decision["winner"] == "C" else kde_b
    out = src[["edge_id"]].copy()
    for period in PERIODS:
        for sev in SEVERITIES:
            col = f"kde_{period}_{sev}_bw{BW_PRIMARY}"
            if col not in src.columns:
                log.warning(f"  missing {col} in winning option")
                continue
            new_col = f"kde_{period}_{sev}_bw{BW_PRIMARY}"
            out[new_col] = src[col].values
    out.to_parquet(OUT_FINAL, index=False)
    log.info(f"saved {OUT_FINAL.relative_to(ROOT)} "
             f"({len(out):,} rows × {len(out.columns)} cols)")

    results_json = {
        "bandwidths_m": list(BW_VALUES),
        "bw_primary": BW_PRIMARY,
        "periods": list(PERIODS),
        "severities": list(SEVERITIES),
        "crs": TARGET_CRS,
        "n_edges": int(len(kde_b)),
        "distance_fix": fix,
        "c_vs_b": c_vs_b,
        "decision": decision,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results_json), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(results_json), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    log.info("=" * 72)
    log.info(f"WINNER: {decision['winner']}  "
             f"(mean rho={_fmt(c_vs_b['mean_rho'])}, "
             f"mean jaccard={_fmt(c_vs_b['mean_jaccard_top20'])})")


if __name__ == "__main__":
    main()