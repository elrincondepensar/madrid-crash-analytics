"""
Script 34 — Phase 7 iter 1: Network KDE baseline (Option C).

Option C = 2D KDE fitted on raw accident coordinates, evaluated at the
midpoints of the OSM road network edges. This is the "QGIS heatmap
baseline": no network information is used to define the source, but the
evaluation grid is the network, so the result is comparable per edge.

Deliverable
-----------
- `data/processed/network_kde_c.parquet`:
      edge_id + kde_{period}_{severity}_bw{bw} columns.
      One column per (period, severity, bandwidth).
- `reports/tables/34_kde_bw_selection.json`
- `reports/34_kde.md`

Bandwidth selection
-------------------
For each (bw, severity), compute:
  - stability = mean Spearman rho between periods, over edges
    (train↔val, train↔test, val↔test)
  - contrast = mean Spearman rho between density_all and density_injured,
    across periods

Criteria:
  - stability >= 0.80
  - contrast in [0.60, 0.90]
  - choose the smallest bw passing both. Fallback: best score.

No refit on val/test — the KDE is computed independently on each period,
so no leakage risk. All comparison is descriptive.

Option B (edge-weighted KDE) and C-vs-B comparison live in script 35.
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

ACCIDENTS_CLEAN = DATA_PROCESSED / "accidents_clean.parquet"
SPLITS_PATH = DATA_PROCESSED / "splits.parquet"
NETWORK_PATH = DATA_PROCESSED / "network_osm.parquet"

OUT_PARQUET = DATA_PROCESSED / "network_kde_c.parquet"
OUT_JSON = REPORTS_TABLES / "34_kde_bw_selection.json"
OUT_MD = REPORTS / "34_kde.md"

TARGET_CRS = "EPSG:25830"
BANDWIDTHS_M = (100, 150, 200, 260, 350, 500)
PERIODS = ("train", "val", "test")
SEVERITIES = ("all", "injured")

BW_STABILITY_MIN = 0.80
BW_CONTRAST_RANGE = (0.60, 0.90)

N_JOBS = -1

log = logging.getLogger("phase7_kde_c")


# ---------------------------------------------------------------------------
# Loading helpers (defensive against missing geo metadata — see D076)
# ---------------------------------------------------------------------------

def _reconstruct_geometry(df: pd.DataFrame, geom_col: str) -> gpd.GeoDataFrame:
    """Rebuild a GeoDataFrame from WKB / WKT / shapely column."""
    sample = df[geom_col].dropna().iloc[0] if df[geom_col].notna().any() else None
    if sample is None:
        raise RuntimeError(f"geometry column '{geom_col}' is entirely null")

    if isinstance(sample, (bytes, bytearray)):
        from shapely import wkb
        log.info(f"  parsing '{geom_col}' as WKB")
        geoms = df[geom_col].apply(
            lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else None
        )
    elif isinstance(sample, str):
        from shapely import wkt
        log.info(f"  parsing '{geom_col}' as WKT")
        geoms = df[geom_col].apply(
            lambda g: wkt.loads(g) if isinstance(g, str) and g else None
        )
    else:
        log.info(f"  '{geom_col}' holds shapely objects")
        geoms = df[geom_col]

    n_bad = int(geoms.isna().sum())
    if n_bad:
        log.warning(f"  {n_bad} rows with unparseable geometry")
    return gpd.GeoDataFrame(df.drop(columns=[geom_col]),
                            geometry=geoms, crs="EPSG:4326")


def load_network_robust(path: Path) -> gpd.GeoDataFrame:
    """Load network_osm.parquet, tolerating missing geo metadata."""
    try:
        gdf = gpd.read_parquet(path)
        log.info("  loaded with geopandas (geo metadata present)")
    except (ValueError, KeyError) as e:
        log.info(f"  geopandas read failed ({type(e).__name__}); "
                 f"falling back to pandas + reconstruction")
        df = pd.read_parquet(path)
        geom_col = next(
            (c for c in ("geometry", "geometry_wkt", "geom", "wkt",
                         "geometry_wkb") if c in df.columns),
            None,
        )
        if geom_col is None:
            raise RuntimeError(
                f"no geometry column in {path}. Columns: {list(df.columns)}"
            )
        gdf = _reconstruct_geometry(df, geom_col)

    if "edge_id" not in gdf.columns:
        if gdf.index.name == "edge_id":
            gdf = gdf.reset_index()
            log.info("  edge_id restored from index name")
        else:
            gdf = gdf.reset_index(drop=True)
            gdf["edge_id"] = np.arange(len(gdf), dtype=np.int64)
            log.warning("  edge_id column absent; using positional index")
    gdf["edge_id"] = gdf["edge_id"].astype(np.int64)

    if gdf.crs is None:
        log.warning("  no CRS set; assuming EPSG:4326 (per D072)")
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def load_inputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load accidents + network for Option C.

    Option C evaluates 2D KDE from raw accident UTM coordinates at edge
    midpoints. No accident↔edge assignment is required (that need only
    exists for Option B, computed in script 35 from raw coordinates).
    """
    log.info("Loading accidents + network ...")
    acc = pd.read_parquet(ACCIDENTS_CLEAN)
    spl = pd.read_parquet(SPLITS_PATH)
    net = load_network_robust(NETWORK_PATH)

    need_clean = {"num_expediente", "severity_code",
                  "coordenada_x_utm", "coordenada_y_utm"}
    missing = need_clean - set(acc.columns)
    if missing:
        raise RuntimeError(f"accidents_clean missing: {missing}")

    df = acc[["num_expediente", "severity_code",
              "coordenada_x_utm", "coordenada_y_utm"]].merge(
        spl[["num_expediente", "split"]],
        on="num_expediente", how="inner", validate="one_to_one",
    )
    df = df.dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()
    log.info(f"  accidents: {len(df):,}  |  "
             f"splits: {df['split'].value_counts().to_dict()}")
    log.info(f"  severity: "
             f"{df['severity_code'].value_counts().sort_index().to_dict()}")

    # Reproyectar red a UTM si hace falta
    if net.crs is None:
        raise RuntimeError("network loaded without CRS")
    if net.crs.to_epsg() != 25830:
        log.info(f"  reprojecting network {net.crs.to_epsg()} -> EPSG:25830")
        net = net.to_crs(TARGET_CRS)

    types = set(net.geometry.geom_type.dropna().unique())
    if types - {"LineString", "MultiLineString"}:
        log.warning(f"  non-LineString geometries present: {types}")
    n_valid = int(net.geometry.notna().sum())
    log.info(f"  network edges: {len(net):,}  "
             f"(with valid geometry: {n_valid:,})")

    return df, net


def compute_edge_midpoints(net: gpd.GeoDataFrame) -> pd.DataFrame:
    log.info(f"Computing edge midpoints for {len(net):,} edges ...")
    midpoints = net.geometry.interpolate(0.5, normalized=True)
    df = pd.DataFrame({
        "edge_id": net["edge_id"].astype(np.int64).to_numpy(),
        "mid_x":   midpoints.x.to_numpy(),
        "mid_y":   midpoints.y.to_numpy(),
    })
    n_nan = int(df[["mid_x", "mid_y"]].isna().any(axis=1).sum())
    if n_nan:
        log.warning(f"  {n_nan} edges with NaN midpoint, dropped")
        df = df.dropna(subset=["mid_x", "mid_y"]).reset_index(drop=True)
    log.info(f"  final edges: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# KDE
# ---------------------------------------------------------------------------

def get_period_subset(df: pd.DataFrame, severity: str,
                      period: str) -> pd.DataFrame:
    sub = df if severity == "all" else df[df["severity_code"] >= 1]
    return sub[sub["split"] == period].copy()


def kde_scores(source_xy: np.ndarray,
               eval_xy: np.ndarray,
               bw: float) -> np.ndarray:
    kde = KernelDensity(
        bandwidth=bw, kernel="gaussian", algorithm="ball_tree",
        leaf_size=40, metric="euclidean",
    )
    kde.fit(source_xy)
    return np.exp(kde.score_samples(eval_xy))


def _one_job(df: pd.DataFrame, severity: str, period: str, bw: int,
             midpoints_xy: np.ndarray) -> tuple:
    t0 = time.perf_counter()
    sub = get_period_subset(df, severity, period)
    n = len(sub)
    col = f"kde_{period}_{severity}_bw{bw}"

    if n < 5:
        scores = np.full(len(midpoints_xy), np.nan)
    else:
        src = sub[["coordenada_x_utm", "coordenada_y_utm"]].to_numpy(
            dtype=np.float64)
        scores = kde_scores(src, midpoints_xy, float(bw))

    return severity, period, bw, col, scores, n, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 100:
        return None
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def stability_table(kde_matrix: pd.DataFrame, severity: str, bw: int) -> dict:
    cols = {p: f"kde_{p}_{severity}_bw{bw}" for p in PERIODS}
    pairs = {}
    for p1, p2 in (("train", "val"), ("train", "test"), ("val", "test")):
        if cols[p1] in kde_matrix and cols[p2] in kde_matrix:
            pairs[f"{p1}_vs_{p2}"] = _spearman(
                kde_matrix[cols[p1]].to_numpy(),
                kde_matrix[cols[p2]].to_numpy(),
            )
    valid = [v for v in pairs.values() if v is not None]
    pairs["mean"] = float(np.mean(valid)) if valid else None
    return pairs


def contrast_per_period(kde_matrix: pd.DataFrame, period: str,
                        bw: int) -> float | None:
    c_all = f"kde_{period}_all_bw{bw}"
    c_inj = f"kde_{period}_injured_bw{bw}"
    if c_all not in kde_matrix or c_inj not in kde_matrix:
        return None
    return _spearman(kde_matrix[c_all].to_numpy(),
                     kde_matrix[c_inj].to_numpy())


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def select_bandwidth(kde_matrix: pd.DataFrame) -> dict:
    table = []
    for bw in BANDWIDTHS_M:
        stab_all = stability_table(kde_matrix, "all", bw)
        stab_inj = stability_table(kde_matrix, "injured", bw)
        contrasts = {
            p: contrast_per_period(kde_matrix, p, bw) for p in PERIODS
        }
        cv = [v for v in contrasts.values() if v is not None]
        contrast_mean = float(np.mean(cv)) if cv else None
        table.append({
            "bw": bw,
            "stability_all": stab_all["mean"],
            "stability_injured": stab_inj["mean"],
            "contrast_per_period": contrasts,
            "contrast_mean": contrast_mean,
            "stability_detail_all": stab_all,
            "stability_detail_injured": stab_inj,
        })

    passes_strict = [
        r for r in table
        if (r["stability_all"] is not None
            and r["stability_all"] >= BW_STABILITY_MIN
            and r["contrast_mean"] is not None
            and BW_CONTRAST_RANGE[0] <= r["contrast_mean"] <= BW_CONTRAST_RANGE[1])
    ]

    if passes_strict:
        chosen = passes_strict[0]["bw"]
        reason = "smallest_passing"
    else:
        def score(r):
            s = r["stability_all"]
            c = r["contrast_mean"]
            if s is None or c is None:
                return -np.inf
            c_ok = max(0.0, 1.0 - abs(c - 0.75) / 0.15)
            return s + 0.5 * c_ok
        best = max(table, key=score)
        chosen = best["bw"]
        reason = "best_score_fallback"

    return {
        "table": table,
        "strict_passes": [r["bw"] for r in passes_strict],
        "chosen_bw": chosen,
        "chosen_reason": reason,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(results: dict) -> str:
    lines = ["# Phase 7 · Iter 1 — Network KDE baseline (Option C)\n"]
    lines.append(
        "_Option C = 2D KDE on raw accidents, evaluated at edge midpoints. "
        "Gaussian kernel. Bandwidths in meters._\n"
    )
    sel = results["selection"]
    lines.append("## Bandwidth selection table\n")
    lines.append("| bw (m) | stability_all | stability_injured | contrast_mean |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in sel["table"]:
        lines.append(
            f"| {row['bw']} | {_fmt(row['stability_all'])} | "
            f"{_fmt(row['stability_injured'])} | {_fmt(row['contrast_mean'])} |"
        )
    lines.append(
        f"\n**Chosen bandwidth: {sel['chosen_bw']} m** "
        f"(`{sel['chosen_reason']}`). "
        f"Strict passes: {sel['strict_passes'] or 'none'}.\n"
    )

    lines.append("## Stability detail (chosen bw)\n")
    for sev in SEVERITIES:
        st = stability_table_summary(results, sev)
        lines.append(f"**{sev}**: " + " · ".join(
            f"{k}={_fmt(v)}" for k, v in st.items()
        ))
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- **Stability** = mean Spearman rho between KDE densities across "
        "the three period pairs (train↔val, train↔test, val↔test). "
        "Measures whether the spatial pattern is preserved over time."
    )
    lines.append(
        "- **Severity contrast** = Spearman rho between `density_all` and "
        "`density_injured` per period. Values near 1 mean the bw cannot "
        "distinguish severity composition; values in 0.6–0.9 mean the bw "
        "reveals structural differences; below 0.6 suggests injured too "
        "sparse at that scale."
    )
    lines.append(
        "- **Selection** prefers the smallest bw passing both thresholds "
        "(`stability ≥ 0.80`, `contrast ∈ [0.60, 0.90]`). Fallback: score "
        "`stability + 0.5·contrast_quality`. Larger bw trivially increases "
        "stability, so smallest-passing avoids choosing bw driven by "
        "smoothing rather than signal."
    )
    lines.append(
        "- **KDE is computed independently on each period.** No refit on "
        "train/val/test in the ML sense — each period is treated as an "
        "independent sample of the same underlying process. Comparisons "
        "are descriptive, not predictive."
    )
    return "\n".join(lines)


def stability_table_summary(results: dict, severity: str) -> dict:
    """Rebuild the stability dict for the chosen bw from stored table."""
    bw = results["selection"]["chosen_bw"]
    for row in results["selection"]["table"]:
        if row["bw"] == bw:
            key = f"stability_detail_{severity}"
            return row.get(key) or {}
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    df, net = load_inputs()
    mp_by_edge = compute_edge_midpoints(net)
    midpoints_xy = mp_by_edge[["mid_x", "mid_y"]].to_numpy(dtype=np.float64)

    kde_matrix = mp_by_edge[["edge_id"]].copy()
    n_jobs_total = len(SEVERITIES) * len(PERIODS) * len(BANDWIDTHS_M)
    log.info(f"Running {n_jobs_total} KDE jobs on {N_JOBS} workers ...")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_one_job)(df, sev, period, bw, midpoints_xy)
        for sev in SEVERITIES
        for period in PERIODS
        for bw in BANDWIDTHS_M
    )
    log.info(f"  jobs done in {time.perf_counter() - t0:.1f}s")

    for sev, period, bw, col, scores, n, _elapsed in results:
        kde_matrix[col] = scores

    selection = select_bandwidth(kde_matrix)
    log.info(f"Chosen bw: {selection['chosen_bw']} m "
             f"({selection['chosen_reason']})")
    if selection["strict_passes"]:
        log.info(f"  strict passes: {selection['strict_passes']}")
    else:
        log.warning("  no bw passed strict criteria — fallback used")

    results_json = {
        "bandwidths_m": list(BANDWIDTHS_M),
        "periods": list(PERIODS),
        "severities": list(SEVERITIES),
        "crs": TARGET_CRS,
        "n_edges": int(len(kde_matrix)),
        "strict_criteria": {
            "stability_min": BW_STABILITY_MIN,
            "contrast_range": list(BW_CONTRAST_RANGE),
        },
        "selection": selection,
        "option": "C",
    }

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    kde_matrix.to_parquet(OUT_PARQUET, index=False)
    log.info(f"saved {OUT_PARQUET.relative_to(ROOT)} "
             f"({len(kde_matrix):,} rows × {len(kde_matrix.columns)} cols)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results_json), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(results_json), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    log.info("=" * 72)
    log.info(f"CHOSEN: bw = {selection['chosen_bw']} m "
             f"({selection['chosen_reason']})")


if __name__ == "__main__":
    main()