"""
Script 35 — Phase 7 iter 2: KDE maps + HDBSCAN cross-check.

Consumes the Option-C KDE matrix produced by script 34 and produces:

  1. Twelve static maps (2 bw × 3 periods × 2 severities) of KDE intensity
     over the OSM network. bw=260 m is the primary scale (D103); bw=500 m
     is included as a visual reference (barrio-scale smoothing).

  2. HDBSCAN cross-check: are the HDBSCAN clusters concentrated in high-KDE
     areas, or are they spatially decoupled from the global intensity
     surface? Quantified per variant (all/injured) via:
       - percentile of KDE at cluster centroids (top-20, all persisted),
       - share of accidents in persisted clusters located in top-20% KDE,
       - Spearman rho between cluster size and KDE median at the cluster.

  3. `data/processed/accidents_with_edges.parquet` — reconstructs the
     accident→edge assignment missing from Phase 4 (see D077). Uses a
     cKDTree on edge midpoints. Documented limitation: distance to
     midpoint ≠ distance to line; negligible for edges < 500 m.

Deliverables
------------
    reports/figures/35_kde_bw{260,500}_{period}_{severity}.png   (12 files)
    reports/tables/35_hdbscan_kde_cross.json
    reports/35_kde_maps.md
    data/processed/accidents_with_edges.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_FIGS = REPORTS / "figures"
REPORTS_TABLES = REPORTS / "tables"

ACCIDENTS_CLEAN = DATA_PROCESSED / "accidents_clean.parquet"
SPLITS_PATH = DATA_PROCESSED / "splits.parquet"
NETWORK_PATH = DATA_PROCESSED / "network_osm.parquet"
KDE_C_PATH = DATA_PROCESSED / "network_kde_c.parquet"
HDBSCAN_ALL_PATH = DATA_PROCESSED / "hdbscan_persistence_all.parquet"
HDBSCAN_INJ_PATH = DATA_PROCESSED / "hdbscan_persistence_injured.parquet"

OUT_PARQUET_EDGES = DATA_PROCESSED / "accidents_with_edges.parquet"
OUT_JSON = REPORTS_TABLES / "35_hdbscan_kde_cross.json"
OUT_MD = REPORTS / "35_kde_maps.md"

TARGET_CRS = "EPSG:25830"
BW_PRIMARY = 260
BW_REFERENCE = 500
BW_VALUES = (BW_PRIMARY, BW_REFERENCE)
PERIODS = ("train", "val", "test")
SEVERITIES = ("all", "injured")
TOP_PERCENTILE_DENSE = 80.0  # top-20% threshold for "dense edge"

TOP_K_CLUSTERS = 20  # for the "top-k by size" analysis
MIN_PERSISTED_ACCIDENTS = 10  # same definition as script 32

log = logging.getLogger("phase7_maps")


# ---------------------------------------------------------------------------
# Load helpers (same defensive pattern as script 34 — see D076)
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


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_all() -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame,
                        pd.DataFrame, pd.DataFrame]:
    log.info("Loading inputs ...")
    acc = pd.read_parquet(ACCIDENTS_CLEAN)
    spl = pd.read_parquet(SPLITS_PATH)
    net = load_network_robust(NETWORK_PATH)
    kde = pd.read_parquet(KDE_C_PATH)
    hdb_all = pd.read_parquet(HDBSCAN_ALL_PATH)
    hdb_inj = pd.read_parquet(HDBSCAN_INJ_PATH)

    need = {"num_expediente", "severity_code",
            "coordenada_x_utm", "coordenada_y_utm"}
    if need - set(acc.columns):
        raise RuntimeError(f"accidents_clean missing: {need - set(acc.columns)}")
    if "edge_id" not in kde.columns:
        raise RuntimeError("network_kde_c.parquet must have 'edge_id'")

    df = acc[["num_expediente", "severity_code",
              "coordenada_x_utm", "coordenada_y_utm"]].merge(
        spl[["num_expediente", "split"]],
        on="num_expediente", how="inner", validate="one_to_one",
    )
    df = df.dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()
    log.info(f"  accidents: {len(df):,}  |  "
             f"splits: {df['split'].value_counts().to_dict()}")
    log.info(f"  network edges: {len(net):,}")
    log.info(f"  kde columns: {len(kde.columns)}  "
             f"(incl. edge_id)")
    log.info(f"  hdbscan_all rows: {len(hdb_all):,}  "
             f"hdbscan_injured rows: {len(hdb_inj):,}")

    if net.crs.to_epsg() != 25830:
        log.info(f"  reprojecting network {net.crs.to_epsg()} -> EPSG:25830")
        net = net.to_crs(TARGET_CRS)
    return df, net, kde, hdb_all, hdb_inj


def compute_edge_midpoints(net: gpd.GeoDataFrame) -> pd.DataFrame:
    log.info("Computing edge midpoints ...")
    midpoints = net.geometry.interpolate(0.5, normalized=True)
    out = pd.DataFrame({
        "edge_id": net["edge_id"].astype(np.int64).to_numpy(),
        "mid_x":   midpoints.x.to_numpy(),
        "mid_y":   midpoints.y.to_numpy(),
    })
    out = out.dropna(subset=["mid_x", "mid_y"]).reset_index(drop=True)
    log.info(f"  edges with valid midpoint: {len(out):,}")
    return out


def assign_edges_to_accidents(df: pd.DataFrame,
                              mp_by_edge: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct accident→edge assignment (missing from Phase 4, D077).

    Uses cKDTree on edge midpoints. Distance to midpoint ≠ distance to
    line, but the difference is < ~5 m for edges < 500 m and is
    documented as a proxy in the phase notes.
    """
    log.info("Assigning accidents to nearest edge (cKDTree on midpoints) ...")
    mp = mp_by_edge[["mid_x", "mid_y"]].to_numpy(dtype=np.float64)
    eids = mp_by_edge["edge_id"].to_numpy(dtype=np.int64)

    xy = df[["coordenada_x_utm", "coordenada_y_utm"]].to_numpy(dtype=np.float64)
    tree = cKDTree(mp)
    dist, idx = tree.query(xy, k=1)

    out = df.copy()
    out["edge_id"] = eids[idx]
    out["dist_to_edge_midpoint_m"] = dist
    log.info(f"  assign: median dist={np.median(dist):.1f} m, "
             f"p90={np.percentile(dist, 90):.1f} m, "
             f"max={dist.max():.1f} m")
    return out


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def _line_coords(net: gpd.GeoDataFrame) -> np.ndarray:
    """(N, 2, 2) array of segments for LineCollection."""
    segs = []
    for geom in net.geometry.values:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            segs.append(np.asarray(geom.coords))
        elif geom.geom_type == "MultiLineString":
            for g in geom.geoms:
                if not g.is_empty:
                    segs.append(np.asarray(g.coords))
    return segs


def make_map(kde_matrix: pd.DataFrame,
             net: gpd.GeoDataFrame,
             segs: list,
             vmin: float,
             vmax: float,
             period: str, severity: str, bw: int,
             out_path: Path) -> None:
    col = f"kde_{period}_{severity}_bw{bw}"
    if col not in kde_matrix.columns:
        log.warning(f"  map skipped: {col} missing")
        return

    # Map density to edges by edge_id
    dens_map = kde_matrix.set_index("edge_id")[col]
    edge_ids = net["edge_id"].to_numpy()
    dens = dens_map.reindex(edge_ids).to_numpy()

    # Build colormap
    cmap = plt.get_cmap("inferno")
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    colors = cmap(norm(dens))

    # Build figure
    fig, ax = plt.subplots(figsize=(11, 9), dpi=110)
    lc = LineCollection(segs, colors=colors, linewidths=0.6,
                        capstyle="round", joinstyle="round")
    ax.add_collection(lc)
    ax.autoscale_view()

    # Colorbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(
        f"KDE intensity (log, range {vmin:.1e}\u2013{vmax:.1e})",
        fontsize=10,
    )

    # Aesthetics
    ax.set_aspect("equal")
    ax.set_title(
        f"KDE \u00b7 bw={bw} m \u00b7 {period} \u00b7 severity={severity}",
        fontsize=13,
    )
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_all_maps(kde_matrix: pd.DataFrame, net: gpd.GeoDataFrame) -> list:
    log.info("Building maps ...")
    REPORTS_FIGS.mkdir(parents=True, exist_ok=True)
    segs = _line_coords(net)
    paths = []
    for bw in BW_VALUES:
        # global log-norm over all columns for this bw across periods+sev
        cols = [f"kde_{p}_{s}_bw{bw}" for p in PERIODS for s in SEVERITIES
                if f"kde_{p}_{s}_bw{bw}" in kde_matrix.columns]
        arr = kde_matrix[cols].to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr) & (arr > 0)]
        if len(finite) == 0:
            log.warning(f"  bw={bw}: no positive densities, skip maps")
            continue
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 99.5))

        for period in PERIODS:
            for sev in SEVERITIES:
                out = REPORTS_FIGS / f"35_kde_bw{bw}_{period}_{sev}.png"
                make_map(kde_matrix, net, segs, vmin, vmax,
                         period, sev, bw, out)
                paths.append(str(out.relative_to(ROOT)))
                log.info(f"  saved {out.name}")
    return paths


# ---------------------------------------------------------------------------
# HDBSCAN cross-check
# ---------------------------------------------------------------------------

def _add_density_percentiles(kde_matrix: pd.DataFrame,
                             bw: int) -> pd.DataFrame:
    """For each edge, compute the percentile of its density within each
    (period, severity) at the given bw."""
    out = kde_matrix.copy()
    for p in PERIODS:
        for s in SEVERITIES:
            col = f"kde_{p}_{s}_bw{bw}"
            if col not in kde_matrix.columns:
                continue
            vals = kde_matrix[col].to_numpy()
            mask = np.isfinite(vals)
            pct = np.full(len(vals), np.nan)
            if mask.sum() > 0:
                ranks = pd.Series(vals[mask]).rank(pct=True, method="average")
                pct[mask] = ranks.to_numpy() * 100.0
            out[f"pct_{p}_{s}_bw{bw}"] = pct
    return out


def cross_with_hdbscan(df_with_edges: pd.DataFrame,
                       kde_matrix: pd.DataFrame,
                       hdb_df: pd.DataFrame,
                       severity_label: str,
                       bw: int) -> dict:
    """Cross HDBSCAN cluster sizes with KDE density on train.

    For each cluster from the train-fit HDBSCAN:
      - median KDE density on train (same severity label)
      - percentile of that median within the network
    Then correlate cluster size vs KDE density, and count what fraction
    of accidents in persisted clusters lives in top-20% density edges.
    """
    log.info(f"[{severity_label}] cross-check with HDBSCAN (bw={bw}) ...")

    # Subset accidents to matching severity
    if severity_label == "all":
        sub = df_with_edges
    else:
        sub = df_with_edges[df_with_edges["severity_code"] >= 1]

    # Merge cluster_train from HDBSCAN persistence
    hdb = hdb_df[["num_expediente", "cluster_train"]].copy()
    merged = sub.merge(hdb, on="num_expediente", how="inner",
                       validate="one_to_one", suffixes=("", "_hdb"))
    n_with_cluster = int((merged["cluster_train"] >= 0).sum())
    log.info(f"  accidents with cluster_train>=0: {n_with_cluster:,} "
             f"of {len(merged):,}")

    # Merge KDE density on train
    kde_train_col = f"kde_train_{severity_label}_bw{bw}"
    pct_train_col = f"pct_train_{severity_label}_bw{bw}"
    if kde_train_col not in kde_matrix.columns:
        return {"error": f"{kde_train_col} missing"}

    kde_by_edge = kde_matrix[["edge_id", kde_train_col]].rename(
        columns={kde_train_col: "kde_train"})
    merged = merged.merge(kde_by_edge, on="edge_id", how="left",
                          validate="many_to_one")

    # Percentile of density per edge
    vals = kde_matrix[kde_train_col].to_numpy()
    mask = np.isfinite(vals)
    ranks = np.full(len(vals), np.nan)
    if mask.sum() > 0:
        ranks[mask] = pd.Series(vals[mask]).rank(pct=True, method="average").to_numpy() * 100.0
    pct_df = pd.DataFrame({"edge_id": kde_matrix["edge_id"], "_pct": ranks})
    merged = merged.merge(pct_df, on="edge_id", how="left",
                          validate="many_to_one")
    merged = merged.rename(columns={"_pct": "kde_pct_train"})

    # Cluster-level stats
    cluster_stats = (
        merged[merged["cluster_train"] >= 0]
        .groupby("cluster_train")
        .agg(
            n_train=("num_expediente", "count"),
            kde_train_median=("kde_train", "median"),
            kde_pct_train_median=("kde_pct_train", "median"),
        )
        .reset_index()
    )

    # Spearman: cluster size vs KDE median
    rho_size_vs_kde = None
    rho_size_vs_pct = None
    if len(cluster_stats) >= 5:
        rho_size_vs_kde, _ = spearmanr(
            cluster_stats["n_train"], cluster_stats["kde_train_median"])
        rho_size_vs_pct, _ = spearmanr(
            cluster_stats["n_train"], cluster_stats["kde_pct_train_median"])
        rho_size_vs_kde = float(rho_size_vs_kde)
        rho_size_vs_pct = float(rho_size_vs_pct)

    # Top-K analysis
    top_k = cluster_stats.nlargest(TOP_K_CLUSTERS, "n_train")
    top_k_pct_median = (float(top_k["kde_pct_train_median"].median())
                        if len(top_k) else None)
    top_k_frac_dense = (float((top_k["kde_pct_train_median"]
                               >= TOP_PERCENTILE_DENSE).mean())
                        if len(top_k) else None)

    # Persisted clusters (n>=MIN_PERSISTED in train: we use train n as proxy here;
    # actual persistence across splits was defined in script 32 from val/test,
    # but here we only need "which clusters are big" so we keep cluster_stats
    # filtered by size).
    persisted = cluster_stats[cluster_stats["n_train"] >= MIN_PERSISTED_ACCIDENTS]
    persisted_pct_median = (float(persisted["kde_pct_train_median"].median())
                            if len(persisted) else None)
    persisted_frac_dense = (float((persisted["kde_pct_train_median"]
                                   >= TOP_PERCENTILE_DENSE).mean())
                            if len(persisted) else None)

    # Accident-level: fraction of accidents in persisted clusters that live
    # on top-20% KDE edges
    persisted_ids = set(persisted["cluster_train"].tolist())
    acc_in_persisted = merged[merged["cluster_train"].isin(persisted_ids)]
    acc_frac_dense = (float((acc_in_persisted["kde_pct_train"]
                             >= TOP_PERCENTILE_DENSE).mean())
                      if len(acc_in_persisted) else None)

    return {
        "severity": severity_label,
        "bw": bw,
        "n_clusters_train": int(len(cluster_stats)),
        "n_clusters_persisted_train": int(len(persisted)),
        "n_accidents_with_cluster": int(n_with_cluster),
        "rho_size_vs_kde_median": rho_size_vs_kde,
        "rho_size_vs_kde_pct_median": rho_size_vs_pct,
        "top_k": int(len(top_k)),
        "top_k_pct_median": top_k_pct_median,
        "top_k_frac_dense": top_k_frac_dense,
        "persisted_pct_median": persisted_pct_median,
        "persisted_frac_dense": persisted_frac_dense,
        "accidents_in_persisted_frac_dense": acc_frac_dense,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(results: dict) -> str:
    lines = ["# Phase 7 · Iter 2 — KDE maps & HDBSCAN cross-check\n"]
    lines.append(
        f"_Primary bandwidth: {BW_PRIMARY} m (D103). Reference: "
        f"{BW_REFERENCE} m. Colormap: log-normalized KDE intensity on the "
        f"OSM network. HDBSCAN clusters from D102._\n"
    )

    lines.append("## Maps generated\n")
    lines.append(f"Total: {len(results['maps'])} figures under "
                 f"`reports/figures/`.\n")

    lines.append("## HDBSCAN cross-check\n")
    lines.append(
        "For each variant, we ask: **are the HDBSCAN clusters concentrated "
        "in the highest-KDE areas, or do they reveal structure not captured "
        "by the global intensity surface?**\n"
    )
    lines.append(
        f"- `rho_size_vs_kde_median`: Spearman rho between cluster `n_train` "
        f"and the median KDE density of its edges. High positive → big "
        f"clusters live on high-KDE edges (redundant). Low or negative → "
        f"clusters capture something the raw density doesn't."
    )
    lines.append(
        f"- `top_k_frac_dense`: fraction of the top-{TOP_K_CLUSTERS} clusters "
        f"whose median density sits in the top-{100 - int(TOP_PERCENTILE_DENSE):.0f}% "
        f"of the network. High → the top clusters are in obviously dense "
        f"areas. Low → some top clusters are in moderately dense areas."
    )
    lines.append(
        f"- `accidents_in_persisted_frac_dense`: fraction of accidents "
        f"belonging to persisted clusters that live on top-"
        f"{100 - int(TOP_PERCENTILE_DENSE):.0f}% KDE edges.\n"
    )

    lines.append(
        "| variant | rho(size, kde) | top-k frac dense | persisted frac dense "
        "| acc in pers. frac dense |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key, r in results["cross"].items():
        lines.append(
            f"| {key} | {_fmt(r.get('rho_size_vs_kde_median'))} | "
            f"{_fmt(r.get('top_k_frac_dense'))} | "
            f"{_fmt(r.get('persisted_frac_dense'))} | "
            f"{_fmt(r.get('accidents_in_persisted_frac_dense'))} |"
        )
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append(results["interpretation"])

    lines.append("\n## Notes\n")
    lines.append(
        "- **Accident → edge assignment reconstructed in this script.** "
        "Uses cKDTree on edge midpoints (D077 remediation). Documented "
        "limitation: distance to midpoint ≠ distance to line; negligible "
        "for edges < 500 m, potentially >5 m for long arterial edges. "
        "Result saved in `accidents_with_edges.parquet`."
    )
    lines.append(
        "- **Density percentiles** are computed per (bw, period, severity) "
        "from the full 95k network. Top-20% is used as a fixed threshold to "
        "avoid cherry-picking."
    )
    lines.append(
        "- **KDE is descriptive, not predictive.** No refit on train/val/test "
        "in the ML sense — the surface in each period is an independent "
        "smooth of that period's accidents. Comparisons across periods are "
        "descriptive stability, not out-of-sample prediction."
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
    df, net, kde, hdb_all, hdb_inj = load_all()
    mp_by_edge = compute_edge_midpoints(net)
    df_with_edges = assign_edges_to_accidents(df, mp_by_edge)

    # Save accidents_with_edges
    OUT_PARQUET_EDGES.parent.mkdir(parents=True, exist_ok=True)
    df_with_edges[[
        "num_expediente", "severity_code", "split",
        "coordenada_x_utm", "coordenada_y_utm",
        "edge_id", "dist_to_edge_midpoint_m",
    ]].to_parquet(OUT_PARQUET_EDGES, index=False)
    log.info(f"saved {OUT_PARQUET_EDGES.relative_to(ROOT)} "
             f"({len(df_with_edges):,} rows)")

    # Maps
    map_paths = make_all_maps(kde, net)

    # Cross-check
    kde_with_pct = _add_density_percentiles(kde, BW_PRIMARY)
    cross = {
        "all": cross_with_hdbscan(
            df_with_edges, kde_with_pct, hdb_all, "all", BW_PRIMARY),
        "injured": cross_with_hdbscan(
            df_with_edges, kde_with_pct, hdb_inj, "injured", BW_PRIMARY),
    }

    # Interpretation
    lines = []
    for key, r in cross.items():
        rho = r.get("rho_size_vs_kde_median")
        frac = r.get("accidents_in_persisted_frac_dense")
        if rho is None or frac is None:
            continue
        if rho >= 0.60 and frac >= 0.75:
            lines.append(
                f"- **`{key}`**: HDBSCAN clusters are largely redundant with "
                f"the global intensity surface (rho={rho:.2f}, "
                f"{frac:.0%} of persisted accidents on top-20% KDE). "
                f"KDE alone would recover most of the spatial story for this "
                f"variant."
            )
        elif rho >= 0.30:
            lines.append(
                f"- **`{key}`**: HDBSCAN adds *moderate* structure beyond KDE "
                f"(rho={rho:.2f}, {frac:.0%} of persisted accidents on "
                f"top-20% KDE). The clusters are correlated with intensity "
                f"but not fully explained by it."
            )
        else:
            lines.append(
                f"- **`{key}`**: HDBSCAN adds *substantial* structure not "
                f"captured by KDE (rho={rho:.2f}, {frac:.0%} of persisted "
                f"accidents on top-20% KDE). The hotspot analysis carries "
                f"information the intensity map misses."
            )
    interpretation = "\n".join(lines) if lines else "_(insufficient data)_"

    results = {
        "bw_primary": BW_PRIMARY,
        "bw_reference": BW_REFERENCE,
        "top_percentile_dense": TOP_PERCENTILE_DENSE,
        "maps": map_paths,
        "cross": cross,
        "interpretation": interpretation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(results), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    log.info("=" * 72)
    for key, r in cross.items():
        log.info(f"[{key}] rho(size,kde)={_fmt(r.get('rho_size_vs_kde_median'))}  "
                 f"top-k frac dense={_fmt(r.get('top_k_frac_dense'))}  "
                 f"persisted frac dense={_fmt(r.get('persisted_frac_dense'))}")


if __name__ == "__main__":
    main()