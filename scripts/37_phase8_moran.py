"""
Script 37 — Phase 8 iter 1: Local Moran's I (LISA) on Madrid accidents.

v8 — Phase 4 fix integration:
  - Replaced the local `_reconstruct_geom` (bug: hardcoded crs=EPSG:4326,
    exploded on the UTM callejero) with `madrid_safety.io.load_geometry_parquet`
    (CRS auto-detected from bounds, handles GeoParquet natively).
  - Replaced the local `repair_geometries` with `madrid_safety.io.repair_geometries`
    (uses shapely.make_valid, keeps polygonal component of GeometryCollections).
  - The callejero parquet was repaired and saved as GeoParquet in 17c.
  - The OSM network parquet was saved as GeoParquet in 18b.
  - Both are read directly by geopandas now; no reprojection gymnastics.

Corrections retained from v7:
  - Pointy-top hexagons (angles 30°, 90°, ..., 330°).
  - Island-dropping + W rebuild (no w_subset).
  - Relative variance check (var / mean^2).
  - Split try/except for Moran (global) and Moran_Local (local).
  - Fallback permutations 999 → 199 if needed.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon

try:
    import libpysal
    import esda
    from esda.moran import Moran, Moran_Local
    try:
        from esda.eb import EmpiricalBayes
        HAS_EB = True
    except Exception:  # noqa: BLE001
        HAS_EB = False
except ImportError as e:
    print(f"ERROR: required package not installed: {e}")
    print("Install: conda install -c conda-forge libpysal esda statsmodels")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
ADMIN_DIR = DATA_PROCESSED / "admin"
REPORTS = ROOT / "reports"
REPORTS_TABLES = REPORTS / "tables"

ACCIDENTS_PATH = DATA_PROCESSED / "accidents_clean.parquet"
NETWORK_KDE_PATH = DATA_PROCESSED / "network_kde_final.parquet"
NETWORK_OSM_PATH = DATA_PROCESSED / "network_osm.parquet"
CALLEJERO_PATH = DATA_PROCESSED / "network_callejero.parquet"

TARGET_CRS = "EPSG:25830"
GEO_URLS = {
    "distritos": "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.zip",
    "barrios":   "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.zip",
    "termino":   "https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Termino_Municipal/Termino_municipal.zip",
}

HEX_SIZES = {"hex200m": 200, "hex250m": 250, "hex300m": 300}
KDE_BANDWIDTH_COL = "bw260"
PERMUTATIONS = 999
PERMUTATIONS_FALLBACK = 199
FDR_ALPHA = 0.05
RANDOM_SEED = 42
MIN_VALID_UNITS = 20
REL_VAR_EPS = 1e-10

PROTECTED_INPUTS = {
    DATA_PROCESSED / "accidents_clean.parquet",
    DATA_PROCESSED / "network_kde_final.parquet",
    DATA_PROCESSED / "network_osm.parquet",
    DATA_PROCESSED / "network_callejero.parquet",
    DATA_PROCESSED / "splits.parquet",
}

log = logging.getLogger("phase8_moran")


# ---------------------------------------------------------------------------
# Admin geometries
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log.info(f"  downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    return dest


def _extract_and_load(zip_path: Path, key: str) -> gpd.GeoDataFrame:
    extract_dir = ADMIN_DIR / key
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*.shp")) or list(extract_dir.rglob("*.geojson"))
    if not candidates:
        raise RuntimeError(f"no SHP in {zip_path}")
    gdf = gpd.read_file(candidates[0])
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    if gdf.crs.to_epsg() != 25830:
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def load_admin_geometries() -> dict:
    log.info("Loading administrative geometries ...")
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in GEO_URLS.items():
        zip_path = ADMIN_DIR / f"{key}.zip"
        _download(url, zip_path)
        gdf = _extract_and_load(zip_path, key)
        log.info(f"  {key}: {len(gdf)} rows")
        out[key] = gdf
        gdf.to_parquet(ADMIN_DIR / f"{key}.parquet", index=False)
    return out


def pick_id_col(gdf: gpd.GeoDataFrame, label: str) -> str:
    nunique = {}
    for c in gdf.columns:
        if c == "geometry":
            continue
        try:
            nunique[c] = int(gdf[c].nunique())
        except Exception:  # noqa: BLE001
            continue
    if not nunique:
        raise RuntimeError(f"{label}: no usable id column")
    priority = ["top_id", "cod_barrio", "cod_bar", "cod_disbar",
                "cod_distrito", "cod_dis", "barrio", "distrito",
                "nombre", "name"]
    for p in priority:
        for c, n in nunique.items():
            if p in c.lower() and 5 <= n <= 5000:
                log.info(f"  [{label}] id column: {c} ({n} unique)")
                return c
    candidates = {c: n for c, n in nunique.items() if 5 <= n <= 5000}
    if not candidates:
        candidates = nunique
    best = max(candidates, key=candidates.get)
    log.info(f"  [{label}] id column (fallback): {best} "
             f"({candidates[best]} unique)")
    return best


# ---------------------------------------------------------------------------
# Hex grid
# ---------------------------------------------------------------------------

def make_hex_grid(bbox, side_length: float, crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    """Pointy-top hexagonal grid with side `side_length` (circumradius)."""
    minx, miny, maxx, maxy = bbox
    dx = np.sqrt(3.0) * side_length
    dy = 1.5 * side_length
    xs = np.arange(minx - dx, maxx + dx, dx)
    ys = np.arange(miny - dy, maxy + dy, dy)
    # Pointy-top: vertices at 30°, 90°, ..., 330°
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    polys = []
    for row, y in enumerate(ys):
        x_off = 0.0 if row % 2 == 0 else dx / 2
        for x in xs:
            cx, cy = x + x_off, y
            pts = [(round(cx + side_length * cos_a[k], 3),
                    round(cy + side_length * sin_a[k], 3)) for k in range(6)]
            polys.append(Polygon(pts))
    return gpd.GeoDataFrame(
        {"hex_id": np.arange(len(polys), dtype=np.int64)},
        geometry=polys, crs=crs,
    )


def build_hexes(termino: gpd.GeoDataFrame) -> dict:
    log.info("Building hexagonal grids ...")
    bounds = termino.total_bounds
    out = {}
    for name, side in HEX_SIZES.items():
        gdf = make_hex_grid(bounds, side, crs=TARGET_CRS)
        gdf = gpd.sjoin(gdf, termino[["geometry"]], how="inner", predicate="intersects")
        gdf = gdf.drop(columns=["index_right"]).reset_index(drop=True)

        sample = gdf.iloc[:min(50, len(gdf))].copy()
        W_test, _ = build_W_raw(sample, use_knn=False)
        n_edges = sum(len(W_test.neighbors[i]) for i in W_test.neighbors) // 2 if W_test else 0

        log.info(f"  {name}: {len(gdf)} hexes  "
                 f"(sample W: {n_edges} edges between first 50)")
        out[name] = gdf
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def accidents_gdf(acc: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        acc, geometry=gpd.points_from_xy(acc["coordenada_x_utm"],
                                         acc["coordenada_y_utm"]),
        crs=TARGET_CRS,
    )


def load_network() -> gpd.GeoDataFrame:
    """Load OSM drive network via the centralized io loader."""
    net = ms_io.load_geometry_parquet(NETWORK_OSM_PATH,
                                      expected_crs="EPSG:4326")
    if net.crs.to_epsg() != 25830:
        net = net.to_crs(TARGET_CRS)
    return net.reset_index(drop=True)


def check_kde_alignment(kde: pd.DataFrame, net: gpd.GeoDataFrame) -> None:
    log.info(f"  kde_matrix: {len(kde):,} rows | net: {len(net):,} edges")
    if len(kde) != len(net):
        raise RuntimeError(f"KDE ({len(kde)}) and net ({len(net)}) sizes differ")
    if not np.array_equal(np.sort(kde["edge_id"].to_numpy()), np.arange(len(net))):
        raise RuntimeError("edge_id not a permutation of 0..N-1")
    log.info("  edge_id alignment: OK")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_counts(unit_gdf, acc_pts, unit_id_col) -> pd.DataFrame:
    joined = gpd.sjoin(acc_pts, unit_gdf[[unit_id_col, "geometry"]],
                       how="inner", predicate="within")
    g = joined.groupby(unit_id_col)
    out = pd.DataFrame({
        "counts_all":     g.size().astype(int),
        "counts_injured": g["severity_code"].apply(lambda s: int((s >= 1).sum())),
    }).reset_index()
    full = pd.DataFrame({unit_id_col: unit_gdf[unit_id_col].values})
    full = full.merge(out, on=unit_id_col, how="left").fillna(0)
    full["counts_all"] = full["counts_all"].astype(int)
    full["counts_injured"] = full["counts_injured"].astype(int)
    return full


def compute_unit_kde(unit_gdf, net, kde_matrix, unit_id_col,
                     col_all, col_inj) -> pd.DataFrame:
    if len(unit_gdf) == 0:
        return pd.DataFrame({unit_id_col: [], "kde_all_mean": [],
                             "kde_inj_mean": [], "n_edges": []})

    kde_map = kde_matrix.set_index("edge_id")
    net2 = net.copy()
    net2["edge_id"] = np.arange(len(net2), dtype=np.int64)
    net2["kde_all"] = kde_map[col_all].reindex(net2["edge_id"]).to_numpy()
    net2["kde_inj"] = kde_map[col_inj].reindex(net2["edge_id"]).to_numpy()
    net2["len_m"] = net2.geometry.length

    joined = gpd.sjoin(net2[["geometry", "kde_all", "kde_inj", "len_m"]],
                       unit_gdf[[unit_id_col, "geometry"]],
                       how="inner", predicate="intersects")
    if unit_id_col not in joined.columns:
        raise RuntimeError(f"sjoin lost {unit_id_col}")

    def _wmean(sub, c):
        w = sub["len_m"].to_numpy(dtype=np.float64)
        v = sub[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(v) & (w > 0)
        if not ok.any():
            return np.nan
        return float((v[ok] * w[ok]).sum() / w[ok].sum())

    grp = joined.groupby(unit_id_col)
    out = pd.DataFrame({
        "kde_all_mean": grp.apply(lambda s: _wmean(s, "kde_all")),
        "kde_inj_mean": grp.apply(lambda s: _wmean(s, "kde_inj")),
        "n_edges":      grp.size().astype(int),
    }).reset_index()

    full = pd.DataFrame({unit_id_col: unit_gdf[unit_id_col].values})
    full = full.merge(out, on=unit_id_col, how="left")
    return full


# ---------------------------------------------------------------------------
# EB
# ---------------------------------------------------------------------------

def eb_injury_rate(counts: pd.DataFrame) -> pd.Series:
    y = counts["counts_injured"].to_numpy(dtype=np.float64)
    n = counts["counts_all"].to_numpy(dtype=np.float64)
    global_rate = float(y.sum() / max(n.sum(), 1.0))
    if HAS_EB:
        try:
            eb = esda.eb.EmpiricalBayes(y, n)
            r = np.asarray(eb.r if hasattr(eb, "r") else eb, dtype=np.float64)
            r[n == 0] = global_rate
            return pd.Series(r, index=counts.index, name="injury_rate_eb")
        except Exception as e:  # noqa: BLE001
            log.warning(f"  EB failed ({type(e).__name__}); manual fallback")
    alpha = max(global_rate, 1e-6)
    smoothed = (y + alpha) / (n + 1.0)
    return pd.Series(smoothed, index=counts.index, name="injury_rate_eb")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def build_W_raw(sub_gdf, use_knn):
    sub = sub_gdf.reset_index(drop=True)
    if use_knn:
        try:
            c = sub.geometry.centroid
            coords = np.column_stack([c.x.to_numpy(), c.y.to_numpy()])
            ok = np.isfinite(coords).all(axis=1)
            if ok.sum() < MIN_VALID_UNITS:
                return None, f"knn: only {ok.sum()} finite coords"
            coords = coords[ok]
            k = min(6, len(coords) - 1)
            W = libpysal.weights.KNN.from_array(coords, k=k)
            return W, f"knn_k{k}"
        except Exception as e:  # noqa: BLE001
            return None, f"knn_failed: {type(e).__name__}"
    try:
        W = libpysal.weights.Queen.from_dataframe(
            sub, use_index=False, silence_warnings=True)
        return W, "queen"
    except Exception as e_q:  # noqa: BLE001
        log.info(f"    queen failed: {type(e_q).__name__}")
    try:
        W = libpysal.weights.Rook.from_dataframe(
            sub, use_index=False, silence_warnings=True)
        return W, "rook"
    except Exception as e_r:  # noqa: BLE001
        log.info(f"    rook failed: {type(e_r).__name__}")
    try:
        c = sub.geometry.centroid
        coords = np.column_stack([c.x.to_numpy(), c.y.to_numpy()])
        ok = np.isfinite(coords).all(axis=1)
        coords = coords[ok]
        if len(coords) < MIN_VALID_UNITS:
            return None, "knn: too few finite coords"
        W = libpysal.weights.KNN.from_array(coords, k=6)
        return W, "knn_k6"
    except Exception as e_k:  # noqa: BLE001
        return None, f"all_failed: {type(e_k).__name__}"


def build_W_no_islands(sub_gdf, use_knn, max_iter=5):
    current_gdf = sub_gdf.reset_index(drop=True).copy()
    original_positions = np.arange(len(current_gdf))
    for iteration in range(max_iter):
        W, kind = build_W_raw(current_gdf, use_knn)
        if W is None:
            return None, f"failed_at_iter{iteration}: {kind}", np.array([], dtype=int)
        islands = list(W.islands)
        if len(islands) == 0:
            return W, kind, original_positions
        log.info(f"    iter{iteration}: dropping {len(islands)} islands")
        keep = np.array([i for i in range(len(current_gdf)) if i not in islands],
                        dtype=int)
        if len(keep) < MIN_VALID_UNITS:
            return None, f"too_few_after_islands ({len(keep)})", np.array([], dtype=int)
        current_gdf = current_gdf.iloc[keep].reset_index(drop=True)
        original_positions = original_positions[keep]
    return None, "max_iter_reached", np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Moran
# ---------------------------------------------------------------------------

def _run_moran_local(v, W, perms):
    try:
        return Moran_Local(v, W, permutations=perms, seed=RANDOM_SEED), perms, None
    except Exception as e:  # noqa: BLE001
        log.info(f"    Moran_Local({perms}) failed: {type(e).__name__}: {e}")
        if perms > PERMUTATIONS_FALLBACK:
            try:
                return (Moran_Local(v, W, permutations=PERMUTATIONS_FALLBACK,
                                    seed=RANDOM_SEED),
                        PERMUTATIONS_FALLBACK, None)
            except Exception as e2:  # noqa: BLE001
                return None, 0, f"{type(e2).__name__}: {e2}"
        return None, 0, f"{type(e).__name__}: {e}"


def moran_pair(values, unit_gdf, use_knn, label):
    n_orig = len(unit_gdf)
    valid = np.isfinite(values)
    if valid.sum() < MIN_VALID_UNITS:
        return {"error": f"only {valid.sum()} valid (<{MIN_VALID_UNITS})"}

    v = values[valid].astype(np.float64)
    mean_abs = float(np.abs(v).mean()) + 1e-30
    rel_var = float(np.var(v)) / (mean_abs ** 2)
    if rel_var < REL_VAR_EPS:
        return {"error": f"zero relative variance ({rel_var:.2e}, n={len(v)})"}

    sub_gdf = unit_gdf[valid].reset_index(drop=True).copy()

    W, kind, kept_local = build_W_no_islands(sub_gdf, use_knn)
    if W is None:
        return {"error": f"W build failed: {kind}"}

    if len(kept_local) < len(sub_gdf):
        v = v[kept_local]
        sub_gdf = sub_gdf.iloc[kept_local].reset_index(drop=True)
        log.info(f"    after island drop: n={len(v)}")

    if len(v) < MIN_VALID_UNITS:
        return {"error": f"too few after island drop ({len(v)})"}

    mean_abs = float(np.abs(v).mean()) + 1e-30
    rel_var = float(np.var(v)) / (mean_abs ** 2)
    if rel_var < REL_VAR_EPS:
        return {"error": f"zero variance after drop ({rel_var:.2e})"}

    try:
        g = Moran(v, W, permutations=PERMUTATIONS)
        I_val = float(g.I)
    except Exception as e:  # noqa: BLE001
        return {"error": f"Moran global failed: {type(e).__name__}: {e}"}

    l, perms_used, err = _run_moran_local(v, W, PERMUTATIONS)
    if l is None:
        return {"error": f"Moran_Local failed: {err}"}

    p_sim = np.asarray(l.p_sim, dtype=np.float64)
    if p_sim.shape != (len(v),):
        return {"error": f"p_sim shape {p_sim.shape} != ({len(v)},)"}

    try:
        from statsmodels.stats.multitest import fdrcorrection
        _, p_adj = fdrcorrection(p_sim, alpha=FDR_ALPHA, method="indep")
    except Exception:  # noqa: BLE001
        m = len(p_sim)
        order = np.argsort(p_sim)
        ranked = p_sim[order] * m / np.arange(1, m + 1)
        p_adj = np.empty_like(p_sim)
        p_adj[order] = np.minimum.accumulate(ranked[::-1])[::-1]

    sig = p_adj < FDR_ALPHA
    q = l.q.astype(int)
    cluster = np.array(["NS"] * len(v), dtype=object)
    cluster[(q == 1) & sig] = "HH"
    cluster[(q == 2) & sig] = "LH"
    cluster[(q == 3) & sig] = "LL"
    cluster[(q == 4) & sig] = "HL"

    idx_valid = np.where(valid)[0]
    idx_orig = idx_valid[kept_local]
    cluster_full = np.array(["NA"] * n_orig, dtype=object)
    cluster_full[idx_orig] = cluster
    p_adj_full = np.full(n_orig, np.nan, dtype=np.float64)
    p_adj_full[idx_orig] = p_adj

    return {
        "label": label, "n_units": int(len(v)), "weights_kind": kind,
        "permutations_used": int(perms_used),
        "moran_I": I_val,
        "moran_p_norm": float(g.p_norm) if hasattr(g, "p_norm") else None,
        "moran_z_norm": float(g.z_norm) if hasattr(g, "z_norm") else None,
        "n_HH": int((cluster == "HH").sum()),
        "n_LL": int((cluster == "LL").sum()),
        "n_HL": int((cluster == "HL").sum()),
        "n_LH": int((cluster == "LH").sum()),
        "n_NS": int((cluster == "NS").sum()),
        "pct_HH": float((cluster == "HH").mean()),
        "pct_LL": float((cluster == "LL").mean()),
        "pct_HL": float((cluster == "HL").mean()),
        "pct_LH": float((cluster == "LH").mean()),
        "pct_NS": float((cluster == "NS").mean()),
        "_cluster": cluster_full,
        "_p_adj": p_adj_full,
    }


# ---------------------------------------------------------------------------
# Per-unit
# ---------------------------------------------------------------------------

def run_one_unit(unit_name, unit_gdf, unit_id_col, acc_pts, net, kde_matrix,
                 use_knn=False):
    log.info(f"=== Unit: {unit_name} ({len(unit_gdf)} rows) ===")
    n_before = len(unit_gdf)
    unit_gdf = unit_gdf.dropna(subset=[unit_id_col]).reset_index(drop=True)
    unit_gdf = unit_gdf.drop_duplicates(subset=[unit_id_col]).reset_index(drop=True)

    if not unit_gdf.geometry.is_valid.all():
        n_invalid = int((~unit_gdf.geometry.is_valid).sum())
        log.info(f"  repairing {n_invalid} invalid geometries")
        unit_gdf = ms_io.repair_geometries(unit_gdf, unit_name)

    if len(unit_gdf) != n_before:
        log.info(f"  cleaned: {n_before} -> {len(unit_gdf)}")

    if len(unit_gdf) < MIN_VALID_UNITS:
        log.warning(f"  too few valid units ({len(unit_gdf)}). Skipping.")
        return {"error": f"too few valid units ({len(unit_gdf)})",
                "weights_kind": "none"}

    counts = aggregate_counts(unit_gdf, acc_pts, unit_id_col)
    counts = counts.merge(unit_gdf[[unit_id_col]], on=unit_id_col, how="left")
    counts["injury_rate_eb"] = eb_injury_rate(counts).values

    kde_by_unit = compute_unit_kde(
        unit_gdf, net, kde_matrix, unit_id_col,
        col_all=f"kde_train_all_{KDE_BANDWIDTH_COL}",
        col_inj=f"kde_train_injured_{KDE_BANDWIDTH_COL}",
    )
    merged = unit_gdf.merge(counts, on=unit_id_col, how="left")
    merged = merged.merge(kde_by_unit, on=unit_id_col, how="left")

    results = {"unit": unit_name, "n_units": int(len(unit_gdf)),
               "weights_kind": "pending", "variables": {}}
    per_var_clusters = {}

    for var, colname in [
        ("counts_all",     "counts_all"),
        ("counts_injured", "counts_injured"),
        ("kde_all",        "kde_all_mean"),
        ("injury_rate_eb", "injury_rate_eb"),
    ]:
        if colname not in merged.columns:
            log.info(f"  skipping {var} (missing)")
            continue
        values = merged[colname].to_numpy(dtype=np.float64)
        r = moran_pair(values, unit_gdf, use_knn, f"{unit_name}.{var}")
        if "error" in r:
            log.warning(f"  [{var}] {r['error']}")
            results["variables"][var] = {"error": r["error"]}
            continue
        log.info(f"  [{var}] I={r['moran_I']:.4f}  n={r['n_units']}  "
                 f"HH={r['pct_HH']:.2%}  LL={r['pct_LL']:.2%}  "
                 f"W={r['weights_kind']}  perms={r['permutations_used']}")
        per_var_clusters[var] = {
            "cluster": r["_cluster"],
            "p_adj": r["_p_adj"],
        }
        results["variables"][var] = {k: v for k, v in r.items()
                                     if not k.startswith("_")}

    out_df = unit_gdf[[unit_id_col, "geometry"]].copy()
    for var, data in per_var_clusters.items():
        out_df[f"{var}_cluster"] = data["cluster"]
        out_df[f"{var}_p_adj"] = data["p_adj"]

    return {"summary": results, "output_df": out_df}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def render_md(results):
    lines = ["# Phase 8 · Iter 1 — Local Moran's I (LISA)\n"]
    lines.append(
        f"_Weights: Queen/Rook/KNN cascade (islands dropped iteratively). "
        f"{PERMUTATIONS} permutations (fallback {PERMUTATIONS_FALLBACK}). "
        f"FDR α={FDR_ALPHA}. Relative variance threshold {REL_VAR_EPS}._\n"
    )
    for unit_key in ("hex250m", "barrios", "distritos", "callejero"):
        if unit_key not in results:
            continue
        r = results[unit_key]
        if "error" in r:
            lines.append(f"## `{unit_key}` — ERROR: {r['error']}\n")
            continue
        lines.append(f"## `{unit_key}` — {r['n_units']} units\n")
        lines.append("| variable | Moran I | n | HH% | LL% | HL% | LH% | NS% | W |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for var, v in r["variables"].items():
            if "error" in v:
                lines.append(f"| `{var}` | error: {v['error']} | | | | | | | |")
                continue
            lines.append(
                f"| `{var}` | {_fmt(v.get('moran_I'), 4)} | "
                f"{v.get('n_units', 'n/a')} | "
                f"{_fmt(v.get('pct_HH'), 4)} | "
                f"{_fmt(v.get('pct_LL'), 4)} | "
                f"{_fmt(v.get('pct_HL'), 4)} | "
                f"{_fmt(v.get('pct_LH'), 4)} | "
                f"{_fmt(v.get('pct_NS'), 4)} | "
                f"{v.get('weights_kind', '?')} |"
            )
        lines.append("")

    if "sensitivity" in results:
        lines.append("## Sensitivity (hex 200/300 m, counts_all)\n")
        lines.append("| hex size | n_units | Moran I | HH% |")
        lines.append("| --- | ---: | ---: | ---: |")
        for s in results["sensitivity"]:
            lines.append(
                f"| {s['hex_name']} | {s.get('n_units', 'n/a')} | "
                f"{_fmt(s.get('moran_I'), 4)} | {_fmt(s.get('pct_HH'), 4)} |"
            )
        lines.append("")

    lines.append("## Notes\n")
    lines.append("- **Islands are dropped from the subset before Moran.**")
    lines.append("- **Relative variance check**: `var / mean^2`, scale-invariant.")
    lines.append("- **Moran on raw counts reflects volume, not risk (rule 9).**")
    lines.append("- **Callejero** repaired via shapely.make_valid + buffer(0) in 17c.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    outputs = {
        ADMIN_DIR / "distritos.parquet",
        ADMIN_DIR / "barrios.parquet",
        ADMIN_DIR / "termino.parquet",
        *[DATA_PROCESSED / f"moran_{u}.parquet"
          for u in ("hex250m", "barrios", "distritos", "callejero")],
    }
    if outputs & PROTECTED_INPUTS:
        raise RuntimeError(f"collision: {outputs & PROTECTED_INPUTS}")

    log.info("Loading accidents + network + kde ...")
    acc = pd.read_parquet(ACCIDENTS_PATH)[
        ["num_expediente", "severity_code",
         "coordenada_x_utm", "coordenada_y_utm"]]
    acc = acc.dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()
    log.info(f"  accidents: {len(acc):,}")
    acc_pts = accidents_gdf(acc)
    net = load_network()
    kde = pd.read_parquet(NETWORK_KDE_PATH)
    check_kde_alignment(kde, net)

    admin = load_admin_geometries()
    hexes = build_hexes(admin["termino"])

    log.info("Loading callejero ...")
    cal = ms_io.load_geometry_parquet(CALLEJERO_PATH,
                                      expected_crs="EPSG:25830")
    if cal.crs.to_epsg() != 25830:
        cal = cal.to_crs(TARGET_CRS)
    cal = cal.reset_index(drop=True)
    if not cal.geometry.is_valid.all():
        n_invalid = int((~cal.geometry.is_valid).sum())
        log.warning(f"  callejero: {n_invalid} invalid geoms — re-repairing")
        cal = ms_io.repair_geometries(cal, "callejero")
    if "TOP_ID" not in cal.columns:
        cal["TOP_ID"] = np.arange(len(cal), dtype=np.int64)
    log.info(f"  callejero: {len(cal):,} valid polygons")

    barrios_id = pick_id_col(admin["barrios"], "barrios")
    distritos_id = pick_id_col(admin["distritos"], "distritos")
    log.info(f"  chosen: barrios_id={barrios_id}  distritos_id={distritos_id}")

    results = {}

    log.info("\n--- MAIN UNITS ---")
    for unit_name, gdf, idcol, use_knn in [
        ("hex250m",   hexes["hex250m"],   "hex_id",     False),
        ("barrios",   admin["barrios"],   barrios_id,   False),
        ("distritos", admin["distritos"], distritos_id, False),
        ("callejero", cal,                "TOP_ID",     True),
    ]:
        try:
            r = run_one_unit(unit_name, gdf, idcol, acc_pts, net, kde, use_knn)
            results[unit_name] = r.get("summary", r)
            if "output_df" in r:
                outpath = DATA_PROCESSED / f"moran_{unit_name}.parquet"
                r["output_df"].to_parquet(outpath, index=False)
                log.info(f"  saved {outpath.relative_to(ROOT)} "
                         f"({len(r['output_df'])} rows)")
        except Exception as e:  # noqa: BLE001
            log.error(f"[{unit_name}] failed: {type(e).__name__}: {e}")
            import traceback
            log.error(traceback.format_exc())
            results[unit_name] = {"error": f"{type(e).__name__}: {e}"}

    log.info("\n--- SENSITIVITY ---")
    sensitivity = []
    for hex_name in ("hex200m", "hex300m"):
        try:
            r = run_one_unit(hex_name, hexes[hex_name], "hex_id",
                             acc_pts, net, kde, use_knn=False)
            summary = r.get("summary", r)
            ca = summary.get("variables", {}).get("counts_all", {})
            sensitivity.append({
                "hex_name": hex_name, "n_units": summary.get("n_units"),
                "moran_I": ca.get("moran_I"), "pct_HH": ca.get("pct_HH"),
            })
        except Exception as e:  # noqa: BLE001
            log.error(f"[{hex_name}] failed: {type(e).__name__}: {e}")
            sensitivity.append({"hex_name": hex_name, "error": str(e)})
    results["sensitivity"] = sensitivity

    OUT_JSON = REPORTS_TABLES / "37_moran.json"
    OUT_MD = REPORTS / "37_moran.md"
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")
    OUT_MD.write_text(render_md(results), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    log.info("=" * 72)
    for k, v in results.items():
        if k == "sensitivity":
            continue
        if "error" in v:
            log.info(f"[{k}] ERROR: {v['error']}")
            continue
        log.info(f"[{k}] n={v['n_units']}")
        for var, rv in v["variables"].items():
            if "error" in rv:
                log.info(f"    [{var}] ERROR: {rv['error']}")
            else:
                log.info(f"    [{var}] I={rv['moran_I']:.4f} "
                         f"n={rv['n_units']} W={rv['weights_kind']}")


if __name__ == "__main__":
    main()