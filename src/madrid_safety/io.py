"""
I/O utilities for geometry parquets.

Centralizes the defensive loading and repair logic that was being
copy-pasted across scripts with subtle bugs (notably the EPSG:4326
hardcode that broke `network_callejero.parquet` in Phase 8).

Functions
---------
detect_crs_from_bounds(geoms) -> str
    Heuristic CRS detection. Returns "EPSG:25830" if coordinates look
    projected (|x|>180 or |y|>90), else "EPSG:4326".

load_geometry_parquet(path, expected_crs=None, geom_col=None,
                      wkt_col=None) -> GeoDataFrame
    Loads a parquet that may or may not carry GeoParquet metadata.
    Falls back to WKB / WKT / shapely reconstruction with CRS detection.

repair_geometries(gdf, label="") -> GeoDataFrame
    Repairs invalid geometries via shapely.make_valid (fallback buffer(0)).
    Drops unrecoverable rows. Logs counts.

save_geometry_parquet(gdf, path) -> None
    Writes a GeoParquet that ALSO preserves a `geometry_wkt` string
    column for backwards compatibility with older readers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GEOMETRY_COL_CANDIDATES = ("geometry", "geom", "geom_wkt",
                            "geometry_wkb", "geometry_wkt", "wkt")
MADRID_UTM_CRS = "EPSG:25830"
WGS84_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# CRS detection
# ---------------------------------------------------------------------------

def detect_crs_from_bounds(geoms) -> str:
    """Heuristic CRS detection from geometry bounds.

    Madrid UTM 30N coordinates are around x≈440000, y≈4475000.
    WGS84 (lon, lat) is around x≈-3.7, y≈40.4.

    Any |coord| > 180 is definitively not WGS84 longitude.
    Any |coord| > 90  is definitively not WGS84 latitude.
    """
    for g in geoms:
        if g is None or g.is_empty:
            continue
        try:
            b = g.bounds  # (minx, miny, maxx, maxy)
        except Exception:
            continue
        if not np.all(np.isfinite(b)):
            # Infinite bounds → likely corrupted, cannot infer
            continue
        if abs(b[0]) > 180 or abs(b[1]) > 90:
            return MADRID_UTM_CRS
        return WGS84_CRS
    return WGS84_CRS  # empty / all-corrupted → default


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_geometry_parquet(path: Path,
                          expected_crs: str | None = None,
                          geom_col: str | None = None,
                          wkt_col: str | None = None) -> gpd.GeoDataFrame:
    """Robust loader for geometry parquets.

    Strategy:
      1. Try gpd.read_parquet() — works if GeoParquet metadata exists.
      2. Fall back to pd.read_parquet() + reconstruct from WKB/WKT/shapely.
         Detect CRS from bounds if not given via `expected_crs`.

    The reconstructed geometry is stored in a column named 'geometry'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    # Fast path: GeoParquet with metadata
    try:
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(expected_crs or
                              detect_crs_from_bounds(gdf.geometry))
        log.info(f"  loaded {path.name} via geopandas "
                 f"({len(gdf):,} rows, CRS={gdf.crs.to_epsg()})")
        return gdf
    except (ValueError, KeyError) as e:
        log.info(f"  {path.name}: geopandas read failed ({type(e).__name__}); "
                 f"falling back to pandas")

    # Slow path: pandas + reconstruction
    df = pd.read_parquet(path)

    # Find geometry column
    if geom_col is None:
        for c in GEOMETRY_COL_CANDIDATES:
            if c in df.columns:
                geom_col = c
                break
    if geom_col is None:
        raise RuntimeError(
            f"no geometry column found in {path}. "
            f"Columns: {list(df.columns)}"
        )
    if wkt_col is None and geom_col != "geometry_wkt" and "geometry_wkt" in df.columns:
        # Prefer geometry_wkt if present (string form is safer to parse)
        wkt_col = "geometry_wkt"

    target_col = wkt_col or geom_col
    sample = df[target_col].dropna().iloc[0] if df[target_col].notna().any() else None

    if sample is None:
        raise RuntimeError(f"{target_col} is entirely null in {path}")

    if isinstance(sample, (bytes, bytearray)):
        from shapely import wkb
        geoms = df[target_col].apply(
            lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else None)
        log.info(f"  {path.name}: parsed '{target_col}' as WKB")
    elif isinstance(sample, str):
        from shapely import wkt
        geoms = df[target_col].apply(
            lambda s: wkt.loads(s) if isinstance(s, str) and s else None)
        log.info(f"  {path.name}: parsed '{target_col}' as WKT")
    else:
        log.info(f"  {path.name}: '{target_col}' holds shapely objects")
        geoms = df[target_col]

    crs = expected_crs or detect_crs_from_bounds(geoms)
    log.info(f"  {path.name}: CRS detected = {crs}")

    # Drop the geometry source column from attributes (we recreate it)
    other_cols = [c for c in df.columns if c != target_col]
    gdf = gpd.GeoDataFrame(df[other_cols], geometry=geoms, crs=crs)
    return gdf


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _polygonal_only(g):
    """Extract polygonal component from a geometry (make_valid may return
    GeometryCollection mixing polygons, lines and points)."""
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        from shapely.ops import unary_union
        polys = [sub for sub in g.geoms
                 if sub.geom_type in ("Polygon", "MultiPolygon")
                 and not sub.is_empty]
        if not polys:
            return None
        return unary_union(polys)
    return None


def repair_geometries(gdf: gpd.GeoDataFrame, label: str = "") -> gpd.GeoDataFrame:
    """Repair invalid geometries; drop unrecoverable; log counts.

    Uses shapely.make_valid as primary, buffer(0) as per-row fallback.
    If the repaired geometry is a GeometryCollection, keeps only the
    polygonal component (for point-in-polygon sjoin correctness).
    """
    n_in = len(gdf)
    if n_in == 0:
        return gdf

    try:
        from shapely import make_valid  # shapely >= 2.0
        has_make_valid = True
    except ImportError:
        has_make_valid = False

    n_valid_original = int(gdf.geometry.is_valid.sum())
    n_invalid = n_in - n_valid_original

    if n_invalid == 0:
        log.info(f"  [{label}] all {n_in:,} geometries valid, no repair needed")
        return gdf

    repaired = []
    n_failed = 0
    for g in gdf.geometry:
        if g is None or g.is_empty:
            repaired.append(None)
            continue
        if g.is_valid:
            repaired.append(g)
            continue
        fixed = None
        if has_make_valid:
            try:
                fixed = make_valid(g)
            except Exception:
                fixed = None
        if fixed is None:
            try:
                fixed = g.buffer(0)
            except Exception:
                fixed = None
        # For points (which may come from Infinite coords), skip
        if fixed is not None:
            fixed = _polygonal_only(fixed) if g.geom_type in ("Polygon", "MultiPolygon") else fixed
            if fixed is not None and fixed.is_empty:
                fixed = None
        if fixed is None:
            n_failed += 1
        repaired.append(fixed)

    out = gpd.GeoDataFrame(
        gdf.drop(columns=[gdf.geometry.name]),
        geometry=repaired,
        crs=gdf.crs,
    )
    out = out[out.geometry.notna()].reset_index(drop=True)
    n_dropped = n_in - len(out)

    log.info(f"  [{label}] repair: {n_in:,} -> {len(out):,} "
             f"(invalid={n_invalid:,}, failed={n_failed:,}, "
             f"dropped_total={n_dropped:,})")

    if len(out) == 0 and n_in > 0:
        samples = [str(g)[:80] for g in gdf.geometry.head(3)]
        log.warning(f"  [{label}] ALL geometries failed repair. Samples: {samples}")

    return out


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_geometry_parquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Save as GeoParquet preserving `geometry` (WKB) + `geometry_wkt`
    (string) for backwards compatibility.

    The 'geometry' column carries GeoParquet metadata (CRS). The
    'geometry_wkt' string column is kept so old readers that call
    `shapely_wkt.loads(df['geometry_wkt'])` still work.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = gdf.copy()
    # Legacy string WKT (safely handles NaN geometries)
    out["geometry_wkt"] = out.geometry.apply(
        lambda g: g.wkt if (g is not None and not g.is_empty) else None
    )
    out.to_parquet(path, index=False)
    log.info(f"  saved {path.name} as GeoParquet "
             f"({len(out):,} rows, CRS={out.crs.to_epsg()})")