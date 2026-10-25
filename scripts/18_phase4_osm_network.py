"""
Phase 4 · Iteration 6 — Build the definitive road network from OpenStreetMap.

Steps:
  1. Compute bbox from accidents (padding 500m).
  2. Fetch OSM drive network with osmnx.
  3. Convert to GeoDataFrame of LineStrings.
  4. Normalize attributes: highway, maxspeed, lanes, name, oneway, length.
  5. Persist as network_osm.parquet (WKT geometry, EPSG:4326 + CRS metadata).
  6. Report structure and coverage.

Run:
    pip install osmnx
    python scripts/18_phase4_osm_network.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

ACCIDENTS = OUT_DATA / "accidents_clean.parquet"


def compute_bbox() -> tuple[float, float, float, float]:
    """Bbox of accidents in WGS84, padded 500 m."""
    acc = pd.read_parquet(
        ACCIDENTS,
        columns=["coordenada_x_utm", "coordenada_y_utm"],
    ).dropna()
    tr = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(
        acc["coordenada_x_utm"].to_numpy(),
        acc["coordenada_y_utm"].to_numpy(),
    )
    pad_lon = 500 / (111_320 * np.cos(np.radians(np.mean(lat))))
    pad_lat = 500 / 110_540
    return (
        float(np.min(lon) - pad_lon),  # left
        float(np.min(lat) - pad_lat),  # bottom
        float(np.max(lon) + pad_lon),  # right
        float(np.max(lat) + pad_lat),  # top
    )


def fetch_osm_drive_network(bbox: tuple[float, float, float, float]
                             ) -> gpd.GeoDataFrame:
    """Fetch OSM drive network in bbox via osmnx. Returns edges as GeoDataFrame."""
    import osmnx as ox

    left, bottom, right, top = bbox
    print(f"  bbox: left={left:.4f} bottom={bottom:.4f} "
          f"right={right:.4f} top={top:.4f}")

    # osmnx 1.x and 2.x — both accept (left, bottom, right, top) tuple
    t0 = time.time()
    try:
        G = ox.graph_from_bbox(
            bbox=(left, bottom, right, top),
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
    except TypeError:
        # osmnx < 1.5 uses bbox=(north, south, east, west)
        G = ox.graph_from_bbox(
            north=top, south=bottom, east=right, west=left,
            network_type="drive", simplify=True, retain_all=False,
            truncate_by_edge=True,
        )
    print(f"  fetched graph in {time.time() - t0:.1f}s: "
          f"{len(G.nodes)} nodes, {len(G.edges)} edges.")

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    return edges


def normalize_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize attribute columns to predictable names and types."""
    e = edges.copy()

    # Standard names
    e["osm_id"] = e["osmid"].astype(str)
    e["u"] = e["u"].astype(int)
    e["v"] = e["v"].astype(int)
    e["key"] = e["key"].astype(int)

    # highway: take first if list
    def _first(x):
        if isinstance(x, list):
            return x[0] if x else None
        return x
    e["highway"] = e["highway"].apply(_first)
    e["name"] = e["name"].apply(_first)

    # maxspeed: OSM stores as int, str, or list; "50 mph", "30", etc.
    def _parse_speed(x):
        if isinstance(x, list):
            x = x[0] if x else None
        if x is None or pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ("none", "signals", "walk", "variable"):
            return np.nan
        if "mph" in s:
            try:
                return float(s.replace("mph", "").strip()) * 1.60934
            except ValueError:
                return np.nan
        try:
            return float(s.replace("km/h", "").strip())
        except ValueError:
            return np.nan
    e["maxspeed_kmh"] = e["maxspeed"].apply(_parse_speed)

    # lanes
    def _parse_lanes(x):
        if isinstance(x, list):
            x = x[0] if x else None
        if x is None or pd.isna(x):
            return np.nan
        try:
            return float(str(x).strip())
        except ValueError:
            return np.nan
    e["lanes_n"] = e["lanes"].apply(_parse_lanes) if "lanes" in e.columns else np.nan

    # oneway
    e["oneway"] = e.get("oneway", False)
    # length already in meters (osmnx computes it)
    if "length" in e.columns:
        e["length_m"] = e["length"].astype(float)

    # Drop heavy or redundant columns
    drop_cols = [c for c in ("osmid", "geometry",
                             "maxspeed", "lanes", "reversed", "ref",
                             "bridge", "tunnel", "access", "service")
                 if c in e.columns]
    keep = ["osm_id", "u", "v", "key", "highway", "name",
            "maxspeed_kmh", "lanes_n", "oneway", "length_m", "geometry"]
    return e[[c for c in keep if c in e.columns]]


def main() -> None:
    print("1/4 Computing bbox from accidents...")
    bbox = compute_bbox()
    save_json({"bbox_wgs84": list(bbox)},
               OUT_TABLES / "18_osm_bbox.json")

    print("\n2/4 Fetching OSM drive network...")
    edges = fetch_osm_drive_network(bbox)
    print(f"  raw edges: {len(edges):,}")

    print("\n3/4 Normalizing attributes...")
    edges_n = normalize_edges(edges)
    print(f"  normalized columns: {list(edges_n.columns)}")
    print(f"  highway distribution (top 10):")
    print(edges_n["highway"].value_counts().head(10).to_string())

    # Persist as WKT (geometry → string)
    out = edges_n.copy()
    if "geometry" in out.columns:
        out["geometry_wkt"] = out["geometry"].apply(
            lambda g: g.wkt if g is not None else None)
        out = out.drop(columns=["geometry"])
    out.to_parquet(OUT_DATA / "network_osm.parquet", index=False)
    print(f"  Saved {OUT_DATA / 'network_osm.parquet'}")

    print("\n4/4 Building summary...")
    def _stats(series: pd.Series) -> dict:
        s = series.dropna()
        return {
            "n_valid": int(len(s)),
            "n_null": int(series.isna().sum()),
            "min": float(s.min()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "p90": float(s.quantile(0.9)) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
        }

    summary = {
        "n_edges": int(len(edges_n)),
        "crs_original": "EPSG:4326",
        "highway_counts": edges_n["highway"].value_counts().to_dict(),
        "name_null_share": round(float(edges_n["name"].isna().mean()), 4),
        "maxspeed_kmh_stats": _stats(edges_n["maxspeed_kmh"]),
        "maxspeed_kmh_counts": (
            edges_n["maxspeed_kmh"].value_counts().head(10).to_dict()
            if "maxspeed_kmh" in edges_n.columns else {}
        ),
        "lanes_n_stats": _stats(edges_n["lanes_n"])
            if "lanes_n" in edges_n.columns else None,
        "length_m_stats": _stats(edges_n["length_m"]),
        "total_network_length_km": round(
            float(edges_n["length_m"].sum()) / 1000, 1),
    }
    save_json(summary, OUT_TABLES / "18_osm_summary.json")

    sections = {
        "OSM bbox": dumps_jsonable({"bbox_wgs84": list(bbox)}, indent=2),
        "OSM network summary": dumps_jsonable(summary, indent=2),
        "Notes": (
            "- network_type='drive' (motorways, primary, secondary, "
            "residential, service principal).\n"
            "- Edges are simplified (osmnx simplify=True).\n"
            "- maxspeed_kmh and lanes_n have nulls where OSM lacks data; "
            "Fase 5 imputes or uses missingness indicators.\n"
            "- length_m is the edge length in metres."
        ),
    }
    save_markdown_report(sections, Path("reports/18_phase4_osm.md"))
    print("\nDone. See reports/18_phase4_osm.md.")


if __name__ == "__main__":
    main()