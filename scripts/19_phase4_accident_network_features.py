"""
Phase 4 · Iteration 7 — Join accidents to OSM edges and derive road features
for Fase 5.

Steps:
  1. Load network_osm.parquet, clean maxspeed, add fallback + is_imputed.
  2. Load callejero rotulación (polígonos) for CV_TX_DENO.
  3. Load accidents, project to EPSG:25830.
  4. Spatial join: accidents → nearest OSM edge (no max distance; note d).
  5. Point-in-polygon: accidents → callejero rotulación (tipo de vía).
  6. Derive per-accident road features.
  7. Save data/processed/accidents_with_network_features.parquet.
  8. Report:
       - match rate + distance distribution
       - feature coverage
       - sanity check: accidents outside Madrid city bounds

Run:
    python scripts/19_phase4_accident_network_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Fallback speeds per highway type (D073)
HIGHWAY_SPEED_FALLBACK = {
    "motorway": 120.0,
    "motorway_link": 80.0,
    "trunk": 100.0,
    "trunk_link": 60.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 30.0,
    "tertiary": 30.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "road": 30.0,
}

# Ordinal encoding (higher = bigger road, more traffic/speed)
HIGHWAY_ORDINAL = {
    "motorway": 7, "motorway_link": 6, "trunk": 6, "trunk_link": 5,
    "primary": 5, "primary_link": 4, "secondary": 4, "secondary_link": 3,
    "tertiary": 3, "tertiary_link": 2, "unclassified": 2,
    "residential": 2, "living_street": 1, "service": 1, "road": 1,
}


def load_osm_network() -> gpd.GeoDataFrame:
    df = pd.read_parquet(OUT_DATA / "network_osm.parquet")
    df["geometry"] = df["geometry_wkt"].apply(shapely_wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


def clean_osm_network(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clean and enrich maxspeed with clip + fallback + is_imputed flag."""
    gdf = gdf.copy()
    gdf["maxspeed_orig"] = gdf["maxspeed_kmh"]
    gdf["maxspeed_kmh"] = gdf["maxspeed_kmh"].clip(lower=10.0, upper=130.0)
    gdf.loc[(gdf["maxspeed_kmh"] < 10) | (gdf["maxspeed_kmh"] > 130),
            "maxspeed_kmh"] = np.nan
    fallback = gdf["highway"].map(HIGHWAY_SPEED_FALLBACK).fillna(30.0)
    gdf["maxspeed_kmh_filled"] = gdf["maxspeed_kmh"].fillna(fallback)
    gdf["maxspeed_is_imputed"] = gdf["maxspeed_kmh"].isna()
    gdf["highway_ordinal"] = (
        gdf["highway"].map(HIGHWAY_ORDINAL).fillna(2).astype(int)
    )
    return gdf


def load_callejero_polys() -> gpd.GeoDataFrame:
    """Load the callejero rotulación polygons (from Fase 4 iter 5)."""
    df = pd.read_parquet(OUT_DATA / "network_callejero.parquet")
    # Geometry may be None / NaN / pd.NA for empty geometries. Keep only
    # valid WKT strings, drop the rest with a warning count.
    wkt = df["geometry_wkt"]
    valid_mask = wkt.apply(lambda v: isinstance(v, str) and v.strip() != "")
    n_drop = int((~valid_mask).sum())
    if n_drop:
        print(f"  [warn] {n_drop} callejero features without WKT — dropped.")
    df = df.loc[valid_mask].copy()
    df["geometry"] = df["geometry_wkt"].apply(shapely_wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25830")
    return gdf


def main() -> None:
    print("1/6 Loading OSM network and cleaning...")
    net = load_osm_network()
    net = clean_osm_network(net)
    print(f"  {len(net)} edges, CRS={net.crs}.")
    print(f"  maxspeed non-null: {net['maxspeed_kmh'].notna().sum()} "
          f"(was {net['maxspeed_orig'].notna().sum()}).")
    print(f"  maxspeed after fill: {net['maxspeed_kmh_filled'].notna().sum()}")
    print(f"  imputed share: {net['maxspeed_is_imputed'].mean():.3f}")

    print("\n2/6 Loading callejero rotulación polygons...")
    calle = load_callejero_polys()
    print(f"  {len(calle)} polygons.")

    print("\n3/6 Loading accidents and projecting...")
    acc = pd.read_parquet(
        OUT_DATA / "accidents_clean.parquet",
        columns=["num_expediente", "year", "fecha", "hour",
                 "severity", "severity_code",
                 "coordenada_x_utm", "coordenada_y_utm"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()
    acc["geometry"] = gpd.points_from_xy(
        acc["coordenada_x_utm"], acc["coordenada_y_utm"]
    )
    acc_gdf = gpd.GeoDataFrame(acc, geometry="geometry", crs="EPSG:25830")
    print(f"  {len(acc_gdf)} accidents with valid coords.")

    # Sanity: detect accidents outside Madrid city bounds
    # City bounds in UTM 25830 (approx, generous): X 425k-460k, Y 4460k-4490k
    out_bounds = (
        (acc_gdf["coordenada_x_utm"] < 425_000)
        | (acc_gdf["coordenada_x_utm"] > 460_000)
        | (acc_gdf["coordenada_y_utm"] < 4_460_000)
        | (acc_gdf["coordenada_y_utm"] > 4_490_000)
    )
    n_out = int(out_bounds.sum())
    print(f"  Accidents outside Madrid UTM bounds: {n_out} "
          f"({100 * n_out / len(acc_gdf):.2f}%)")

    print("\n4/6 Spatial join: accidents → nearest OSM edge...")
    net_proj = net.to_crs("EPSG:25830")
    t0 = pd.Timestamp.now()
    joined = gpd.sjoin_nearest(
        acc_gdf,
        net_proj[["osm_id", "highway", "highway_ordinal", "name",
                  "maxspeed_kmh", "maxspeed_kmh_filled",
                  "maxspeed_is_imputed", "lanes_n", "length_m", "geometry"]],
        how="left", distance_col="dist_to_edge_m",
    )
    # Multiple nearest ties → keep the one with largest length
    joined = (joined
              .sort_values(["num_expediente", "dist_to_edge_m", "length_m"],
                           ascending=[True, True, False])
              .drop_duplicates(subset=["num_expediente"], keep="first"))
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    n_matched = int(joined["osm_id"].notna().sum())
    print(f"  matched {n_matched}/{len(joined)} ({100*n_matched/len(joined):.1f}%)"
          f" in {elapsed:.1f}s.")
    dists = joined["dist_to_edge_m"].dropna()
    print(f"  distance: median={dists.median():.1f}m, "
          f"p90={dists.quantile(0.9):.1f}m, max={dists.max():.1f}m")

    print("\n5/6 Point-in-polygon: accidents → callejero rotulación...")
    # Use callejero (already in 25830) polygons
    pip = gpd.sjoin(
        acc_gdf[["num_expediente", "geometry"]],
        calle[["TOP_ID", "CV_TX_DENO", "DEN_TX_NOM", "geometry"]],
        how="left", predicate="within",
    ).drop_duplicates(subset=["num_expediente"], keep="first")
    pip = pip.rename(columns={
        "TOP_ID": "callejero_top_id",
        "CV_TX_DENO": "tipo_via",
        "DEN_TX_NOM": "nombre_via",
    })
    n_tipo = int(pip["tipo_via"].notna().sum())
    print(f"  {n_tipo}/{len(pip)} accidents inside a callejero polygon.")
    print(f"  tipo_via distribution (top 5): "
          f"{pip['tipo_via'].value_counts().head(5).to_dict()}")

    print("\n6/6 Building final features DataFrame and saving...")
    # Merge per-accident data
    out_cols = [
        "num_expediente", "year", "fecha", "hour",
        "severity", "severity_code",
        "coordenada_x_utm", "coordenada_y_utm",
        "osm_id", "dist_to_edge_m",
        "highway", "highway_ordinal",
        "name", "maxspeed_kmh", "maxspeed_kmh_filled", "maxspeed_is_imputed",
        "lanes_n", "length_m",
    ]
    out = joined[out_cols].copy()
    # Add callejero tipo_via / nombre via
    pip_cols = pip[["num_expediente", "tipo_via", "nombre_via"]]
    out = out.merge(pip_cols, on="num_expediente", how="left")

    # Rename to clear names
    out = out.rename(columns={
        "osm_id": "edge_osm_id",
        "highway": "edge_highway",
        "highway_ordinal": "edge_highway_ordinal",
        "name": "edge_name",
        "maxspeed_kmh": "edge_maxspeed_raw",
        "maxspeed_kmh_filled": "edge_maxspeed",
        "maxspeed_is_imputed": "edge_maxspeed_is_imputed",
        "lanes_n": "edge_lanes",
        "length_m": "edge_length_m",
        "dist_to_edge_m": "edge_distance_m",
    })

    out.to_parquet(OUT_DATA / "accidents_with_network_features.parquet",
                   index=False)
    print(f"  Saved {OUT_DATA / 'accidents_with_network_features.parquet'} "
          f"({len(out)} rows, {out.shape[1]} cols).")

    # Report
    report = {
        "n_accidents_input": int(len(acc_gdf)),
        "n_accidents_matched_to_edge": n_matched,
        "match_share": round(n_matched / len(acc_gdf), 4),
        "distance_to_edge_stats": {
            "median_m": round(float(dists.median()), 1),
            "p90_m": round(float(dists.quantile(0.9)), 1),
            "p99_m": round(float(dists.quantile(0.99)), 1),
            "max_m": round(float(dists.max()), 1),
        },
        "accidents_outside_madrid_utm": n_out,
        "n_accidents_in_callejero_poly": n_tipo,
        "callejero_match_share": round(n_tipo / len(acc_gdf), 4),
        "tipo_via_distribution": (
            out["tipo_via"].value_counts(dropna=False).to_dict()
        ),
        "edge_highway_distribution": (
            out["edge_highway"].value_counts(dropna=False).to_dict()
        ),
        "edge_maxspeed_imputed_share": round(
            float(out["edge_maxspeed_is_imputed"].mean()), 4),
        "edge_lanes_null_share": round(
            float(out["edge_lanes"].isna().mean()), 4),
        "feature_null_shares": {
            c: round(float(out[c].isna().mean()), 4)
            for c in ["edge_highway", "edge_maxspeed", "edge_lanes",
                     "tipo_via", "edge_name"]
        },
    }
    save_json(report, OUT_TABLES / "19_accidents_with_network.json")

    sections = {
        "Join summary": dumps_jsonable(report, indent=2),
        "Notes": (
            "- Edges: OSM drive network from Fase 4 iter 6.\n"
            "- Edges matched by nearest geometry; ties broken by longest edge.\n"
            "- tipo_via comes from point-in-polygon of callejero rotulación "
            "(Calle, Avenida, Paseo...).\n"
            "- maxspeed_kmh after fill covers ~100%. Raw coverage ~37%.\n"
            "- maxspeed_is_imputed=True flags the 63% filled with fallback.\n"
            "- edge_distance_m = distance from accident point to nearest edge."
        ),
    }
    save_markdown_report(sections, Path("reports/19_phase4_features.md"))
    print("\nDone. See reports/19_phase4_features.md.")


if __name__ == "__main__":
    main()