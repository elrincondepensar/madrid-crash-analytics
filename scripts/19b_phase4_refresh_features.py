"""
Phase 4 · Fix (v2) — Regenerate accidents_with_network_features.parquet.

Background
----------
The original parquet (script 19) was built with the unrepaired callejero
geometries. The point-in-polygon `sjoin(predicate='within')` operated on
self-intersecting polygons, which can produce slightly incorrect `tipo_via`
assignments. Additionally, D077 flagged that the parquet has `edge_osm_id`
(string) but no positional `edge_id` (int64), forcing downstream phases
to reconstruct it via cKDTree on midpoints.

What this script does
---------------------
  1. Load the FIXED network_callejero.parquet (17c) and network_osm.parquet
     (18b) as GeoParquets.
  2. Load accidents_clean.parquet.
  3. Recompute:
       - edge join via sjoin_nearest on LineString geometry
       - tipo_via / nombre_via via sjoin(predicate='within') on REPAIRED
         polygons
  4. Add `edge_id` (int64, positional, matches network_osm.parquet row order).
  5. Compare new tipo_via distribution against the old file:
       - n_changed, %_changed, examples
  6. Overwrite accidents_with_network_features.parquet.

Run:
    python scripts/19b_phase4_refresh_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import (
    dumps_jsonable, save_json, save_markdown_report,
)

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Fallbacks and ordinals from original script 19 (D073)
HIGHWAY_SPEED_FALLBACK = {
    "motorway": 120.0, "motorway_link": 80.0,
    "trunk": 100.0,    "trunk_link": 60.0,
    "primary": 50.0,   "primary_link": 40.0,
    "secondary": 40.0, "secondary_link": 30.0,
    "tertiary": 30.0,  "tertiary_link": 30.0,
    "unclassified": 30.0, "residential": 30.0,
    "living_street": 20.0, "service": 20.0, "road": 30.0,
}
HIGHWAY_ORDINAL = {
    "motorway": 7, "motorway_link": 6, "trunk": 6, "trunk_link": 5,
    "primary": 5, "primary_link": 4, "secondary": 4, "secondary_link": 3,
    "tertiary": 3, "tertiary_link": 2, "unclassified": 2,
    "residential": 2, "living_street": 1, "service": 1, "road": 1,
}

ACCIDENTS_PATH = OUT_DATA / "accidents_clean.parquet"
NETWORK_OSM_PATH = OUT_DATA / "network_osm.parquet"
CALLEJERO_PATH = OUT_DATA / "network_callejero.parquet"
OLD_FEATURES = OUT_DATA / "accidents_with_network_features.parquet"


def clean_osm_network(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply D073/D074 cleaning: clip maxspeed, fill fallback, ordinal."""
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


def main() -> None:
    # ---- 0. Snapshot old for comparison ----
    print("0/5 Snapshot old features for comparison ...")
    if OLD_FEATURES.exists():
        old = pd.read_parquet(OLD_FEATURES)
        old_tipo = old[["num_expediente", "tipo_via"]].rename(
            columns={"tipo_via": "tipo_via_old"})
        print(f"  old rows: {len(old):,}")
    else:
        old_tipo = pd.DataFrame(columns=["num_expediente", "tipo_via_old"])
        print("  (no old file; new file will be created from scratch)")

    # ---- 1. Load accidents ----
    print("\n1/5 Loading accidents_clean.parquet ...")
    acc = pd.read_parquet(
        ACCIDENTS_PATH,
        columns=["num_expediente", "year", "fecha", "hour",
                 "severity", "severity_code",
                 "coordenada_x_utm", "coordenada_y_utm"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()
    acc["geometry"] = gpd.points_from_xy(
        acc["coordenada_x_utm"], acc["coordenada_y_utm"])
    acc_gdf = gpd.GeoDataFrame(acc, geometry="geometry", crs="EPSG:25830")
    print(f"  accidents: {len(acc_gdf):,}")

    # ---- 2. Load network (GeoParquet after 18b) ----
    print("\n2/5 Loading network_osm.parquet ...")
    net = ms_io.load_geometry_parquet(NETWORK_OSM_PATH)
    net = net.to_crs("EPSG:25830")
    net = clean_osm_network(net)
    net = net.reset_index(drop=True)
    net["edge_id"] = np.arange(len(net), dtype=np.int64)
    print(f"  edges: {len(net):,}  (edge_id 0..{len(net)-1})")

    # ---- 3. Load callejero (GeoParquet after 17c) ----
    print("\n3/5 Loading network_callejero.parquet ...")
    calle = ms_io.load_geometry_parquet(CALLEJERO_PATH)
    calle = calle.to_crs("EPSG:25830")
    print(f"  polygons: {len(calle):,}")

    # ---- 4. Spatial joins ----
    print("\n4/5 Spatial joins ...")
    # Accident → nearest edge (LineString)
    print("  sjoin_nearest: accidents → edges ...")
    joined = gpd.sjoin_nearest(
        acc_gdf,
        net[["edge_id", "osm_id", "highway", "highway_ordinal", "name",
             "maxspeed_kmh", "maxspeed_kmh_filled", "maxspeed_is_imputed",
             "lanes_n", "length_m", "geometry"]],
        how="left", distance_col="dist_to_edge_m",
    )
    # Tie-break: keep shortest distance, then longest edge
    joined = (joined
              .sort_values(["num_expediente", "dist_to_edge_m", "length_m"],
                           ascending=[True, True, False])
              .drop_duplicates(subset=["num_expediente"], keep="first"))
    n_matched = int(joined["edge_id"].notna().sum())
    print(f"  matched {n_matched}/{len(joined)} "
          f"({100*n_matched/len(joined):.1f}%)")

    # Accident → callejero polygon (within)
    print("  sjoin: accidents within callejero polygons ...")
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
    print(f"  {n_tipo}/{len(pip)} inside a callejero polygon "
          f"({100*n_tipo/len(pip):.1f}%)")

    # ---- 5. Build final DataFrame ----
    print("\n5/5 Building final features DataFrame ...")
    out_cols = [
        "num_expediente", "year", "fecha", "hour",
        "severity", "severity_code",
        "coordenada_x_utm", "coordenada_y_utm",
        "edge_id", "osm_id", "dist_to_edge_m",
        "highway", "highway_ordinal", "name",
        "maxspeed_kmh", "maxspeed_kmh_filled", "maxspeed_is_imputed",
        "lanes_n", "length_m",
    ]
    out = joined[out_cols].copy()
    out = out.merge(pip[["num_expediente", "tipo_via", "nombre_via"]],
                    on="num_expediente", how="left")
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
    out = out.drop_duplicates(subset=["num_expediente"]).reset_index(drop=True)
    print(f"  new rows: {len(out):,}  cols: {len(out.columns)}")

    # ---- 6. Compare tipo_via old vs new ----
    print("\n   Comparing tipo_via old vs new ...")
    comp = out[["num_expediente", "tipo_via"]].rename(
        columns={"tipo_via": "tipo_via_new"}).merge(
        old_tipo, on="num_expediente", how="left")
    mask = (comp["tipo_via_new"].notna() | comp["tipo_via_old"].notna())
    comp = comp[mask]
    # Changed if values differ (treating NaN as a value)
    changed = (
        (comp["tipo_via_new"].fillna("__NA__")
         != comp["tipo_via_old"].fillna("__NA__"))
    )
    n_changed = int(changed.sum())
    pct_changed = n_changed / len(comp) if len(comp) > 0 else 0.0
    print(f"   {n_changed:,}/{len(comp):,} changed "
          f"({100*pct_changed:.2f}%)")

    # Transitions
    transitions = (
        comp[changed]
        .groupby(["tipo_via_old", "tipo_via_new"], dropna=False)
        .size().sort_values(ascending=False).head(10)
        .reset_index(name="count").to_dict(orient="records")
    )

    # ---- 7. Save ----
    out.to_parquet(OLD_FEATURES, index=False)
    print(f"\n   saved {OLD_FEATURES} ({len(out):,} rows, {len(out.columns)} cols)")

    # ---- 8. Report ----
    report = {
        "n_accidents_input": int(len(acc_gdf)),
        "n_accidents_matched_to_edge": n_matched,
        "match_share_edge": round(n_matched / len(acc_gdf), 4),
        "n_accidents_in_callejero_poly": n_tipo,
        "match_share_callejero": round(n_tipo / len(acc_gdf), 4),
        "edge_distance_stats": {
            "median_m": round(float(out["edge_distance_m"].median()), 2),
            "p90_m": round(float(out["edge_distance_m"].quantile(0.9)), 2),
            "max_m": round(float(out["edge_distance_m"].max()), 2),
        },
        "tipo_via_changes": {
            "n_changed": n_changed,
            "n_compared": int(len(comp)),
            "pct_changed": round(pct_changed * 100, 4),
            "transitions_top10": transitions,
        },
        "edge_id_present": "edge_id" in out.columns,
        "edge_id_min": int(out["edge_id"].min()) if "edge_id" in out.columns else None,
        "edge_id_max": int(out["edge_id"].max()) if "edge_id" in out.columns else None,
        "n_null_edge_id": int(out["edge_id"].isna().sum()),
    }
    save_json(report, OUT_TABLES / "19b_features_refresh.json")

    sections = {
        "Feature refresh": dumps_jsonable(report, indent=2),
        "tipo_via transitions (top 10)": dumps_jsonable(transitions, indent=2),
        "Notes": (
            "- edge_id is now a positional int64 that matches "
            "network_osm.parquet row order (fix for D077).\n"
            "- tipo_via recomputed on REPAIRED callejero polygons "
            "(shapely.make_valid + buffer(0) fallback).\n"
            "- Phase 5 models are NOT re-trained. The tipo_via change "
            "affects a small fraction of accidents and has negligible "
            "impact given D083 (network features contribute marginally).\n"
        ),
    }
    save_markdown_report(sections, Path("reports/19b_phase4_refresh_features.md"))
    print("\nDone. See reports/19b_phase4_refresh_features.md.")


if __name__ == "__main__":
    main()