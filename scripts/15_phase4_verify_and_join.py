"""
Phase 4 · Iteration 3 — Verify segment structure, check IMD year, and do the
first spatial join accidents → segments and IMD → segments.

Steps:
  1. Verify the 208223 segment count (are there 3928 Placemarks with coords,
     or more that our parser missed?).
  2. Dump a raw Placemark to confirm it has no name/description.
  3. Parse AÑO distribution in 203962 IMD.
  4. Convert accidents (EPSG:25830) → WGS84 and build GeoDataFrames.
  5. Spatial join accidents → nearest segment (max 100 m).
  6. Spatial join IMD points → nearest segment.
  7. Aggregate IMD per segment.
  8. Save:
       data/processed/segments_geo.parquet
       data/processed/segments_with_imd.parquet
       data/processed/accidents_with_segment.parquet
       reports/tables/15_*.json
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from lxml import etree
from pyproj import Transformer
from shapely import wkt as shapely_wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

TIMEOUT = 180
UA = "madrid-road-safety/phase4/0.3"
OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

KML_208223 = "https://informo.madrid.es/informo/tmadrid/INTENSIDADES.kml"
KMZ_203962 = ("https://datos.madrid.es/dataset/203962-0-trafico-imd/resource/"
              "203962-3-trafico-imd-kmz/download/203962-3-trafico-imd-kmz")


def download(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# 1. Verify 208223 raw structure
# ---------------------------------------------------------------------------
def verify_208223() -> dict[str, Any]:
    content = download(KML_208223)
    root = etree.fromstring(content)
    placemarks = list(root.iter("{*}Placemark"))
    linestrings_in_pm = sum(
        1 for pm in placemarks if pm.find(".//{*}LineString") is not None
    )
    coords_in_pm = sum(
        1 for pm in placemarks
        if pm.find(".//{*}LineString/{*}coordinates") is not None
    )
    # First placemark raw XML
    first_pm_xml = ""
    if placemarks:
        first_pm_xml = etree.tostring(
            placemarks[0], pretty_print=True, encoding="unicode"
        )[:1500]
    # Count names inside placemarks
    names_in_pm = sum(
        1 for pm in placemarks if pm.find("{*}name") is not None
    )
    return {
        "n_placemarks": len(placemarks),
        "n_placemarks_with_linestring": linestrings_in_pm,
        "n_placemarks_with_coords": coords_in_pm,
        "n_placemarks_with_name": names_in_pm,
        "first_placemark_xml": first_pm_xml,
    }


# ---------------------------------------------------------------------------
# 2. Verify IMD: AÑO distribution and parsing
# ---------------------------------------------------------------------------
def parse_imd_number(s):
    """Parse a number from the IMD KML (Spanish format mixed with plain floats)."""
    if pd.isna(s) or str(s).strip() in ("-", "", "nan", "None", "NULL"):
        return np.nan
    s = str(s).strip().replace("%", "").strip()
    if "," in s:
        try:
            return float(s.replace(".", "").replace(",", "."))
        except ValueError:
            return np.nan
    if "." in s:
        parts = s.split(".")
        if len(parts) == 2:
            int_part, frac_part = parts
            if int_part.isdigit() and frac_part.isdigit():
                if len(frac_part) == 3 and int(int_part) < 1000:
                    # "4.769" → 4769
                    return float(int_part + frac_part)
                try:
                    return float(s)  # "7354.000000" → 7354.0
                except ValueError:
                    return np.nan
        else:
            try:
                return float(s.replace(".", ""))
            except ValueError:
                return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def verify_imd(df: pd.DataFrame) -> dict[str, Any]:
    year_counts = df["AÑO"].value_counts(dropna=False).to_dict()
    year_counts = {str(k): int(v) for k, v in year_counts.items()}
    for c in ("Total S1", "Total S2", "TOTAL", "HP T"):
        df[c + "_parsed"] = df[c].apply(parse_imd_number)
    return {
        "n_points": int(len(df)),
        "year_counts": year_counts,
        "total_parsed_stats": {
            "n_valid": int(df["TOTAL_parsed"].notna().sum()),
            "min": float(df["TOTAL_parsed"].min()),
            "median": float(df["TOTAL_parsed"].median()),
            "max": float(df["TOTAL_parsed"].max()),
            "mean": round(float(df["TOTAL_parsed"].mean()), 1),
        },
        "rango_inte_counts": (df["Rango Inte"].value_counts(dropna=False)
                                .head(10).to_dict()),
        "sample_TOTAL_values": df["TOTAL"].head(5).tolist(),
        "sample_TOTAL_parsed": df["TOTAL_parsed"].head(5).tolist(),
    }


# ---------------------------------------------------------------------------
# 3. Spatial joins
# ---------------------------------------------------------------------------
def main() -> None:
    print("1/5 Verifying 208223 raw structure...")
    verify = verify_208223()
    save_json(verify, OUT_TABLES / "15_verify_208223.json")
    print(f"  Placemarks={verify['n_placemarks']}  "
          f"with LineString={verify['n_placemarks_with_linestring']}  "
          f"with coords={verify['n_placemarks_with_coords']}  "
          f"with name={verify['n_placemarks_with_name']}")

    print("\n2/5 Loading and verifying 203962 IMD points...")
    imd_df = pd.read_parquet(OUT_DATA / "imd_points.parquet")
    imd_ver = verify_imd(imd_df)
    save_json(imd_ver, OUT_TABLES / "15_verify_imd.json")
    print(f"  AÑO: {imd_ver['year_counts']}")
    print(f"  TOTAL parsed stats: {imd_ver['total_parsed_stats']}")

    print("\n3/5 Building GeoDataFrames...")
    # Segments
    seg_df = pd.read_parquet(OUT_DATA / "network_segments.parquet")
    seg_df["geometry"] = seg_df["geometry_wkt"].apply(shapely_wkt.loads)
    seg_gdf = gpd.GeoDataFrame(seg_df, geometry="geometry", crs="EPSG:4326")
    print(f"  Segments: {len(seg_gdf)}")

    # IMD points
    imd_df["geometry"] = gpd.points_from_xy(imd_df["lon"], imd_df["lat"])
    imd_gdf = gpd.GeoDataFrame(imd_df, geometry="geometry", crs="EPSG:4326")
    print(f"  IMD points: {len(imd_gdf)}")

    # Accidents: UTM 30N → WGS84
    acc = pd.read_parquet(
        OUT_DATA / "accidents_clean.parquet",
        columns=["num_expediente", "year", "severity", "severity_code",
                 "coordenada_x_utm", "coordenada_y_utm"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()
    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    acc["lon"], acc["lat"] = transformer.transform(
        acc["coordenada_x_utm"].to_numpy(),
        acc["coordenada_y_utm"].to_numpy(),
    )
    acc["geometry"] = gpd.points_from_xy(acc["lon"], acc["lat"])
    acc_gdf = gpd.GeoDataFrame(acc, geometry="geometry", crs="EPSG:4326")
    print(f"  Accidents with valid coords: {len(acc_gdf)}")

    print("\n4/5 Spatial joins...")
    # Project everything to metric CRS for accurate distance
    seg_proj = seg_gdf.to_crs("EPSG:25830")
    acc_proj = acc_gdf.to_crs("EPSG:25830")
    imd_proj = imd_gdf.to_crs("EPSG:25830")

    # Accidents → nearest segment (max 100 m)
    acc_join = gpd.sjoin_nearest(
        acc_proj, seg_proj[["segment_id", "geometry"]],
        how="left", max_distance=100, distance_col="dist_m",
    )
    # Multiple nearest segments per accident → keep the closest
    acc_join = acc_join.sort_values("dist_m").drop_duplicates(
        subset=["num_expediente"], keep="first"
    )
    n_acc_matched = int(acc_join["segment_id"].notna().sum())
    print(f"  Accidents matched to a segment: {n_acc_matched}/{len(acc_join)} "
          f"({100 * n_acc_matched / len(acc_join):.1f}%)")

    # IMD points → nearest segment (no max distance, generous buffer)
    imd_join = gpd.sjoin_nearest(
        imd_proj, seg_proj[["segment_id", "geometry"]],
        how="left", distance_col="dist_m",
    )
    imd_join = imd_join.sort_values("dist_m").drop_duplicates(
        subset=["point_id"], keep="first"
    )
    n_imd_matched = int(imd_join["segment_id"].notna().sum())
    print(f"  IMD points matched to a segment: {n_imd_matched}/{len(imd_join)}")

    print("\n5/5 Aggregating IMD per segment and saving...")
    # Aggregate IMD per segment
    imd_agg = imd_join.groupby("segment_id").agg(
        n_imd_points=("point_id", "size"),
        imd_total_mean=("TOTAL_parsed", "mean"),
        imd_total_median=("TOTAL_parsed", "median"),
        imd_total_max=("TOTAL_parsed", "max"),
        heavy_pct_mean=("HP T_parsed", "mean"),
        rango_inte_mode=("Rango Inte", lambda s: s.mode().iloc[0]
                          if len(s.mode()) > 0 else np.nan),
    ).reset_index()

    # Aggregate accidents per segment
    acc_agg = (acc_join.dropna(subset=["segment_id"])
                        .groupby("segment_id")
                        .agg(
                            n_accidents=("num_expediente", "size"),
                            n_fatal=("severity_code",
                                     lambda s: (s == 3).sum()),
                            n_serious=("severity_code",
                                       lambda s: (s == 2).sum()),
                            mean_severity=("severity_code", "mean"),
                        ).reset_index())

    # Merge into segments
    seg_out = seg_gdf.merge(imd_agg, on="segment_id", how="left") \
                       .merge(acc_agg, on="segment_id", how="left")
    seg_out["n_imd_points"] = seg_out["n_imd_points"].fillna(0).astype(int)
    seg_out["n_accidents"] = seg_out["n_accidents"].fillna(0).astype(int)
    seg_out["n_fatal"] = seg_out["n_fatal"].fillna(0).astype(int)
    seg_out["n_serious"] = seg_out["n_serious"].fillna(0).astype(int)

    # Segment length (projected)
    seg_out["segment_length_m"] = seg_out.to_crs("EPSG:25830").length

    seg_out.drop(columns=["geometry"]).to_parquet(
        OUT_DATA / "segments_with_imd.parquet", index=False
    )

    # Accidents with segment assignment
    acc_out = acc_join.drop(columns=["geometry"]).copy()
    acc_out.to_parquet(
        OUT_DATA / "accidents_with_segment.parquet", index=False
    )

    # ---- Coverage summary ----
    n_seg = len(seg_out)
    n_seg_with_imd = int((seg_out["n_imd_points"] > 0).sum())
    n_seg_with_acc = int((seg_out["n_accidents"] > 0).sum())
    dist_stats = {
        "n_matched": n_acc_matched,
        "share_matched": round(n_acc_matched / len(acc_join), 4),
        "dist_m_quantiles": {
            "50": round(float(acc_join["dist_m"].quantile(0.5)), 1),
            "90": round(float(acc_join["dist_m"].quantile(0.9)), 1),
            "99": round(float(acc_join["dist_m"].quantile(0.99)), 1),
            "max": round(float(acc_join["dist_m"].max()), 1),
        },
    }
    coverage = {
        "n_segments": n_seg,
        "n_segments_with_imd": n_seg_with_imd,
        "n_segments_with_accidents": n_seg_with_acc,
        "segments_share_with_imd": round(n_seg_with_imd / n_seg, 4),
        "segments_share_with_accidents": round(n_seg_with_acc / n_seg, 4),
        "accidents_matching": dist_stats,
        "imd_total_per_segment_stats": {
            "n_notnull": int(seg_out["imd_total_mean"].notna().sum()),
            "median": (round(float(seg_out["imd_total_mean"].median()), 1)
                       if seg_out["imd_total_mean"].notna().any() else None),
            "p90": (round(float(seg_out["imd_total_mean"].quantile(0.9)), 1)
                    if seg_out["imd_total_mean"].notna().any() else None),
        },
        "n_accidents_by_segment_stats": {
            "n_with_0": int((seg_out["n_accidents"] == 0).sum()),
            "n_with_1_to_5": int(((seg_out["n_accidents"] >= 1)
                                  & (seg_out["n_accidents"] <= 5)).sum()),
            "n_with_6_to_20": int(((seg_out["n_accidents"] >= 6)
                                   & (seg_out["n_accidents"] <= 20)).sum()),
            "n_with_21_plus": int((seg_out["n_accidents"] >= 21).sum()),
            "max": int(seg_out["n_accidents"].max()),
        },
    }
    save_json(coverage, OUT_TABLES / "15_coverage.json")

    sections = {
        "208223 raw structure": dumps_jsonable(verify, indent=2),
        "203962 AÑO and TOTAL parsing": dumps_jsonable(imd_ver, indent=2),
        "Spatial join coverage": dumps_jsonable(coverage, indent=2),
        "Notes": (
            "- Segments and IMD points matched by nearest geometry "
            "(no attribute ID match, since 208223 has no attributes).\n"
            "- Accidents limited to 100 m max distance from a segment.\n"
            "- IMD year: check 'AÑO' distribution. If all 2013, the intensity "
            "is a 2013 snapshot, not 2019-2025."
        ),
    }
    save_markdown_report(sections, Path("reports/15_phase4_verify_and_join.md"))
    print("\nDone. See reports/15_phase4_verify_and_join.md.")


if __name__ == "__main__":
    main()