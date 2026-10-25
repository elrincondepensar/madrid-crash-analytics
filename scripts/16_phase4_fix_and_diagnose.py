"""
Phase 4 · Iteration 4 — Fix IMD parsing and diagnose the 25.7% accident non-match.

Three outputs:
  1. Corrected imd_points.parquet (IMD values in real veh/day units).
  2. Corrected segments_with_imd.parquet (real IMD values per segment).
  3. Diagnostic of the accidents that fail to match a segment at 100m:
     - distribution by year, district, severity
     - distance to nearest segment (at 100, 200, 500m)
     - share of accidents that would match at larger thresholds

Run:
    python scripts/16_phase4_fix_and_diagnose.py
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

# Sanity thresholds derived from Rango Inte labels
RANGO_BOUNDS = {
    "<  1.000 Veh/día": (0, 1_000),
    "Entre 1.000-5.000 Veh/día": (1_000, 5_000),
    "Entre 5.000-10.000 Veh/día": (5_000, 10_000),
    "Entre 10.000-20.000 Veh/día": (10_000, 20_000),
    "Entre 20.000-40.000 Veh/día": (20_000, 40_000),
    "Entre 40.000-60.000 Veh/día": (40_000, 60_000),
    "Entre 60.000-80.000 Veh/día": (60_000, 80_000),
    "Entre 80.000-100.000 Veh/día": (80_000, 100_000),
    "> 100.000 Veh/día": (100_000, 10_000_000),
}


def parse_number_with_range_sanity(raw, rango_label):
    """
    Parse a TOTAL string using Rango Inte as authoritative range.
    Try formats: "7.354" (ES thousands) → 7354; "7354.0" (float) → 7354;
    "7,354" (comma) → 7.354 (if in range) else 7354.
    Pick the interpretation that falls inside the Rango Inte bounds.
    """
    if pd.isna(raw) or str(raw).strip() in ("-", "", "nan", "None", "NULL"):
        return np.nan
    s = str(raw).strip().replace("%", "").strip()
    lo, hi = RANGO_BOUNDS.get(rango_label, (0, 10_000_000))

    candidates = []
    # 1. Try direct float
    try:
        candidates.append(float(s))
    except ValueError:
        pass
    # 2. Try removing thousands-dots: "7.354" → 7354, "99.500" → 99500
    s_dots_removed = s.replace(".", "")
    try:
        candidates.append(float(s_dots_removed))
    except ValueError:
        pass
    # 3. Try replacing comma with dot: "7,354" → 7.354
    s_commas = s.replace(".", "").replace(",", ".")
    try:
        candidates.append(float(s_commas))
    except ValueError:
        pass
    # 4. Try removing all separators (most aggressive): "7.354" → 7354
    s_clean = s.replace(".", "").replace(",", "")
    try:
        candidates.append(float(s_clean))
    except ValueError:
        pass

    # Prefer candidates inside the range, then closest to the range midpoint
    mid = (lo + hi) / 2
    in_range = [c for c in candidates if lo <= c <= hi]
    if in_range:
        return min(in_range, key=lambda c: abs(c - mid))
    # Fallback: closest to mid
    if candidates:
        return min(candidates, key=lambda c: abs(c - mid))
    return np.nan


def fix_imd(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Re-parse TOTAL and HP T using Rango Inte as sanity."""
    df = df.copy()
    df["TOTAL_fixed"] = [
        parse_number_with_range_sanity(v, r)
        for v, r in zip(df["TOTAL"], df["Rango Inte"])
    ]
    df["HP_T_fixed"] = [
        parse_number_with_range_sanity(v, r)
        for v, r in zip(df["HP T"], df["Rango Inte"])
    ]
    # Sanity check: how many still disagree with Rango Inte
    def _in_range(v, r):
        lo, hi = RANGO_BOUNDS.get(r, (0, 10_000_000))
        return not pd.isna(v) and lo <= v <= hi
    df["_in_range"] = [
        _in_range(v, r) for v, r in zip(df["TOTAL_fixed"], df["Rango Inte"])
    ]

    stats = {
        "n_rows": int(len(df)),
        "n_parsed": int(df["TOTAL_fixed"].notna().sum()),
        "n_still_off_range": int((~df["_in_range"]).sum()),
        "TOTAL_fixed_stats": {
            "min": float(df["TOTAL_fixed"].min()),
            "median": float(df["TOTAL_fixed"].median()),
            "mean": round(float(df["TOTAL_fixed"].mean()), 1),
            "p90": float(df["TOTAL_fixed"].quantile(0.9)),
            "max": float(df["TOTAL_fixed"].max()),
        },
        "HP_T_fixed_stats": {
            "min": float(df["HP_T_fixed"].min()),
            "median": float(df["HP_T_fixed"].median()),
            "max": float(df["HP_T_fixed"].max()),
        },
        "rango_counts_after_fix": df["Rango Inte"].value_counts().to_dict(),
    }
    return df, stats


def rebuild_segments_with_imd() -> tuple[gpd.GeoDataFrame, dict]:
    """Rebuild segment-to-IMD aggregation with the fixed IMD values."""
    seg_df = pd.read_parquet(OUT_DATA / "network_segments.parquet")
    seg_df["geometry"] = seg_df["geometry_wkt"].apply(shapely_wkt.loads)
    seg_gdf = gpd.GeoDataFrame(seg_df, geometry="geometry", crs="EPSG:4326")

    imd_df = pd.read_parquet(OUT_DATA / "imd_points.parquet")
    imd_df["geometry"] = gpd.points_from_xy(imd_df["lon"], imd_df["lat"])
    imd_gdf = gpd.GeoDataFrame(imd_df, geometry="geometry", crs="EPSG:4326")

    seg_proj = seg_gdf.to_crs("EPSG:25830")
    imd_proj = imd_gdf.to_crs("EPSG:25830")

    imd_join = gpd.sjoin_nearest(
        imd_proj, seg_proj[["segment_id", "geometry"]],
        how="left", distance_col="dist_m",
    ).sort_values("dist_m").drop_duplicates(
        subset=["point_id"], keep="first"
    )

    imd_agg = imd_join.groupby("segment_id").agg(
        n_imd_points=("point_id", "size"),
        imd_total_mean=("TOTAL_fixed", "mean"),
        imd_total_median=("TOTAL_fixed", "median"),
        imd_total_max=("TOTAL_fixed", "max"),
        heavy_pct_mean=("HP_T_fixed", "mean"),
        rango_inte_mode=("Rango Inte", lambda s: s.mode().iloc[0]
                          if len(s.mode()) > 0 else np.nan),
    ).reset_index()

    seg_out = seg_gdf.merge(imd_agg, on="segment_id", how="left")
    seg_out["n_imd_points"] = seg_out["n_imd_points"].fillna(0).astype(int)
    seg_out["segment_length_m"] = seg_out.to_crs("EPSG:25830").length

    seg_out.drop(columns=["geometry"]).to_parquet(
        OUT_DATA / "segments_with_imd.parquet", index=False
    )
    stats = {
        "n_segments": int(len(seg_out)),
        "n_segments_with_imd": int((seg_out["n_imd_points"] > 0).sum()),
        "imd_mean_per_segment_stats": {
            "n_notnull": int(seg_out["imd_total_mean"].notna().sum()),
            "median": (float(seg_out["imd_total_mean"].median())
                       if seg_out["imd_total_mean"].notna().any() else None),
            "p90": (float(seg_out["imd_total_mean"].quantile(0.9))
                    if seg_out["imd_total_mean"].notna().any() else None),
        },
    }
    return seg_out, stats


def diagnose_no_match() -> dict:
    """Diagnose the ~25% of accidents that don't match a 100m-radius segment."""
    seg_df = pd.read_parquet(OUT_DATA / "network_segments.parquet")
    seg_df["geometry"] = seg_df["geometry_wkt"].apply(shapely_wkt.loads)
    seg_gdf = gpd.GeoDataFrame(seg_df, geometry="geometry", crs="EPSG:4326")
    seg_proj = seg_gdf.to_crs("EPSG:25830")

    acc = pd.read_parquet(
        OUT_DATA / "accidents_clean.parquet",
        columns=["num_expediente", "year", "severity", "severity_code",
                 "cod_distrito", "distrito",
                 "coordenada_x_utm", "coordenada_y_utm"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"]).copy()

    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    acc["lon"], acc["lat"] = transformer.transform(
        acc["coordenada_x_utm"].to_numpy(),
        acc["coordenada_y_utm"].to_numpy(),
    )
    acc["geometry"] = gpd.points_from_xy(acc["lon"], acc["lat"])
    acc_gdf = gpd.GeoDataFrame(acc, geometry="geometry", crs="EPSG:4326")
    acc_proj = acc_gdf.to_crs("EPSG:25830")

    # Distance to nearest segment, no max
    acc_dist = gpd.sjoin_nearest(
        acc_proj[["num_expediente", "geometry"]],
        seg_proj[["segment_id", "geometry"]],
        how="left", distance_col="dist_m",
    ).sort_values("dist_m").drop_duplicates(
        subset=["num_expediente"], keep="first"
    )
    acc_full = acc.merge(acc_dist[["num_expediente", "dist_m"]],
                          on="num_expediente", how="left")

    thresholds = [50, 100, 200, 300, 500, 1000]
    match_rates = {}
    for t in thresholds:
        n_match = int((acc_full["dist_m"] <= t).sum())
        match_rates[f"{t}m"] = {
            "n_matched": n_match,
            "share": round(n_match / len(acc_full), 4),
        }

    acc_full["matched_100m"] = acc_full["dist_m"] <= 100

    # By year
    by_year = (acc_full.groupby("year")["matched_100m"].mean().round(4).to_dict())
    # By district (injury_rate high vs low)
    by_district = (acc_full.groupby("distrito")["matched_100m"]
                    .agg(["mean", "size"]).round(4)
                    .reset_index().to_dict(orient="records"))
    # By severity
    by_severity = (acc_full.groupby("severity")["matched_100m"]
                    .agg(["mean", "size"]).round(4)
                    .reset_index().to_dict(orient="records"))
    # Distance distribution of the non-matched
    unmatched_dist = acc_full.loc[~acc_full["matched_100m"], "dist_m"]
    unmatched_dist_stats = {
        "n": int(len(unmatched_dist)),
        "p50": (float(unmatched_dist.quantile(0.5))
                if len(unmatched_dist) else None),
        "p90": (float(unmatched_dist.quantile(0.9))
                if len(unmatched_dist) else None),
        "p99": (float(unmatched_dist.quantile(0.99))
                if len(unmatched_dist) else None),
        "max": (float(unmatched_dist.max())
                if len(unmatched_dist) else None),
    }

    return {
        "n_accidents_total": int(len(acc_full)),
        "match_rates_by_threshold": match_rates,
        "match_rate_100m_by_year": by_year,
        "match_rate_100m_by_district": by_district,
        "match_rate_100m_by_severity": by_severity,
        "unmatched_distance_stats": unmatched_dist_stats,
    }


def main() -> None:
    print("1/3 Fixing IMD parsing...")
    imd_df = pd.read_parquet(OUT_DATA / "imd_points.parquet")
    imd_fixed, imd_stats = fix_imd(imd_df)
    imd_fixed.drop(columns=["geometry"], errors="ignore").to_parquet(
        OUT_DATA / "imd_points.parquet", index=False
    )
    save_json(imd_stats, OUT_TABLES / "16_imd_fixed.json")
    print(f"  TOTAL fixed stats: {imd_stats['TOTAL_fixed_stats']}")
    print(f"  Rows still off-range: {imd_stats['n_still_off_range']}")

    print("\n2/3 Rebuilding segments_with_imd.parquet...")
    seg_out, seg_stats = rebuild_segments_with_imd()
    save_json(seg_stats, OUT_TABLES / "16_segments_with_imd.json")
    print(f"  Segments with IMD: {seg_stats['n_segments_with_imd']}/"
          f"{seg_stats['n_segments']}")
    print(f"  IMD median per segment: "
          f"{seg_stats['imd_mean_per_segment_stats']['median']}")

    print("\n3/3 Diagnosing 100m non-match...")
    diag = diagnose_no_match()
    save_json(diag, OUT_TABLES / "16_no_match_diagnosis.json")
    print(f"  Match rates by threshold: {diag['match_rates_by_threshold']}")

    sections = {
        "IMD parsing fix": dumps_jsonable(imd_stats, indent=2),
        "Segments with IMD (fixed)": dumps_jsonable(seg_stats, indent=2),
        "Accident → segment non-match diagnosis":
            dumps_jsonable(diag, indent=2),
    }
    save_markdown_report(sections, Path("reports/16_phase4_fix.md"))
    print("\nDone. See reports/16_phase4_fix.md.")


if __name__ == "__main__":
    main()