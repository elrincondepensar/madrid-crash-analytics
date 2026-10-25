"""
Script 40d — Fix units in exposure_hex250m_monthly.

Bug in 40b: `intensity_day = mean × 288` (assumes all 5-min intervals,
incorrect because `periodo_integracion` varies 1-15 min AND `intensidad`
is absolute count per interval, not a rate).

Correct formula:
    intensity_day_mean = total_intensidad_sum / n_days_with_data

Where total_intensidad_sum already is the daily total (sum of counts).

Cost: 3 min (no downloads; reads per_id_*.parquet files).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
PER_ID_DIR = DATA / "exposure_by_id"
SAMPLES_DIR = Path("data/raw/40a_range_headers")
OUT_PARQUET = DATA / "exposure_hex250m_monthly.parquet"
OUT_JSON = Path("reports/tables/40d_fix_units.json")
OUT_MD = Path("reports/40d_fix_units.md")

INTENSITY_CAP = 500_000  # veh/day; physical max for one point

log = logging.getLogger("phase9_40d")


def days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load all per_id files
    log.info("Loading per_id parquets ...")
    per_id_files = sorted(PER_ID_DIR.glob("per_id_*.parquet"))
    if not per_id_files:
        raise SystemExit(f"no per_id files in {PER_ID_DIR}")

    frames = []
    for p in per_id_files:
        df = pd.read_parquet(p)
        frames.append(df)
    per_id_all = pd.concat(frames, ignore_index=True)
    log.info(f"  {len(per_id_all):,} sensor-months across {len(per_id_files)} files")

    # Recompute intensity_day_mean from total sum / n_days_in_month
    log.info("Recomputing intensity_day_mean (sum/days) ...")
    per_id_all["n_days_in_month"] = [
        days_in_month(int(y), int(m))
        for y, m in zip(per_id_all["year"], per_id_all["month"])
    ]
    per_id_all["intensity_day_mean"] = (
        per_id_all["intensity_sum"] / per_id_all["n_days_in_month"]
    )

    # Distribution before and after cap
    before = per_id_all["intensity_day_mean"].describe(
        percentiles=[0.5, 0.9, 0.99, 0.999])
    n_capped = int((per_id_all["intensity_day_mean"] > INTENSITY_CAP).sum())
    per_id_all["is_capped"] = (
        per_id_all["intensity_day_mean"] > INTENSITY_CAP
    )
    per_id_all.loc[per_id_all["is_capped"], "intensity_day_mean"] = INTENSITY_CAP
    log.info(f"  capped {n_capped} sensor-months > {INTENSITY_CAP:,} veh/day")
    log.info(f"  before cap: median={before['50%']:.0f}, "
             f"p99={before['99%']:.0f}, max={before['max']:.0f}")

    # Sanity: compare with 203962 IMD
    log.info("Sanity vs IMD ...")
    imd = pd.read_parquet(DATA / "imd_points.parquet")
    imd = imd[["lon", "lat", "TOTAL_fixed"]].dropna()
    imd = imd[imd["TOTAL_fixed"] > 0]
    imd_pts = gpd.GeoDataFrame(
        imd,
        geometry=gpd.points_from_xy(imd["lon"], imd["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")
    hexes = ms_io.load_geometry_parquet(
        DATA / "moran_hex250m.parquet", expected_crs="EPSG:25830")
    imd_hex = gpd.sjoin(imd_pts, hexes[["hex_id", "geometry"]],
                        how="inner", predicate="within")
    imd_per_hex = imd_hex.groupby("hex_id")["TOTAL_fixed"].median().to_dict()

    # Join sensors to catalog
    log.info("Joining sensors to catalog coordinates ...")
    catalog = pd.read_csv(
        SAMPLES_DIR / "202468_catalog.csv",
        sep=";", encoding="utf-8-sig", on_bad_lines="skip",
    )
    catalog.columns = [c.strip().lower() for c in catalog.columns]
    catalog["id"] = catalog["id"].astype(int)

    merged = per_id_all.merge(
        catalog[["id", "utm_x", "utm_y", "tipo_elem"]],
        on="id", how="inner",
    )
    log.info(f"  matched {len(merged):,} of {len(per_id_all):,}")

    # Assign to hex
    pts = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["utm_x"], merged["utm_y"]),
        crs="EPSG:25830",
    )
    joined = gpd.sjoin(pts, hexes[["hex_id", "geometry"]],
                       how="inner", predicate="within")
    joined = joined.drop_duplicates(subset=["id", "year", "month"],
                                     keep="first")
    log.info(f"  assigned {len(joined):,} sensor-months to hexes")

    # Aggregate per (hex, year, month)
    log.info("Aggregating by (hex, year, month) ...")
    grp = joined.groupby(["hex_id", "year", "month"])
    exposure = grp.agg(
        n_sensors=("id", "nunique"),
        n_sensors_capped=("is_capped", "sum"),
        intensity_day_mean_mean=("intensity_day_mean", "mean"),
        intensity_day_median=("intensity_day_mean", "median"),
        intensity_day_p90=("intensity_day_mean",
                            lambda s: s.quantile(0.9)),
        intensity_day_sum=("intensity_day_mean", "sum"),
        coverage_mean=("coverage", "mean"),
        n_urb=("tipo_elem", lambda s: (s == "URB").sum()),
        n_m30_c30=("tipo_elem", lambda s: s.isin(["M30", "C30"]).sum()),
    ).reset_index()

    log.info(f"  {len(exposure):,} (hex, year, month) rows")

    # Sanity: compare with IMD
    sub = exposure[(exposure["year"] == 2022) & (exposure["month"] == 6)]
    sub = sub[["hex_id", "intensity_day_mean_mean"]].copy()
    sub["imd"] = sub["hex_id"].map(imd_per_hex)
    sub = sub.dropna()
    rho = None
    ratio = None
    if len(sub) >= 10:
        rho, _ = spearmanr(sub["intensity_day_mean_mean"], sub["imd"])
        ratio = float(
            (sub["intensity_day_mean_mean"] / sub["imd"]).median())
    log.info(f"  vs IMD: n_overlap={len(sub)}, rho={rho}, "
             f"median_ratio={ratio}")

    # Compute overall stats
    after = exposure["intensity_day_mean_mean"].describe(
        percentiles=[0.5, 0.9, 0.99])
    log.info(f"  AFTER: median={after['50%']:.0f} veh/day, "
             f"p90={after['90%']:.0f}, p99={after['99%']:.0f}")

    # Save
    exposure.to_parquet(OUT_PARQUET, index=False)
    log.info(f"saved {OUT_PARQUET} ({len(exposure):,} rows)")

    # Per-year medians
    by_year = {}
    for y in sorted(exposure["year"].unique()):
        s = exposure[exposure["year"] == y]
        by_year[int(y)] = {
            "median": float(s["intensity_day_mean_mean"].median()),
            "p90": float(s["intensity_day_mean_mean"].quantile(0.9)),
        }

    result = {
        "n_capped": n_capped,
        "cap_value": INTENSITY_CAP,
        "before_cap": {
            "median": float(before["50%"]),
            "p99": float(before["99%"]),
            "max": float(before["max"]),
        },
        "after_cap": {
            "median": float(after["50%"]),
            "p90": float(after["90%"]),
            "p99": float(after["99%"]),
        },
        "n_rows": int(len(exposure)),
        "n_hexes_with_data": int(exposure["hex_id"].nunique()),
        "vs_imd_check": {
            "n_overlap": int(len(sub)),
            "spearman_rho": float(rho) if rho is not None else None,
            "median_ratio_208627_over_imd":
                float(ratio) if ratio is not None else None,
        },
        "by_year": by_year,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(result), encoding="utf-8")
    log.info(f"saved {OUT_JSON}")

    # Simple markdown
    md = ["# Phase 9 · Iter 2d — Units fix\n\n"]
    md.append("## Cap\n")
    md.append(f"- Sensors capped at {INTENSITY_CAP:,} veh/day: "
              f"{n_capped}\n")
    md.append(f"- Before cap: median={before['50%']:.0f}, "
              f"p99={before['99%']:.0f}, max={before['max']:.0f}\n")
    md.append(f"- After cap: median={after['50%']:.0f}, "
              f"p90={after['90%']:.0f}, p99={after['99%']:.0f}\n\n")
    md.append("## Sanity vs 203962 IMD\n")
    md.append(f"- Overlapping hexes: {len(sub)}\n")
    if rho is not None:
        md.append(f"- Spearman rho: **{rho:.3f}**\n")
        md.append(f"- Median ratio: **{ratio:.2f}**\n")
    md.append("\n## By year\n")
    md.append("| year | median | p90 |\n| --- | ---: | ---: |\n")
    for y, v in sorted(by_year.items()):
        md.append(f"| {y} | {v['median']:.0f} | {v['p90']:.0f} |\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    log.info(f"Done. Median intensity: {after['50%']:.0f} veh/day "
             f"(was {before['50%']:.0f} before cap, "
             f"~{65_679 if False else before['50%']:.0f})")


if __name__ == "__main__":
    main()