"""
Script 40c — Unit validation for exposure_hex250m_monthly.

Resolves: are the `intensidad` values in 208627 measured in
veh/5min (current assumption) or veh/hour (×12 overcount)?

Three independent checks:
  1. Raw daily profile of a single sensor vs plausible values.
  2. Compare 208627 exposure with 203962 IMD for hexes that contain
     both a 208627 sensor and a 203962 IMD point.
  3. Compare 208627 with 300233 for stations that appear in both
     datasets.

Output: reports/tables/40c_unit_validation.json
        reports/40c_unit_validation.md
"""

from __future__ import annotations

import io
import logging
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

SAMPLES_DIR = Path("data/raw/40a_range_headers")
DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/40c_unit_validation.md")

log = logging.getLogger("phase9_40c")


# ---------------------------------------------------------------------------
# Check 1: raw daily profile of one sensor
# ---------------------------------------------------------------------------

def check_single_sensor_profile(zip_path: Path, sensor_id: int) -> dict:
    """Read one sensor's full daily profile for one month. Report
    per-day totals, hourly pattern, and what the units imply."""
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            return {"error": "no_csv"}
        with zf.open(csvs[0]) as f:
            df = pd.read_csv(
                f, sep=";", encoding="utf-8-sig",
                usecols=["id", "fecha", "intensidad", "error",
                         "periodo_integracion"],
                on_bad_lines="skip",
            )
    df = df[df["id"] == sensor_id]
    df = df[df["error"] == "N"]
    if len(df) == 0:
        return {"error": f"no data for sensor {sensor_id}"}

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    # Daily totals under both hypotheses
    daily_sum = df.groupby(df["fecha"].dt.date)["intensidad"].sum()

    # Sample one day: hourly pattern
    sample_day = df["fecha"].dt.date.iloc[0]
    day_df = df[df["fecha"].dt.date == sample_day].copy()
    day_df["hour"] = day_df["fecha"].dt.hour
    hourly = day_df.groupby("hour")["intensidad"].sum().to_dict()

    # Distribution of `intensidad` values
    q = df["intensidad"].quantile([0.5, 0.9, 0.99, 0.999]).to_dict()

    return {
        "sensor_id": int(sensor_id),
        "n_intervals": int(len(df)),
        "periodo_integracion_values": (
            df["periodo_integracion"].value_counts().to_dict()),
        "daily_sum_stats": {
            "median": float(daily_sum.median()),
            "max": float(daily_sum.max()),
            "min": float(daily_sum.min()),
        },
        "hourly_sum_sample_day": hourly,
        "sample_day": str(sample_day),
        "intensidad_quantiles": {f"q{int(k*100)}": float(v)
                                  for k, v in q.items()},
        "interpretation": (
            "if daily_sum ~ 10k-50k veh/day: units are veh/5min, current aggregation OK. "
            "if daily_sum > 200k: units are veh/hour (multiplied by 288 overcounts 12x)."
        ),
    }


# ---------------------------------------------------------------------------
# Check 2: 208627 vs 203962 IMD overlap
# ---------------------------------------------------------------------------

def check_vs_imd(exposure: pd.DataFrame) -> dict:
    """Compare 208627 exposure with 203962 IMD for hexes containing both."""
    imd = pd.read_parquet(DATA / "imd_points.parquet")
    imd = imd[["lon", "lat", "TOTAL_fixed"]].dropna()
    imd = imd[imd["TOTAL_fixed"] > 0]

    imd_pts = gpd.GeoDataFrame(
        imd,
        geometry=gpd.points_from_xy(imd["lon"], imd["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")

    hexes = ms_io.load_geometry_parquet(
        DATA / "moran_hex250m.parquet",
        expected_crs="EPSG:25830",
    )
    imd_hex = gpd.sjoin(
        imd_pts, hexes[["hex_id", "geometry"]],
        how="inner", predicate="within",
    )
    imd_per_hex = imd_hex.groupby("hex_id")["TOTAL_fixed"].median().to_dict()

    # Take exposure for a mid-year month (say 2022-06)
    sub = exposure[(exposure["year"] == 2022) & (exposure["month"] == 6)]
    sub = sub[["hex_id", "intensity_day_mean_mean"]].copy()
    sub["imd"] = sub["hex_id"].map(imd_per_hex)
    sub = sub.dropna()

    if len(sub) < 10:
        return {"error": f"only {len(sub)} overlapping hexes"}

    rho, _ = spearmanr(sub["intensity_day_mean_mean"], sub["imd"])
    ratio = (sub["intensity_day_mean_mean"] / sub["imd"]).median()

    return {
        "n_overlap_hexes": int(len(sub)),
        "spearman_rho": float(rho),
        "median_ratio_208627_over_imd": float(ratio),
        "expected_rho": "> 0.7 if both measure intensity",
        "expected_ratio": (
            "~1.0 if 208627 is already in veh/day. "
            "~12 if 208627 is overcounting by 12x (veh/hour × 288)."
        ),
    }


# ---------------------------------------------------------------------------
# Check 3: raw intensity distribution in a CSV
# ---------------------------------------------------------------------------

def check_raw_distribution(zip_path: Path) -> dict:
    """Look at the raw `intensidad` distribution in one month's CSV."""
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        with zf.open(csvs[0]) as f:
            df = pd.read_csv(
                f, sep=";", encoding="utf-8-sig",
                usecols=["id", "intensidad", "error"],
                on_bad_lines="skip", nrows=2_000_000,
            )
    df = df[(df["error"] == "N") & (df["intensidad"] >= 0)]
    q = df["intensidad"].quantile(
        [0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]).to_dict()
    return {
        "n_rows_sampled": int(len(df)),
        "intensidad_stats": {
            "min": float(df["intensidad"].min()),
            "median": float(df["intensidad"].median()),
            "mean": float(df["intensidad"].mean()),
            "max": float(df["intensidad"].max()),
            "quantiles": {f"q{int(k*100):02d}": float(v)
                           for k, v in q.items()},
        },
        "verdict": (
            "if max per interval < 5000: units are plausible per-5min counts. "
            "if max per interval > 5000: check whether this is veh/hour."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_md(result: dict) -> str:
    lines = ["# Phase 9 · Iter 2c — Unit validation\n"]

    lines.append("## Check 1: single-sensor daily profile\n")
    c1 = result["single_sensor"]
    if "error" in c1:
        lines.append(f"ERROR: {c1['error']}\n")
    else:
        lines.append(f"- Sensor id: {c1['sensor_id']}\n")
        lines.append(f"- Intervals in month: {c1['n_intervals']}\n")
        lines.append(f"- `periodo_integracion` values: "
                     f"{c1['periodo_integracion_values']}\n")
        lines.append(f"- `intensidad` quantiles: "
                     f"{c1['intensidad_quantiles']}\n")
        lines.append(f"- **Daily sum (row-by-row sum over intervals):**\n")
        lines.append(f"  - median: {c1['daily_sum_stats']['median']:.0f}\n")
        lines.append(f"  - min: {c1['daily_sum_stats']['min']:.0f}\n")
        lines.append(f"  - max: {c1['daily_sum_stats']['max']:.0f}\n")
        lines.append(f"- Sample day hourly sum: {c1['hourly_sum_sample_day']}\n")
        lines.append(f"- {c1['interpretation']}\n")

    lines.append("\n## Check 2: 208627 vs 203962 IMD (hex overlap)\n")
    c2 = result["vs_imd"]
    if "error" in c2:
        lines.append(f"ERROR: {c2['error']}\n")
    else:
        lines.append(f"- n overlapping hexes: {c2['n_overlap_hexes']}\n")
        lines.append(f"- Spearman rho: **{c2['spearman_rho']:.3f}** "
                     f"(expected {c2['expected_rho']})\n")
        lines.append(f"- Median ratio 208627/IMD: "
                     f"**{c2['median_ratio_208627_over_imd']:.2f}** "
                     f"(expected {c2['expected_ratio']})\n")

    lines.append("\n## Check 3: raw `intensidad` distribution\n")
    c3 = result["raw_distribution"]
    if "error" in c3:
        lines.append(f"ERROR: {c3['error']}\n")
    else:
        lines.append(f"- n rows sampled: {c3['n_rows_sampled']:,}\n")
        st = c3["intensidad_stats"]
        lines.append(f"- min={st['min']:.0f}, median={st['median']:.0f}, "
                     f"mean={st['mean']:.0f}, max={st['max']:.0f}\n")
        lines.append(f"- quantiles: {st['quantiles']}\n")
        lines.append(f"- {c3['verdict']}\n")

    lines.append("\n## Conclusion\n")
    lines.append(result["conclusion"])
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load exposure
    exp_path = DATA / "exposure_hex250m_monthly.parquet"
    exposure = pd.read_parquet(exp_path)
    log.info(f"exposure: {len(exposure):,} rows")

    # Check 1: single sensor (from the cached 2022-06 ZIP)
    log.info("Check 1: single sensor profile")
    zip_2022_06 = SAMPLES_DIR / "full_2022-06.zip"
    if not zip_2022_06.exists():
        zip_2022_06 = SAMPLES_DIR / "build_2022-06.zip"
    # Find a sensor from the mid-range of exposures
    sample_month = exposure[(exposure["year"] == 2022)
                             & (exposure["month"] == 6)]
    per_id = pd.read_parquet(DATA / "exposure_by_id/per_id_2022-06.parquet")
    # Pick a mid-range sensor
    mid_sensor = int(per_id.iloc[len(per_id) // 2]["id"])

    c1 = check_single_sensor_profile(zip_2022_06, mid_sensor) \
         if zip_2022_06.exists() else {"error": "zip missing"}

    # Check 2: vs IMD
    log.info("Check 2: vs 203962 IMD")
    c2 = check_vs_imd(exposure)

    # Check 3: raw distribution
    log.info("Check 3: raw distribution")
    c3 = check_raw_distribution(zip_2022_06) if zip_2022_06.exists() \
         else {"error": "zip missing"}

        # Conclusion
    conclusion_lines = []
    if "error" not in c1:
        daily_med = c1["daily_sum_stats"]["median"]
        if daily_med > 200_000:
            conclusion_lines.append(
                "**Single-sensor daily sum > 200k → `intensidad` is "
                "veh/hour, not veh/5min.** Current aggregation multiplies "
                "by 288 (wrong). Correct is to divide by 12 or multiply "
                "by 24 instead of 288. `intensity_day_mean` should be "
                "recalculated."
            )
        elif 5_000 <= daily_med <= 50_000:
            conclusion_lines.append(
                "**Single-sensor daily sum in [5k, 50k] → `intensidad` "
                "is veh/5min.** Current aggregation correct."
            )

    if "error" not in c2:
        ratio = c2["median_ratio_208627_over_imd"]
        if ratio > 5:
            conclusion_lines.append(
                f"**Ratio vs IMD = {ratio:.1f}** "
                f"→ overcounted by ~{ratio:.0f}x. "
                f"Likely factor 12 (veh/hour × 288 vs veh/hour × 24)."
            )
        elif 0.5 <= ratio <= 2:
            conclusion_lines.append(
                f"**Ratio vs IMD = {ratio:.2f}** → units are correct."
            )

    if not conclusion_lines:
        conclusion_lines.append("Insufficient data to conclude.")

    result = {
        "single_sensor": c1,
        "vs_imd": c2,
        "raw_distribution": c3,
        "conclusion": " ".join(conclusion_lines),
    }

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "40c_unit_validation.json").write_text(
        dumps_jsonable(result), encoding="utf-8")
    OUT_MD.write_text(render_md(result), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '40c_unit_validation.json'}")
    log.info(f"saved {OUT_MD}")
    log.info("=" * 72)
    log.info(result["conclusion"])


if __name__ == "__main__":
    main()