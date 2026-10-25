"""
Script 40b — Phase 9 iter 2b: build exposure_hex250m_monthly.

Processes 84 months (2019-01 to 2025-12) of 208627 historical traffic
and produces per-hex250m monthly exposure metrics.

Method
------
1. For each month ZIP:
   - Read CSV (utf-8-sig, on_bad_lines='skip', chunked)
   - Keep rows with error=='N' AND intensidad>=0
   - Aggregate per sensor id: sum, count, sum_sq of `intensidad`
     (5-min vehicle counts)
2. Save per-month per-id parquet (~10 KB each) → resumable
3. Load 202468 catalog: id → (utm_x, utm_y, tipo_elem)
4. Assign each id to its hex250m (sjoin within)
5. Aggregate per (hex, month):
   - n_sensors, n_sensors_with_data
   - intensity_day_mean: mean over sensors of (intensity_sum/count × 288)
   - intensity_day_median
   - intensity_day_p90
   - intensity_day_sum
   - coverage_mean (avg data availability)
6. Save exposure_hex250m_monthly.parquet.

Cost
----
~84 months × 100 s / 3 parallel workers = ~50 min total.
ZIPs already cached in data/raw/40a_range_headers/ for 3 months;
the remaining 81 will be downloaded (~6.5 GB total).
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

CKAN = "https://datos.madrid.es/api/3/action"
UA = "madrid-road-safety/phase9/40b"
TIMEOUT = 240
N_JOBS = 8

SAMPLES_DIR = Path("data/raw/40a_range_headers")
PER_ID_DIR = Path("data/processed/exposure_by_id")
OUT_DATA = Path("data/processed")
OUT_JSON = Path("reports/tables/40b_exposure_build.json")
OUT_MD = Path("reports/40b_exposure_build.md")

PER_ID_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("phase9_40b")


# ---------------------------------------------------------------------------
# Tolerant month regex (from 40a-bis)
# ---------------------------------------------------------------------------

import re

_MONTH_RE = re.compile(r"(?P<mm>\d{1,2})[_\-\s]?(?P<yyyy>20\d{2})",
                       re.IGNORECASE)
_ANNUAL_RE = re.compile(r"^(?P<yyyy>20\d{2})$")
_SPANISH = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
            "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
            "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


def infer_month_tolerant(filename: str) -> str | None:
    if not filename:
        return None
    stem = filename.strip().rstrip("/").replace(".csv", "").replace(".CSV", "")
    m = _ANNUAL_RE.match(stem)
    if m:
        return f"{m.group('yyyy')}-ANNUAL"
    m = _MONTH_RE.search(stem)
    if m:
        mm = int(m.group("mm"))
        if 1 <= mm <= 12:
            return f"{m.group('yyyy')}-{mm:02d}"
    parts = stem.lower().split()
    if len(parts) >= 2 and parts[0].strip() in _SPANISH \
       and parts[-1].isdigit():
        return f"{parts[-1]}-{_SPANISH[parts[0].strip()]:02d}"
    return None


# ---------------------------------------------------------------------------
# Load 208627 catalog and build month → resource index map
# ---------------------------------------------------------------------------

def load_resource_map() -> dict:
    """Returns {YYYY-MM: {"url": ..., "resource_index": ...}}."""
    log.info("Loading 208627 resource list from CKAN")
    r = requests.get(f"{CKAN}/package_show",
                     params={"id": "208627-0-transporte-ptomedida-historico"},
                     headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()["result"]
    resources = [res for res in d.get("resources", [])
                 if (res.get("format") or "").upper() == "ZIP"]

    # Load already-cached headers from 40a
    headers_file = Path("reports/tables/40a_structure_audit.json")
    if headers_file.exists():
        raw = json.loads(headers_file.read_text())
        raw_map = raw.get("raw_resource_map", [])
    else:
        raw_map = []

    by_month = {}
    for entry in raw_map:
        idx = entry["resource_index"]
        fname = entry.get("filename", "")
        month = infer_month_tolerant(fname)
        if not month or month.endswith("-ANNUAL"):
            continue
        if idx < len(resources):
            by_month[month] = {
                "url": resources[idx]["url"],
                "resource_index": idx,
            }
    log.info(f"  mapped {len(by_month)} months")
    return by_month


# ---------------------------------------------------------------------------
# Process one month
# ---------------------------------------------------------------------------

def process_one_month(month: str, url: str,
                      cache_zip: Path,
                      out_per_id: Path) -> dict:
    """Download, parse, aggregate per-id, save per-id parquet."""
    if out_per_id.exists() and out_per_id.stat().st_size > 0:
        return {"month": month, "status": "cached",
                "n_ids": None, "elapsed_s": 0}

    t0 = time.time()

    # Download ZIP
    if not cache_zip.exists() or cache_zip.stat().st_size == 0:
        try:
            r = requests.get(url, headers={"User-Agent": UA},
                             timeout=TIMEOUT, stream=True)
            r.raise_for_status()
            cache_zip.parent.mkdir(parents=True, exist_ok=True)
            cache_zip.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            return {"month": month, "status": "download_failed",
                    "error": str(e), "elapsed_s": time.time() - t0}

    # Parse ZIP with chunked reader
    try:
        with zipfile.ZipFile(cache_zip) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                return {"month": month, "status": "no_csv_inside",
                        "elapsed_s": time.time() - t0}
            csv_name = csvs[0]

            # Accumulate per-id stats
            accum = {}  # id -> [sum, count, sum_sq]

            with zf.open(csv_name) as f:
                reader = pd.read_csv(
                    f, sep=";", encoding="utf-8-sig",
                    usecols=["id", "intensidad", "error", "tipo_elem"],
                    dtype={"id": "Int64", "intensidad": "Int64",
                           "error": "string", "tipo_elem": "string"},
                    chunksize=500_000,
                    on_bad_lines="skip",
                )
                for chunk in reader:
                    # Filter valid
                    mask = (
                        (chunk["error"] == "N")
                        & chunk["intensidad"].notna()
                        & (chunk["intensidad"] >= 0)
                    )
                    sub = chunk.loc[mask, ["id", "intensidad"]]
                    if len(sub) == 0:
                        continue
                    grp = sub.groupby("id")["intensidad"]
                    agg = grp.agg(["sum", "count",
                                   lambda s: (s.astype("float64") ** 2).sum()])
                    agg.columns = ["sum", "count", "sum_sq"]
                    for idx, row in agg.iterrows():
                        if idx in accum:
                            accum[idx][0] += row["sum"]
                            accum[idx][1] += row["count"]
                            accum[idx][2] += row["sum_sq"]
                        else:
                            accum[idx] = [row["sum"], row["count"],
                                          row["sum_sq"]]
    except Exception as e:  # noqa: BLE001
        return {"month": month, "status": "parse_failed",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": time.time() - t0}

    if not accum:
        return {"month": month, "status": "no_data",
                "elapsed_s": time.time() - t0}

    # Build per-id DataFrame
    rows = []
    for sensor_id, (s, c, sq) in accum.items():
        if c <= 0:
            continue
        mean = s / c
        var = max(0.0, sq / c - mean ** 2)
        std = var ** 0.5
        # Mean per day assuming full 24h coverage (288 intervals/day)
        # (extrapolating from available intervals to full day)
        mean_day = mean * 288
        rows.append({
            "id": int(sensor_id),
            "n_valid_intervals": int(c),
            "intensity_sum": float(s),
            "intensity_mean_5min": float(mean),
            "intensity_std_5min": float(std),
            "intensity_mean_day": float(mean_day),
        })
    df = pd.DataFrame(rows)
    df["year"] = int(month[:4])
    df["month"] = int(month[5:7])
    df["coverage"] = df["n_valid_intervals"] / (31 * 288)  # approx

    out_per_id.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_per_id, index=False)

    return {"month": month, "status": "ok",
            "n_ids": int(len(df)),
            "elapsed_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Aggregate per id → hex
# ---------------------------------------------------------------------------

def aggregate_to_hex(per_id_all: pd.DataFrame,
                     catalog: pd.DataFrame,
                     hexes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assign each sensor id to a hex, then aggregate monthly metrics."""
    log.info("Joining sensors to catalog coordinates")
    cat_slim = catalog[["id", "utm_x", "utm_y", "tipo_elem"]].copy()
    cat_slim["id"] = cat_slim["id"].astype(int)

    merged = per_id_all.merge(cat_slim, on="id", how="inner")
    log.info(f"  matched {len(merged):,} of {len(per_id_all):,} sensor-months")

    # Build point GeoDataFrame
    pts = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["utm_x"], merged["utm_y"]),
        crs="EPSG:25830",
    )

    # Assign to hex (inner join; sensors outside Madrid are dropped)
    log.info("Spatial join sensors → hex250m")
    joined = gpd.sjoin(pts, hexes[["hex_id", "geometry"]],
                       how="inner", predicate="within")
    # If a point falls in two hexes (boundary), keep first
    joined = joined.drop_duplicates(subset=["id", "year", "month"],
                                     keep="first")
    log.info(f"  assigned {len(joined):,} sensor-months to hexes")

    # Aggregate per (hex_id, year, month)
    log.info("Aggregating by (hex, year, month)")
    grouped = joined.groupby(["hex_id", "year", "month"])
    out = grouped.agg(
        n_sensors=("id", "nunique"),
        intensity_day_mean_mean=("intensity_mean_day", "mean"),
        intensity_day_median=("intensity_mean_day", "median"),
        intensity_day_p90=("intensity_mean_day", lambda s: s.quantile(0.9)
                            if len(s) >= 1 else np.nan),
        intensity_day_sum=("intensity_mean_day", "sum"),
        coverage_mean=("coverage", "mean"),
        n_urb=("tipo_elem", lambda s: (s == "URB").sum()),
        n_m30_c30=("tipo_elem", lambda s: s.isin(["M30", "C30"]).sum()),
    ).reset_index()

    log.info(f"  {len(out):,} (hex, year, month) combinations")
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_md(result: dict) -> str:
    lines = ["# Phase 9 · Iter 2b — Exposure build\n"]
    lines.append(f"**Months processed:** {result['n_processed']}/84\n\n")
    lines.append("## Status per year\n")
    lines.append("| year | months | ids (median) | elapsed (s) |\n")
    lines.append("| --- | ---: | ---: | ---: |\n")
    for y, info in sorted(result["per_year"].items()):
        lines.append(f"| {y} | {info['n_months']} | "
                     f"{info['median_ids']:.0f} | "
                     f"{info['total_elapsed_s']:.0f} |\n")

    lines.append("\n## Output\n")
    lines.append(f"- `exposure_hex250m_monthly.parquet`: "
                 f"{result['out_rows']:,} rows\n")
    lines.append(f"- Columns: {result['out_columns']}\n")

    lines.append("\n## Sanity check\n")
    s = result["sanity"]
    lines.append(f"- hexes with any data: {s['hexes_with_data']:,}\n")
    lines.append(f"- median sensors per (hex, month): "
                 f"{s['median_sensors_per_hex_month']:.1f}\n")
    lines.append(f"- median intensity_day_mean: "
                 f"{s['median_intensity_day']:.0f} veh/day\n")
    lines.append(f"- p90 intensity_day_mean: "
                 f"{s['p90_intensity_day']:.0f} veh/day\n")
    lines.append(f"- max intensity_day_mean: "
                 f"{s['max_intensity_day']:.0f} veh/day\n")

    if "national_median_by_year" in s:
        lines.append("\n### Mean intensity_day by year\n")
        lines.append("| year | median_of_hexes | p90_of_hexes |\n")
        lines.append("| --- | ---: | ---: |\n")
        for y, vals in sorted(s["national_median_by_year"].items()):
            lines.append(f"| {y} | {vals['median']:.0f} | "
                         f"{vals['p90']:.0f} |\n")

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

    # Load resource map
    month_map = load_resource_map()
    target_months = sorted(month_map.keys())
    # Filter to 2019-2025
    target_months = [m for m in target_months
                     if "2019-01" <= m <= "2025-12"]
    log.info(f"Target months 2019-2025: {len(target_months)}")

    # Process all months (parallel)
    log.info(f"Processing {len(target_months)} months with n_jobs={N_JOBS}...")
    t_start = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_one_month)(
            month,
            month_map[month]["url"],
            SAMPLES_DIR / f"build_{month}.zip",
            PER_ID_DIR / f"per_id_{month}.parquet",
        )
        for month in target_months
    )
    log.info(f"All months processed in {time.time() - t_start:.0f}s")

    # Aggregate per year
    per_year = {}
    for res in results:
        year = res["month"][:4]
        per_year.setdefault(year, {
            "n_months": 0, "total_elapsed_s": 0.0, "ids_list": [],
        })
        per_year[year]["n_months"] += 1
        per_year[year]["total_elapsed_s"] += res.get("elapsed_s", 0)
        if res.get("n_ids"):
            per_year[year]["ids_list"].append(res["n_ids"])

    for y, info in per_year.items():
        ids = info.pop("ids_list")
        info["median_ids"] = float(np.median(ids)) if ids else 0

    # Load per-id parquet files
    log.info("Loading all per-id parquets")
    per_id_files = sorted(PER_ID_DIR.glob("per_id_*.parquet"))
    per_id_all = pd.concat(
        [pd.read_parquet(p) for p in per_id_files],
        ignore_index=True,
    )
    log.info(f"  {len(per_id_all):,} sensor-months total")

    # Load catalog
    catalog = pd.read_csv(
        SAMPLES_DIR / "202468_catalog.csv",
        sep=";", encoding="utf-8-sig",
        on_bad_lines="skip",
    )
    catalog.columns = [c.strip().lower() for c in catalog.columns]
    # rename to expected names
    rename = {"tipo_elem": "tipo_elem", "id": "id",
              "utm_x": "utm_x", "utm_y": "utm_y"}
    catalog = catalog.rename(columns=rename)
    log.info(f"  catalog: {len(catalog)} sensors")

    # Load hex grid (from Phase 8)
    log.info("Loading hex250m grid")
    hexes = ms_io.load_geometry_parquet(
        Path("data/processed/moran_hex250m.parquet"),
        expected_crs="EPSG:25830",
    )
    log.info(f"  {len(hexes)} hexes")

    # Assign and aggregate
    exposure = aggregate_to_hex(per_id_all, catalog, hexes)

    # Save final
    out_path = OUT_DATA / "exposure_hex250m_monthly.parquet"
    exposure.to_parquet(out_path, index=False)
    log.info(f"saved {out_path} ({len(exposure):,} rows, "
             f"{len(exposure.columns)} cols)")

    # Sanity
    hexes_with_data = exposure["hex_id"].nunique()
    median_sensors = float(exposure["n_sensors"].median())
    median_int = float(exposure["intensity_day_mean_mean"].median())
    p90_int = float(exposure["intensity_day_mean_mean"].quantile(0.9))
    max_int = float(exposure["intensity_day_mean_mean"].max())

    by_year = {}
    for y in sorted(exposure["year"].unique()):
        sub = exposure[exposure["year"] == y]
        by_year[int(y)] = {
            "median": float(sub["intensity_day_mean_mean"].median()),
            "p90": float(sub["intensity_day_mean_mean"].quantile(0.9)),
        }

    sanity = {
        "hexes_with_data": int(hexes_with_data),
        "median_sensors_per_hex_month": median_sensors,
        "median_intensity_day": median_int,
        "p90_intensity_day": p90_int,
        "max_intensity_day": max_int,
        "national_median_by_year": by_year,
    }

    result = {
        "n_processed": sum(1 for r in results if r["status"] in ("ok", "cached")),
        "n_total": len(target_months),
        "per_year": per_year,
        "out_rows": int(len(exposure)),
        "out_columns": list(exposure.columns),
        "sanity": sanity,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(result), encoding="utf-8")
    log.info(f"saved {OUT_JSON}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(result), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    log.info(f"Processed: {result['n_processed']}/{result['n_total']}")
    log.info(f"Output: {result['out_rows']:,} hex×month rows, "
             f"{len(result['out_columns'])} cols")
    log.info(f"Sanity: {hexes_with_data:,} hexes with data, "
             f"median intensity={median_int:.0f} veh/day")


if __name__ == "__main__":
    main()