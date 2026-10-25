"""
Script 37a — Audit of administrative geometries (districts, neighbourhoods).

Purpose
-------
Before Phase 8 (Moran) we need to know which administrative boundaries
are available locally and in what form. This script does not download
anything; it reports what exists and what CKAN candidates are relevant.

Checks
------
1. List files in data/raw/ and data/processed/ that may carry admin
   boundaries: *.shp, *.geojson, *.gpkg, *.parquet.
2. For each, if it is a GeoDataFrame, print: CRS, geometry type, n rows,
   columns that look like names/codes of districts or neighbourhoods.
3. Inspect network_callejero.parquet: does it already have district /
   neighbourhood columns, even if only as attributes?
4. Query Madrid CKAN (package_search) for datasets whose titles match
   `distrito`, `barrio`, `límites administrativos`, `division`.
   Print the top hits with their resource formats.

Output
------
reports/tables/37a_admin_geometries.json
reports/37a_admin_geometries.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_TABLES = REPORTS / "tables"

OUT_JSON = REPORTS_TABLES / "37a_admin_geometries.json"
OUT_MD = REPORTS / "37a_admin_geometries.md"

CKAN_BASE = "https://datos.madrid.es/api/3/action"

GEO_EXTS = {".shp", ".geojson", ".gpkg", ".parquet"}
NAME_HINTS = ("distrito", "barrio", "nombre", "name", "cod_", "codigo",
              "district", "neighbo", "denominacion")

QUERIES = (
    "distritos municipales",
    "barrios municipales",
    "limites administrativos",
    "division territorial",
)

log = logging.getLogger("check_admin_geoms")


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------

def list_local_candidates() -> list[Path]:
    found = []
    for base in (DATA_RAW, DATA_PROCESSED):
        if not base.exists():
            continue
        for p in base.iterdir():
            if p.suffix.lower() in GEO_EXTS:
                found.append(p)
    return sorted(found)


def inspect_geo_file(path: Path) -> dict:
    info = {"path": str(path.relative_to(ROOT)), "size_kb": round(path.stat().st_size / 1024, 1)}
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            info["type"] = "parquet (pandas)"
            info["n_rows"] = int(len(df))
            info["columns"] = list(df.columns)[:50]
            info["n_columns"] = int(len(df.columns))
            geom_col = next((c for c in df.columns
                             if c in ("geometry", "geometry_wkt", "geom", "wkt")), None)
            if geom_col:
                sample = df[geom_col].dropna().iloc[0] if df[geom_col].notna().any() else None
                info["geometry_col"] = geom_col
                info["geometry_sample_type"] = type(sample).__name__ if sample is not None else None
                if isinstance(sample, str):
                    info["geometry_first_token"] = sample.split("(")[0][:40]
            else:
                info["geometry_col"] = None
            # Name hints
            info["name_like_columns"] = [
                c for c in df.columns if any(h in c.lower() for h in NAME_HINTS)
            ]
        elif path.suffix.lower() in (".shp", ".gpkg", ".geojson"):
            import geopandas as gpd
            gdf = gpd.read_file(path, rows=10)
            info["type"] = path.suffix.lower()
            info["crs"] = str(gdf.crs)
            info["columns"] = list(gdf.columns)
            info["sample_geometry_types"] = sorted(
                {g.geom_type for g in gdf.geometry if g is not None})
            info["name_like_columns"] = [
                c for c in gdf.columns if any(h in c.lower() for h in NAME_HINTS)
            ]
        else:
            info["type"] = "unknown"
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def inspect_callejero() -> dict:
    p = DATA_PROCESSED / "network_callejero.parquet"
    if not p.exists():
        return {"exists": False}
    df = pd.read_parquet(p)
    # Check columns that look like district / neighbourhood codes
    dist_like = [c for c in df.columns if "distr" in c.lower()]
    bar_like = [c for c in df.columns
                if any(h in c.lower() for h in ("barrio", "neighbo"))]
    return {
        "exists": True,
        "path": str(p.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "district_like_columns": dist_like,
        "neighbourhood_like_columns": bar_like,
        "district_values_sample": (
            df[dist_like[0]].dropna().astype(str).unique()[:10].tolist()
            if dist_like else None
        ),
    }


# ---------------------------------------------------------------------------
# CKAN
# ---------------------------------------------------------------------------

def ckan_search(q: str, rows: int = 5) -> dict:
    try:
        r = requests.get(
            f"{CKAN_BASE}/package_search",
            params={"q": q, "rows": rows},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            return {"query": q, "error": "ckan success=false"}
        hits = []
        for pkg in data["result"]["results"]:
            resources = [
                {"name": res.get("name"), "format": res.get("format"),
                 "url": res.get("url")}
                for res in pkg.get("resources", [])
            ]
            hits.append({
                "id": pkg.get("id"),
                "title": pkg.get("title"),
                "name": pkg.get("name"),
                "n_resources": len(resources),
                "resource_formats": sorted(
                    {res["format"] for res in resources if res.get("format")}
                ),
                "resources_preview": resources[:4],
            })
        return {"query": q, "n_hits": len(hits), "hits": hits}
    except Exception as e:  # noqa: BLE001
        return {"query": q, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_md(results: dict) -> str:
    lines = ["# Phase 8 pre-check · Administrative geometries\n"]
    lines.append("## Local files\n")
    if not results["local_files"]:
        lines.append("_No geographic files (.shp/.geojson/.gpkg/.parquet) "
                     "found in data/raw or data/processed._\n")
    else:
        lines.append("| path | type | n_rows | CRS | geom | name cols |")
        lines.append("| --- | --- | ---: | --- | --- | --- |")
        for f in results["local_files"]:
            lines.append(
                f"| `{f['path']}` | {f.get('type', 'n/a')} | "
                f"{f.get('n_rows', 'n/a')} | {f.get('crs', 'n/a')} | "
                f"{f.get('geometry_sample_type', f.get('sample_geometry_types', 'n/a'))} | "
                f"{f.get('name_like_columns', [])} |"
            )
        lines.append("")

    lines.append("## Callejero (network_callejero.parquet)\n")
    cal = results["callejero"]
    if not cal.get("exists"):
        lines.append("_Not found._\n")
    else:
        lines.append(f"- rows: **{cal['n_rows']:,}**")
        lines.append(f"- columns: `{cal['columns']}`")
        lines.append(f"- district-like: `{cal['district_like_columns']}`")
        lines.append(f"- neighbourhood-like: `{cal['neighbourhood_like_columns']}`")
        if cal.get("district_values_sample"):
            lines.append(f"- sample district values: `{cal['district_values_sample']}`")
        lines.append("")

    lines.append("## CKAN candidates\n")
    for res in results["ckan"]:
        lines.append(f"### Query: `{res['query']}`\n")
        if "error" in res:
            lines.append(f"_Error: {res['error']}_\n")
            continue
        if res["n_hits"] == 0:
            lines.append("_No hits._\n")
            continue
        for h in res["hits"]:
            lines.append(
                f"- **{h['title']}** (`{h['id']}`) — "
                f"{h['n_resources']} resources, formats: {h['resource_formats']}"
            )
            for r in h["resources_preview"]:
                lines.append(f"    - `{r['format']}`: {r['url']}")
        lines.append("")

    lines.append("## Decision checklist for Phase 8\n")
    lines.append("Based on these results, decide:\n")
    lines.append("- **Districts**: use local file if present, else download from "
                 "the CKAN hit that lists a SHP/WFS resource.")
    lines.append("- **Neighbourhoods**: same.")
    lines.append("- **Street segments**: use `network_callejero.parquet` (P1=B) "
                 "regardless.")
    lines.append("- **Hex 250m**: no external geometry needed; generated from "
                 "the Madrid municipal boundary (available in `network_osm` bbox "
                 "and in `network_callejero`).\n")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Listing local geographic files ...")
    paths = list_local_candidates()
    log.info(f"  found {len(paths)} candidates")

    log.info("Inspecting candidates ...")
    files_info = [inspect_geo_file(p) for p in paths]

    log.info("Inspecting network_callejero.parquet ...")
    callejero = inspect_callejero()
    log.info(f"  exists: {callejero.get('exists')}, "
             f"district cols: {callejero.get('district_like_columns')}")

    log.info("Querying CKAN for admin boundaries ...")
    ckan_results = []
    for q in QUERIES:
        log.info(f"  query: {q}")
        ckan_results.append(ckan_search(q, rows=5))

    results = {
        "local_files": files_info,
        "callejero": callejero,
        "ckan": ckan_results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(results), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    # Console summary
    log.info("=" * 72)
    log.info(f"local geo candidates: {len(files_info)}")
    for f in files_info:
        log.info(f"  {f['path']}  ({f.get('n_rows', '?')} rows, "
                 f"{f.get('crs', '?')})")
    log.info(f"callejero: exists={callejero.get('exists')}, "
             f"district_cols={callejero.get('district_like_columns')}, "
             f"barrio_cols={callejero.get('neighbourhood_like_columns')}")
    for res in ckan_results:
        if "error" in res:
            log.info(f"ckan[{res['query']}]: ERROR {res['error']}")
        else:
            log.info(f"ckan[{res['query']}]: {res['n_hits']} hits")


if __name__ == "__main__":
    main()