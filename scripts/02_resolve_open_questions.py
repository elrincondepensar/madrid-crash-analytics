"""
Phase 1 – Iteration 3: resolve remaining open questions before closing Phase 1.

Run from the repository root:
    python scripts/02_resolve_open_questions.py

Outputs go to reports/tables/02_* and reports/02_open_questions.md
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from madrid_safety.data_audit import save_json, save_markdown_report

TIMEOUT = 90
USER_AGENT = "madrid-road-safety-audit/0.3"
CKAN_API = "https://datos.madrid.es/api/3/action"

OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Robust CSV reader for Madrid open data
# ---------------------------------------------------------------------------
def read_madrid_csv(url: str, nrows: int | None = None,
                    usecols: list[str] | None = None,
                    chunk_size: int | None = None) -> pd.DataFrame:
    """
    Read a CSV from datos.madrid.es trying UTF-8-BOM first, then latin-1.
    Handles ';' separator.
    """
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    content = resp.content

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(content),
                sep=";",
                encoding=encoding,
                nrows=nrows,
                usecols=usecols,
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {url}")


def read_column_extent(url: str, column: str) -> dict[str, Any]:
    """Read a single column and return min/max/len without loading the rest."""
    df = read_madrid_csv(url, usecols=[column])
    s = df[column].astype(str).str.strip()
    s = s[s != ""]
    return {
        "n_rows": int(len(df)),
        "n_non_empty": int(len(s)),
        "min": str(s.min()),
        "max": str(s.max()),
    }


# ---------------------------------------------------------------------------
# 1. Accidents: schema + year coverage per resource
# ---------------------------------------------------------------------------
def ckan_package_show(dataset_id: str) -> dict[str, Any]:
    url = f"{CKAN_API}/package_show?id={dataset_id}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        raise RuntimeError(payload)
    return payload["result"]


def ckan_package_search(query: str, rows: int = 20) -> list[dict[str, Any]]:
    url = f"{CKAN_API}/package_search?q={requests.utils.quote(query)}&rows={rows}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        return []
    return payload["result"].get("results", [])


def audit_accidents() -> dict[str, Any]:
    ds = ckan_package_show("300228-0-accidentes-trafico-detalle")
    report = {"resources": []}
    for res in ds.get("resources", []):
        url = res.get("url", "")
        if not (url.lower().endswith(".csv") or "csv" in res.get("format", "").lower()):
            continue
        entry = {"name": res.get("name", ""), "url": url}
        try:
            head = read_madrid_csv(url, nrows=2)
            cols = [c.lstrip("\ufeff") for c in head.columns]
            entry["has_num_expediente"] = "num_expediente" in cols
            entry["has_coords"] = (
                "coordenada_x_utm" in cols and "coordenada_y_utm" in cols
            )
            date_col = "fecha" if "fecha" in cols else ("FECHA" if "FECHA" in cols else None)
            if date_col:
                entry["date_extent"] = read_column_extent(url, date_col)
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
        report["resources"].append(entry)
    return report


# ---------------------------------------------------------------------------
# 2. Permanent counts: classify resources + coverage of measurement files
# ---------------------------------------------------------------------------
def audit_permanent_counts() -> dict[str, Any]:
    ds = ckan_package_show("300233-0-aforo-trafico-permanentes")
    report = {"n_resources": 0, "metadata": [], "measurements": [], "unreadable": []}
    for res in ds.get("resources", []):
        url = res.get("url", "")
        if not (url.lower().endswith(".csv") or "csv" in res.get("format", "").lower()):
            continue
        report["n_resources"] += 1
        entry = {
            "name": res.get("name", ""),
            "url": url,
            "last_modified": res.get("last_modified"),
        }
        try:
            head = read_madrid_csv(url, nrows=2)
            cols = [c.lstrip("\ufeff") for c in head.columns]
            if any(c.lower() in ("estación", "estacion") for c in cols) and \
               any(c.lower() in ("latitud", "longitud") for c in cols):
                # station metadata
                stations = read_madrid_csv(url)
                stations.columns = [c.lstrip("\ufeff") for c in stations.columns]
                entry["n_stations"] = int(len(stations))
                entry["columns"] = list(stations.columns)
                report["metadata"].append(entry)
            elif any(c.lower() == "fdia" for c in cols):
                fd_extent = read_column_extent(url, "FDIA")
                entry["date_extent"] = fd_extent
                report["measurements"].append(entry)
            else:
                entry["columns"] = cols
                report["unreadable"].append(entry)
        except Exception as exc:
            entry["error"] = str(exc)
            report["unreadable"].append(entry)
    return report


# ---------------------------------------------------------------------------
# 3. Traffic history: try alternatives
# ---------------------------------------------------------------------------
def audit_traffic_history() -> dict[str, Any]:
    report = {"attempts": []}

    # (a) Direct package_show
    try:
        ds = ckan_package_show("208627-0-transporte-ptomedida-historico")
        report["attempts"].append({
            "method": "package_show(208627-0-...)",
            "n_resources": len(ds.get("resources", [])),
        })
    except Exception as exc:
        report["attempts"].append({
            "method": "package_show(208627-0-...)",
            "error": str(exc),
        })

    # (b) package_search
    try:
        results = ckan_package_search("ptomedida OR \"punto de medida\" OR transporte-historico")
        report["attempts"].append({
            "method": "package_search",
            "n_results": len(results),
            "top_results": [
                {"name": r.get("name"), "title": r.get("title"),
                 "n_resources": len(r.get("resources", []))}
                for r in results[:10]
            ],
        })
    except Exception as exc:
        report["attempts"].append({
            "method": "package_search",
            "error": str(exc),
        })

    # (c) Scrape the dataset HTML for resource links
    try:
        page = "https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico"
        r = requests.get(page, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        links = re.findall(r'href="([^"]+\.(?:csv|zip|xlsx|json)[^"]*)"', r.text)
        report["attempts"].append({
            "method": "html_scrape",
            "status_code": r.status_code,
            "links_found": sorted(set(links))[:20],
        })
    except Exception as exc:
        report["attempts"].append({
            "method": "html_scrape",
            "error": str(exc),
        })

    return report


# ---------------------------------------------------------------------------
# 4. Road network geometry: search alternatives
# ---------------------------------------------------------------------------
def audit_road_network() -> dict[str, Any]:
    report = {"ckan_search": [], "notes": []}

    # Search in CKAN for likely candidates
    for query in ("red viaria", "callejero geometria", "cartomadrid", "WFS", "vial wkt"):
        try:
            results = ckan_package_search(query, rows=5)
            report["ckan_search"].append({
                "query": query,
                "n_results": len(results),
                "titles": [r.get("title", r.get("name")) for r in results[:5]],
            })
        except Exception as exc:
            report["ckan_search"].append({"query": query, "error": str(exc)})

    # Confirm the official callejero has no segment geometry
    try:
        ds = ckan_package_show("213605-0-callejero-oficial-madrid")
        geom_columns_by_resource = {}
        for res in ds.get("resources", []):
            url = res.get("url", "")
            if not (url.lower().endswith(".csv") or "csv" in res.get("format", "").lower()):
                continue
            head = read_madrid_csv(url, nrows=2)
            cols = [c.lstrip("\ufeff").lower() for c in head.columns]
            geom_columns_by_resource[res.get("name", "")] = {
                "has_wkt": any("wkt" in c or "geometry" in c for c in cols),
                "has_utm_point": any(c.startswith("utmx") or c == "utmx_ed" for c in cols),
                "has_lon_lat": "longitud" in cols and "latitud" in cols,
            }
        report["callejero_geometry_check"] = geom_columns_by_resource
        report["notes"].append(
            "El callejero solo expone puntos de dirección (UTM/lon-lat) y viales, "
            "no segmentos con geometría. Para KDE/Moran sobre red hace falta otra fuente."
        )
    except Exception as exc:
        report["callejero_geometry_check"] = {"error": str(exc)}

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("1/4 Auditing accidents (year coverage + schema)...")
    accidents = audit_accidents()
    save_json(accidents, OUT_DIR / "02_accidents_coverage.json")

    print("2/4 Auditing permanent counts (resource classification)...")
    aforos = audit_permanent_counts()
    save_json(aforos, OUT_DIR / "02_permanent_counts.json")

    print("3/4 Auditing traffic history alternatives...")
    history = audit_traffic_history()
    save_json(history, OUT_DIR / "02_traffic_history.json")

    print("4/4 Auditing road network alternatives...")
    road = audit_road_network()
    save_json(road, OUT_DIR / "02_road_network.json")

    # Markdown summary
    sections = {
        "Accidents coverage": "\n".join(
            f"- `{r['name']}`: schema={'nuevo' if r.get('has_num_expediente') else 'antiguo'}"
            f", coords={r.get('has_coords')}, dates={r.get('date_extent')}"
            for r in accidents.get("resources", [])
        ),
        "Permanent counts": (
            f"- Total CSV resources: {aforos.get('n_resources')}\n"
            f"- Station metadata files: {len(aforos.get('metadata', []))}\n"
            f"- Measurement files: {len(aforos.get('measurements', []))}\n"
            f"- Unclassified: {len(aforos.get('unreadable', []))}\n"
        ),
        "Traffic history": json.dumps(history.get("attempts", []), indent=2, ensure_ascii=False),
        "Road network": json.dumps(road, indent=2, ensure_ascii=False),
    }
    save_markdown_report(sections, Path("reports/02_open_questions.md"))
    print("\nDone. See reports/02_open_questions.md and reports/tables/02_*.json")


if __name__ == "__main__":
    main()