"""
Phase 4 · Iteration 1 — Diagnose road-network sources before committing.

Four targeted diagnostics:

  1. 208223 KMLs (INTENSIDADES.kml, INTENS_POLIG.kml):
     read raw, inspect namespaces, count tags, look for NetworkLink/Folder,
     detect <ExtendedData>. Determine why ElementTree finds 0 placemarks.

  2. 203962 KMZ doc.kml:
     re-parse with a namespace-tolerant reader. Extract the actual contents
     of <description>, <ExtendedData>, <coordinates>. Confirm geometry type
     and whether IDs / IMD values are inside each Placemark.

  3. 203962 XLSX:
     open with openpyxl, list sheets, print first rows of each. Identify
     column names, look for trama/tramo/segmento/ID columns.

  4. Additional CKAN search for road-network datasets with geometry.

Run:
    pip install openpyxl lxml
    python scripts/13_phase4_diagnose_network.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

TIMEOUT = 120
UA = "madrid-road-safety/phase4/0.1"
CKAN = "https://datos.madrid.es/api/3/action"
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)


def download(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# 1. 208223 KMLs — raw inspection
# ---------------------------------------------------------------------------
KML_TAGS = ["Placemark", "Document", "Folder", "NetworkLink",
            "LineString", "Point", "Polygon", "coordinates",
            "ExtendedData", "Data name="]


def inspect_kml_raw(name: str, url: str) -> dict[str, Any]:
    out = {"name": name, "url": url}
    try:
        content = download(url)
    except Exception as exc:
        out["error"] = str(exc)
        return out
    out["size_bytes"] = len(content)

    # First 3000 bytes as string (may not be utf-8; fall back to latin-1)
    try:
        head = content[:3000].decode("utf-8", errors="replace")
    except Exception:
        head = content[:3000].decode("latin-1", errors="replace")
    out["head_first_2000_chars"] = head[:2000]

    # Namespace declarations
    ns_match = re.findall(r'xmlns(?::(\w+))?="([^"]+)"', head)
    out["namespaces"] = [{"prefix": p or "(default)", "uri": u} for p, u in ns_match]

    # Element names that appear
    for tag in KML_TAGS:
        out[f"n_{tag.replace(' ', '_').replace('=', '_')}"] = content.count(
            tag.encode("utf-8"))
    # Also look for gx: prefixed tags
    for tag in ["gx:Track", "gx:coord", "gx:MultiTrack"]:
        out[f"n_{tag.replace(':', '_')}"] = content.count(tag.encode("utf-8"))
    return out


# ---------------------------------------------------------------------------
# 2. 203962 KMZ — tolerant parse with lxml, full detail on a sample
# ---------------------------------------------------------------------------
def parse_kmz_detailed(url: str, n_samples: int = 3) -> dict[str, Any]:
    from lxml import etree

    out = {"url": url}
    try:
        content = download(url)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            out["n_files"] = len(kml_names)
            out["kml_names"] = kml_names
            if not kml_names:
                return out
            kml_bytes = z.read(kml_names[0])
    except Exception as exc:
        out["error"] = str(exc)
        return out
    out["kml_size_bytes"] = len(kml_bytes)

    try:
        root = etree.fromstring(kml_bytes)
    except Exception as exc:
        out["parse_error"] = str(exc)
        out["head"] = kml_bytes[:2000].decode("utf-8", errors="replace")
        return out

    # Namespace map
    out["root_tag"] = root.tag
    out["nsmap"] = {k or "(default)": v for k, v in (root.nsmap or {}).items()}

    # Count elements by localname
    counts: dict[str, int] = {}
    for el in root.iter():
        local = etree.QName(el).localname
        counts[local] = counts.get(local, 0) + 1
    out["element_counts"] = dict(sorted(counts.items(),
                                        key=lambda kv: -kv[1])[:30])

    # Sample Placemarks
    samples = []
    pm_iter = root.iter("{*}Placemark")
    for i, pm in enumerate(pm_iter):
        if i >= n_samples:
            break
        s: dict[str, Any] = {}
        # Name
        name_el = pm.find("{*}name")
        s["name"] = name_el.text if name_el is not None else None
        # Description
        desc_el = pm.find("{*}description")
        s["description"] = (desc_el.text[:500]
                            if desc_el is not None and desc_el.text else None)
        # ExtendedData
        ext = pm.find("{*}ExtendedData")
        if ext is not None:
            data_items = []
            for d in ext.iter("{*}Data"):
                k = d.get("name", "")
                v_el = d.find("{*}value")
                v = v_el.text if v_el is not None else None
                data_items.append({"key": k, "value": v})
            s["extended_data"] = data_items
        # SchemaData
        sd = pm.find("{*}SchemaData")
        if sd is not None:
            sd_items = []
            for d in sd.iter("{*}SimpleData"):
                sd_items.append({"name": d.get("name", ""), "value": d.text})
            s["schema_data"] = sd_items
        # Geometry
        geom_tags = []
        for g in pm.iter():
            local = etree.QName(g).localname
            if local in ("Point", "LineString", "Polygon", "MultiGeometry",
                         "Track", "MultiTrack"):
                geom_tags.append(local)
        s["geometry_tags"] = geom_tags
        # Coordinates anywhere in pm
        coords_el = pm.find(".//{*}coordinates")
        if coords_el is not None and coords_el.text:
            coords = coords_el.text.strip().split()
            s["n_coordinate_tuples"] = len(coords)
            s["first_coord"] = coords[0] if coords else None
        else:
            s["n_coordinate_tuples"] = 0
        samples.append(s)
    out["samples"] = samples
    return out


# ---------------------------------------------------------------------------
# 3. 203962 XLSX — open with openpyxl
# ---------------------------------------------------------------------------
def inspect_xlsx(url: str) -> dict[str, Any]:
    out = {"url": url}
    try:
        content = download(url)
    except Exception as exc:
        out["error"] = str(exc)
        return out
    try:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:
        out["open_error"] = str(exc)
        return out
    out["sheet_names"] = wb.sheetnames
    sheets = {}
    for sh in wb.sheetnames:
        ws = wb[sh]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            rows.append(list(row))
            if i >= 4:
                break
        sheets[sh] = {
            "n_rows_sampled": len(rows),
            "first_rows": rows,
        }
    out["sheets"] = sheets
    wb.close()
    return out


# ---------------------------------------------------------------------------
# 4. Additional CKAN search
# ---------------------------------------------------------------------------
def search_ckan(queries: list[str], rows: int = 10) -> list[dict[str, Any]]:
    results = []
    for q in queries:
        try:
            url = f"{CKAN}/package_search?q={requests.utils.quote(q)}&rows={rows}"
            r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                results.append({"query": q, "error": "ckan success=false"})
                continue
            hits = data["result"].get("results", [])
            results.append({
                "query": q,
                "n_hits": len(hits),
                "hits": [{
                    "name": h.get("name"),
                    "title": h.get("title"),
                    "n_resources": len(h.get("resources", [])),
                    "formats": sorted(set(
                        (res.get("format") or "").upper()
                        for res in h.get("resources", [])
                    )),
                } for h in hits],
            })
        except Exception as exc:
            results.append({"query": q, "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("1/4 Inspecting 208223 KMLs...")
    kml_sources = {
        "intensidades": "https://informo.madrid.es/informo/tmadrid/INTENSIDADES.kml",
        "intens_polig": "https://informo.madrid.es/informo/tmadrid/INTENS_POLIG.kml",
    }
    kmls = {name: inspect_kml_raw(name, url) for name, url in kml_sources.items()}
    save_json(kmls, OUT / "13_208223_kml_raw.json")

    print("2/4 Re-parsing 203962 KMZ in detail...")
    kmz = parse_kmz_detailed(
        "https://datos.madrid.es/dataset/203962-0-trafico-imd/resource/"
        "203962-3-trafico-imd-kmz/download/203962-3-trafico-imd-kmz"
    )
    save_json(kmz, OUT / "13_203962_kmz.json")

    print("3/4 Opening 203962 XLSX...")
    xlsx = inspect_xlsx(
        "https://datos.madrid.es/dataset/203962-0-trafico-imd/resource/"
        "203962-0-trafico-imd-xlsx/download/203962-0-trafico-imd-xlsx.xlsx"
    )
    save_json(xlsx, OUT / "13_203962_xlsx.json")

    print("4/4 Additional CKAN search for road network sources...")
    queries = [
        "tramos via geometria",
        "segmentos viarios",
        "grafo via madrid",
        "cartografia viaria",
        "viales shp",
        "malla viaria",
        "intensidad trafico tramos",
    ]
    ckan_results = search_ckan(queries, rows=5)
    save_json(ckan_results, OUT / "13_ckan_search.json")

    # Markdown report
    sections = {
        "208223 KML raw diagnosis": dumps_jsonable(kmls, indent=2),
        "203962 KMZ structure": dumps_jsonable(
            {k: v for k, v in kmz.items() if k != "samples"}
            | {"samples": kmz.get("samples", [])[:3]}, indent=2),
        "203962 XLSX sheets and first rows": dumps_jsonable(xlsx, indent=2),
        "Additional CKAN search": dumps_jsonable(ckan_results, indent=2),
    }
    save_markdown_report(sections, Path("reports/13_phase4_diagnose.md"))
    print("Done. See reports/13_phase4_diagnose.md.")


if __name__ == "__main__":
    main()