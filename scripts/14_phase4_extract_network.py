"""
Phase 4 · Iteration 2 — Extract 208223 (road segments) and 203962 (IMD points)
to canonical parquet.

Outputs:
    data/processed/network_segments.parquet
        One row per 208223 Placemark (7,856 expected). Columns:
        segment_id, name, description, geometry_wkt (LineString), n_points,
        bbox_minx, bbox_miny, bbox_maxx, bbox_maxy.

    data/processed/imd_points.parquet
        One row per 203962 KMZ Placemark (1,961 expected). Columns:
        point_id, name, lon, lat, and each SimpleField as a column.

    reports/14_phase4_extract.md

Run:
    pip install lxml shapely
    python scripts/14_phase4_extract_network.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

TIMEOUT = 180
UA = "madrid-road-safety/phase4/0.2"
OUT_DATA = Path("data/processed")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)


def download(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# 208223 — road segments
# ---------------------------------------------------------------------------
def parse_kml22_segments(content: bytes) -> list[dict[str, Any]]:
    """
    Parse a KML 2.2 file (namespace earth.google.com/kml/2.2) with LineString
    placemarks. Returns a list of dicts with name, description and WKT geometry.
    """
    root = etree.fromstring(content)
    rows = []
    for i, pm in enumerate(root.iter("{*}Placemark")):
        name_el = pm.find("{*}name")
        desc_el = pm.find("{*}description")
        coords_el = pm.find(".//{*}LineString/{*}coordinates")
        if coords_el is None or not coords_el.text:
            continue
        # KML coordinates: "lon,lat,alt lon,lat,alt ..."
        raw = coords_el.text.strip()
        pairs = []
        for tok in raw.split():
            parts = tok.split(",")
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            pairs.append((lon, lat))
        if len(pairs) < 2:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        wkt = "LINESTRING (" + ", ".join(f"{x} {y}" for x, y in pairs) + ")"
        rows.append({
            "segment_id": i,
            "name": name_el.text if name_el is not None else None,
            "description": (desc_el.text[:300]
                            if desc_el is not None and desc_el.text else None),
            "geometry_wkt": wkt,
            "n_points": len(pairs),
            "bbox_minx": min(xs),
            "bbox_miny": min(ys),
            "bbox_maxx": max(xs),
            "bbox_maxy": max(ys),
        })
    return rows


def extract_208223() -> dict[str, Any]:
    url = "https://informo.madrid.es/informo/tmadrid/INTENSIDADES.kml"
    print(f"  downloading {url}...")
    content = download(url)
    print(f"  parsing {len(content):,} bytes...")
    rows = parse_kml22_segments(content)
    df = pd.DataFrame(rows)
    print(f"  parsed {len(df):,} LineString segments.")
    df.to_parquet(OUT_DATA / "network_segments.parquet", index=False)
    return {
        "n_segments": len(df),
        "n_missing_name": int(df["name"].isna().sum()),
        "n_missing_description": int(df["description"].isna().sum()),
        "n_points_stats": {
            "min": int(df["n_points"].min()),
            "median": float(df["n_points"].median()),
            "max": int(df["n_points"].max()),
            "mean": round(float(df["n_points"].mean()), 1),
        },
        "bbox": {
            "min_lon": round(df["bbox_minx"].min(), 5),
            "max_lon": round(df["bbox_maxx"].max(), 5),
            "min_lat": round(df["bbox_miny"].min(), 5),
            "max_lat": round(df["bbox_maxy"].max(), 5),
        },
        "names_sample": df["name"].dropna().head(10).tolist(),
        "descriptions_sample": df["description"].dropna().head(5).tolist(),
        "n_unique_names": int(df["name"].nunique()),
        "duplicated_names": int(df["name"].duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# 203962 — IMD points
# ---------------------------------------------------------------------------
def parse_kmz_points(content: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse a KMZ (zip with doc.kml inside). Returns (rows, schema_fields).
    Each row is one Placemark: name, lon, lat, and one column per SimpleData field.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise RuntimeError("No KML inside KMZ")
        kml_bytes = z.read(kml_names[0])

    root = etree.fromstring(kml_bytes)

    # Extract Schema field definitions
    schema_fields = []
    for sf in root.iter("{*}SimpleField"):
        schema_fields.append({
            "name": sf.get("name", ""),
            "type": sf.get("type", ""),
        })

    rows = []
    for i, pm in enumerate(root.iter("{*}Placemark")):
        name_el = pm.find("{*}name")
        coords_el = pm.find(".//{*}Point/{*}coordinates")
        if coords_el is None or not coords_el.text:
            continue
        tok = coords_el.text.strip().split()[0]
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        row: dict[str, Any] = {
            "point_id": i,
            "name": name_el.text if name_el is not None else None,
            "lon": lon,
            "lat": lat,
        }
        # SchemaData / SimpleData
        for sd in pm.findall(".//{*}SimpleData"):
            key = sd.get("name", "")
            val = sd.text
            row[key] = val
        rows.append(row)
    return rows, schema_fields


def extract_203962() -> dict[str, Any]:
    url = ("https://datos.madrid.es/dataset/203962-0-trafico-imd/resource/"
           "203962-3-trafico-imd-kmz/download/203962-3-trafico-imd-kmz")
    print(f"  downloading {url}...")
    content = download(url)
    rows, schema_fields = parse_kmz_points(content)
    df = pd.DataFrame(rows)
    print(f"  parsed {len(df):,} points, {df.shape[1]} columns.")
    df.to_parquet(OUT_DATA / "imd_points.parquet", index=False)
    return {
        "n_points": len(df),
        "n_columns": df.shape[1],
        "schema_fields": schema_fields,
        "columns": list(df.columns),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "bbox": {
            "min_lon": round(float(df["lon"].min()), 5),
            "max_lon": round(float(df["lon"].max()), 5),
            "min_lat": round(float(df["lat"].min()), 5),
            "max_lat": round(float(df["lat"].max()), 5),
        },
        "name_nulls": int(df["name"].isna().sum()),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("1/2 Extracting 208223 → network_segments.parquet")
    seg_summary = extract_208223()
    save_json(seg_summary, OUT_TABLES / "14_network_segments.json")

    print("\n2/2 Extracting 203962 KMZ → imd_points.parquet")
    imd_summary = extract_203962()
    save_json(imd_summary, OUT_TABLES / "14_imd_points.json")

    sections = {
        "208223 — network_segments.parquet": dumps_jsonable(seg_summary, indent=2),
        "203962 — imd_points.parquet": dumps_jsonable(imd_summary, indent=2),
    }
    save_markdown_report(sections, Path("reports/14_phase4_extract.md"))
    print("\nDone. See reports/14_phase4_extract.md")


if __name__ == "__main__":
    main()