"""
Phase 1 – Iteration 4: resolve remaining blocking questions before closing Phase 1.

Questions:
  Q1. What is the structure of 208627 (historical traffic, 162 resources)?
  Q2. Does 208223 (Mapa de tramas) contain segment geometry?
  Q3. Does 203962 (IMD por tramos) provide segment IDs / geometry?
  Q4. What does 202468 (307 resources) contain?
  Q5. What is the real unit of a permanent-count file (esp. the anomalous Dec-2024 file)?

Run:
    python scripts/03_resolve_phase1_gaps.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import save_json, save_markdown_report

TIMEOUT = 180
UA = "madrid-road-safety-audit/0.4"
CKAN = "https://datos.madrid.es/api/3/action"
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(url: str, nrows: int | None = None,
             usecols: list[str] | None = None) -> pd.DataFrame:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    last_exc = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(r.content), sep=";", encoding=enc,
                nrows=nrows, usecols=usecols, low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def ckan(dataset_id: str) -> dict[str, Any]:
    r = requests.get(f"{CKAN}/package_show?id={dataset_id}",
                     headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["result"]


def list_resources(dataset_id: str) -> list[dict[str, Any]]:
    return [
        {"name": r.get("name"), "url": r.get("url"),
         "format": r.get("format"), "size": r.get("size"),
         "last_modified": r.get("last_modified")}
        for r in ckan(dataset_id).get("resources", [])
    ]


# ---------------------------------------------------------------------------
# Q1: 208627 historical traffic
# ---------------------------------------------------------------------------
def audit_208627() -> dict[str, Any]:
    resources = list_resources("208627-0-transporte-ptomedida-historico")
    # Classify by name (year-month hints are usually in the resource title)
    sample_urls = []
    # pick a few across different naming patterns
    by_name = {r["name"]: r for r in resources if r["name"]}
    # deterministic sample: first, a middle one, last
    keys = sorted(by_name.keys())
    for idx in {0, len(keys) // 2, len(keys) - 1}:
        sample_urls.append(by_name[keys[idx]])

    samples = []
    for r in sample_urls:
        url = r["url"]
        entry = {"name": r["name"], "url": url}
        if not url or not url.lower().endswith(".csv"):
            entry["skip"] = "not a csv"
            samples.append(entry)
            continue
        try:
            head = read_csv(url, nrows=3)
            head.columns = [c.lstrip("\ufeff") for c in head.columns]
            entry["columns"] = list(head.columns)
            entry["dtypes"] = {c: str(d) for c, d in head.dtypes.items()}
            # if there's a date column, get its extent
            date_cols = [c for c in head.columns
                         if any(k in c.lower() for k in ("fecha", "fdia", "date"))]
            if date_cols:
                col = date_cols[0]
                full = read_csv(url, usecols=[col])
                s = full[col].astype(str).str.strip()
                s = s[s != ""]
                entry["date_extent"] = {
                    "n_rows": int(len(full)),
                    "min": str(s.min()),
                    "max": str(s.max()),
                }
        except Exception as exc:
            entry["error"] = str(exc)
        samples.append(entry)

    return {"n_resources": len(resources),
            "samples": samples,
            "all_names": [r["name"] for r in resources][:20]}


# ---------------------------------------------------------------------------
# Q2: 208223 Mapa de tramas
# ---------------------------------------------------------------------------
def audit_208223() -> dict[str, Any]:
    resources = list_resources("208223-0-trafico-intensidad-tramas")
    out = {"n_resources": len(resources), "resources": []}
    for r in resources:
        url = r["url"] or ""
        entry = {"name": r["name"], "url": url,
                 "format": r.get("format"), "size": r.get("size")}
        if url.lower().endswith(".csv"):
            try:
                head = read_csv(url, nrows=3)
                head.columns = [c.lstrip("\ufeff") for c in head.columns]
                entry["columns"] = list(head.columns)
                entry["has_wkt"] = any(
                    "wkt" in c.lower() or "geometry" in c.lower() or "geom" in c.lower()
                    for c in head.columns
                )
                entry["has_utm_point"] = any(
                    c.lower().startswith("utmx") or c.lower() == "coordenada_x_utm"
                    for c in head.columns
                )
            except Exception as exc:
                entry["error"] = str(exc)
        else:
            entry["note"] = "not CSV — may be a service or archive"
        out["resources"].append(entry)
    return out


# ---------------------------------------------------------------------------
# Q3: 203962 IMD por tramos
# ---------------------------------------------------------------------------
def audit_203962() -> dict[str, Any]:
    resources = list_resources("203962-0-trafico-imd")
    out = {"n_resources": len(resources), "resources": []}
    for r in resources[:5]:
        url = r["url"] or ""
        entry = {"name": r["name"], "url": url,
                 "format": r.get("format"), "size": r.get("size")}
        if url.lower().endswith(".csv"):
            try:
                head = read_csv(url, nrows=3)
                head.columns = [c.lstrip("\ufeff") for c in head.columns]
                entry["columns"] = list(head.columns)
                entry["has_wkt"] = any(
                    "wkt" in c.lower() or "geometry" in c.lower()
                    for c in head.columns
                )
                # detect a segment/trama identifier
                entry["id_candidates"] = [c for c in head.columns
                                          if any(k in c.lower() for k in
                                                 ("trama", "tramo", "codigo", "cod_via",
                                                  "segmento", "id"))]
            except Exception as exc:
                entry["error"] = str(exc)
        out["resources"].append(entry)
    return out


# ---------------------------------------------------------------------------
# Q4: 202468 Ubicación de puntos de medida
# ---------------------------------------------------------------------------
def audit_202468() -> dict[str, Any]:
    resources = list_resources("202468-0-intensidad-trafico")
    out = {"n_resources": len(resources),
           "names_sample": [r["name"] for r in resources[:30]]}
    # try to load the metadata (should be small, coordinates)
    candidates = [r for r in resources
                  if r["url"] and r["url"].lower().endswith(".csv")]
    loaded = []
    for r in candidates[:5]:
        try:
            head = read_csv(r["url"], nrows=5)
            head.columns = [c.lstrip("\ufeff") for c in head.columns]
            loaded.append({"name": r["name"], "columns": list(head.columns),
                           "dtypes": {c: str(d) for c, d in head.dtypes.items()}})
        except Exception as exc:
            loaded.append({"name": r["name"], "error": str(exc)})
    out["csv_loaded"] = loaded
    return out


# ---------------------------------------------------------------------------
# Q5: permanent count units
# ---------------------------------------------------------------------------
def audit_permanent_count_units() -> dict[str, Any]:
    # Inspect the anomalous Dec-2024 file and a normal one side by side
    normal = ("https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/"
              "resource/300233-56-aforo-trafico-permanentes-csv/download/"
              "300233-56-aforo-trafico-permanentes-csv.csv")
    anomalous = ("https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/"
                 "resource/300233-3-aforo-trafico-permanentes-csv/download/"
                 "300233-3-aforo-trafico-permanentes-csv.csv")

    out = {}

    # Normal file: how many rows, what's in FDIA, what's in FEST/FSEN
    try:
        df = read_csv(normal)
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        out["normal"] = {
            "n_rows": len(df),
            "FDIA_unique": int(df["FDIA"].nunique()) if "FDIA" in df else None,
            "FDIA_min": str(df["FDIA"].astype(str).min()),
            "FDIA_max": str(df["FDIA"].astype(str).max()),
            "FEST_unique": int(df["FEST"].nunique()) if "FEST" in df else None,
            "FSEN_unique": int(df["FSEN"].nunique()) if "FSEN" in df else None,
            "FSEN_values": sorted(map(str, df["FSEN"].dropna().unique()))[:5]
            if "FSEN" in df else None,
            "HOR_columns": [c for c in df.columns if c.startswith("HOR")],
            "row_per_date": (
                len(df) / df["FDIA"].nunique() if "FDIA" in df
                and df["FDIA"].nunique() > 0 else None
            ),
        }
    except Exception as exc:
        out["normal_error"] = str(exc)

    # Anomalous file: same inspection
    try:
        # First 2000 rows to understand pattern without loading 1M rows
        df = read_csv(anomalous, nrows=2000)
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        out["anomalous_head"] = {
            "columns": list(df.columns),
            "sample_rows": df.head(5).to_dict(orient="records"),
        }
        # Full FDIA distribution (single column)
        full = read_csv(anomalous, usecols=["FDIA"])
        out["anomalous_fdia"] = {
            "n_rows": len(full),
            "n_unique": int(full["FDIA"].nunique()),
            "min": str(full["FDIA"].astype(str).min()),
            "max": str(full["FDIA"].astype(str).max()),
            "row_per_date": len(full) / max(1, full["FDIA"].nunique()),
        }
    except Exception as exc:
        out["anomalous_error"] = str(exc)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Q1: historical traffic (208627)...")
    q1 = audit_208627()
    save_json(q1, OUT / "03_history_208627.json")

    print("Q2: tramas (208223)...")
    q2 = audit_208223()
    save_json(q2, OUT / "03_tramas_208223.json")

    print("Q3: IMD por tramos (203962)...")
    q3 = audit_203962()
    save_json(q3, OUT / "03_imd_203962.json")

    print("Q4: puntos de medida (202468)...")
    q4 = audit_202468()
    save_json(q4, OUT / "03_points_202468.json")

    print("Q5: permanent count units...")
    q5 = audit_permanent_count_units()
    save_json(q5, OUT / "03_count_units.json")

    sections = {
        "Q1 – Historical traffic (208627)": json.dumps(q1, indent=2, ensure_ascii=False),
        "Q2 – Tramas (208223)": json.dumps(q2, indent=2, ensure_ascii=False),
        "Q3 – IMD por tramos (203962)": json.dumps(q3, indent=2, ensure_ascii=False),
        "Q4 – Measurement points (202468)": json.dumps(q4, indent=2, ensure_ascii=False),
        "Q5 – Permanent count units": json.dumps(q5, indent=2, ensure_ascii=False),
    }
    save_markdown_report(sections, Path("reports/04_phase1_gaps.md"))
    print("\nDone. See reports/03_phase1_gaps.md and reports/tables/03_*.json")


if __name__ == "__main__":
    main()