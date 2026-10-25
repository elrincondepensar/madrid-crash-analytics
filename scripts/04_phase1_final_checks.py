"""
Phase 1 – Iteration 5: final blocking checks before closing Phase 1.

Q1. What's inside a 208627 ZIP?
Q2. Does 208223 KML have segments with ID + geometry?
Q3. Does 203962 KMZ/XLS have segment ID + geometry?
Q4. What distinct tipo_elem values exist in 202468?
Q5. What's actually in the anomalous 300233-3 (Dec-2024) file?

Run from the repository root:
    python scripts/04_phase1_final_checks.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import save_json, save_markdown_report

TIMEOUT = 180
UA = "madrid-road-safety-audit/0.5"
OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)


def download(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def read_csv_from_bytes(content: bytes, nrows: int | None = None,
                        usecols: list[str] | None = None,
                        sep: str = ";") -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc,
                               nrows=nrows, usecols=usecols, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode CSV")


# ---------------------------------------------------------------------------
# Q1 – 208627 ZIP
# ---------------------------------------------------------------------------
def q1_208627_zip() -> dict[str, Any]:
    url = ("https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/"
           "resource/208627-0-transporte-ptomedida-historico-zip/download/"
           "208627-0-transporte-ptomedida-historico-zip.zip")
    result = {"url": url}
    try:
        content = download(url)
        result["size_bytes"] = len(content)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = z.namelist()
            result["n_files"] = len(names)
            result["sample_names"] = names[:5]
            csv_names = [n for n in names if n.lower().endswith(".csv")]
            if csv_names:
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep=";", nrows=10, encoding="utf-8-sig",
                                     low_memory=False)
                df.columns = [c.lstrip("\ufeff") for c in df.columns]
                result["schema"] = {
                    "columns": list(df.columns),
                    "dtypes": {c: str(d) for c, d in df.dtypes.items()},
                    "sample_rows": df.head(3).to_dict(orient="records"),
                }
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Q2 – 208223 KML
# ---------------------------------------------------------------------------
def parse_kml(content: bytes, max_placemarks: int = 2) -> dict[str, Any]:
    root = ET.fromstring(content)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    # Some KML files use no namespace; handle both
    if not root.findall(".//kml:Placemark", ns):
        ns = {"kml": ""}  # type: ignore[assignment]
        pms = root.findall(".//Placemark")
    else:
        pms = root.findall(".//kml:Placemark", ns)
    result = {"n_placemarks": len(pms), "samples": []}
    for pm in pms[:max_placemarks]:
        sample = {}
        for tag in ("name", "description"):
            el = pm.find(f"kml:{tag}", ns)
            if el is None:
                el = pm.find(tag)
            sample[tag] = (el.text if el is not None and el.text else "")[:200]
        geom_types = []
        for gt in ("Point", "LineString", "Polygon", "MultiGeometry"):
            if pm.find(f".//kml:{gt}", ns) is not None or \
               pm.find(f".//{gt}") is not None:
                geom_types.append(gt)
        coords_el = pm.find(".//kml:coordinates", ns) or pm.find(".//coordinates")
        n_coord_tuples = 0
        if coords_el is not None and coords_el.text:
            n_coord_tuples = len(coords_el.text.strip().split())
        sample["geometry_types"] = geom_types
        sample["n_coordinate_tuples"] = n_coord_tuples
        result["samples"].append(sample)
    return result


def q2_208223_kml() -> dict[str, Any]:
    urls = {
        "intensidades": "https://informo.madrid.es/informo/tmadrid/INTENSIDADES.kml",
        "intens_polig": "https://informo.madrid.es/informo/tmadrid/INTENS_POLIG.kml",
    }
    out = {}
    for name, url in urls.items():
        try:
            content = download(url)
            out[name] = {"size_bytes": len(content), **parse_kml(content)}
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Q3 – 203962 KMZ + XLSX
# ---------------------------------------------------------------------------
def q3_203962() -> dict[str, Any]:
    out = {}
    # KMZ
    kmz_url = ("https://datos.madrid.es/dataset/203962-0-trafico-imd/resource/"
               "203962-3-trafico-imd-kmz/download/203962-3-trafico-imd-kmz")
    try:
        content = download(kmz_url)
        out["kmz_size_bytes"] = len(content)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            out["kmz_files"] = kml_names
            if kml_names:
                kml = z.read(kml_names[0])
                out["kmz_kml"] = parse_kml(kml, max_placemarks=2)
    except Exception as exc:
        out["kmz_error"] = str(exc)
    # XLSX
    xlsx_url = ("https://datos.madrid.es/dataset/203962-0-trafico-imd/resource/"
                "203962-0-trafico-imd-xlsx/download/203962-0-trafico-imd-xlsx.xlsx")
    try:
        content = download(xlsx_url)
        df = pd.read_excel(io.BytesIO(content), nrows=5)
        df.columns = [str(c).strip() for c in df.columns]
        out["xlsx"] = {
            "columns": list(df.columns),
            "dtypes": {c: str(d) for c, d in df.dtypes.items()},
            "sample_rows": df.head(3).to_dict(orient="records"),
            "id_candidates": [c for c in df.columns
                              if any(k in c.lower() for k in
                                     ("trama", "tramo", "codigo", "id", "segmento"))],
            "geom_candidates": [c for c in df.columns
                                if any(k in c.lower() for k in
                                       ("wkt", "geom", "lat", "lon", "utmx", "utmy"))],
        }
    except Exception as exc:
        out["xlsx_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Q4 – 202468 tipo_elem
# ---------------------------------------------------------------------------
def q4_202468_tipos() -> dict[str, Any]:
    import requests as _r
    r = _r.get("https://datos.madrid.es/api/3/action/package_show?id=202468-0-intensidad-trafico",
               headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    resources = r.json()["result"]["resources"]
    csv_resources = [res for res in resources
                     if (res.get("format", "") or "").upper() == "CSV"]
    out = {"n_csv_resources": len(csv_resources), "samples": []}
    for res in csv_resources[:3]:
        try:
            df = pd.read_csv(res["url"], nrows=10000, low_memory=False)
            df.columns = [c.lstrip("\ufeff") for c in df.columns]
            entry = {"name": res.get("name"),
                     "n_rows_sampled": len(df)}
            if "tipo_elem" in df.columns:
                entry["tipo_elem_counts"] = df["tipo_elem"].value_counts().to_dict()
            if "id" in df.columns:
                entry["n_unique_id_in_sample"] = int(df["id"].nunique())
            out["samples"].append(entry)
        except Exception as exc:
            out["samples"].append({"name": res.get("name"), "error": str(exc)})
    return out


# ---------------------------------------------------------------------------
# Q5 – Dec 2024 anomaly diagnosis
# ---------------------------------------------------------------------------
def q5_dec2024() -> dict[str, Any]:
    url = ("https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/"
           "resource/300233-3-aforo-trafico-permanentes-csv/download/"
           "300233-3-aforo-trafico-permanentes-csv.csv")
    try:
        df = pd.read_csv(url, sep=";", encoding="utf-8-sig",
                         usecols=["FDIA", "FEST", "FSEN"], low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(url, sep=";", encoding="latin-1",
                         usecols=["FDIA", "FEST", "FSEN"], low_memory=False)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    n_rows = len(df)
    n_days = df["FDIA"].nunique()
    n_stations = df["FEST"].nunique()
    n_fsen = df["FSEN"].nunique()
    expected_per_day = n_stations * n_fsen
    actual_per_day = n_rows / max(1, n_days)
    n_dups = int(df.duplicated(subset=["FDIA", "FEST", "FSEN"]).sum())
    # How many rows per (station, fsen) pair per day?
    counts = df.groupby(["FDIA", "FEST", "FSEN"]).size()
    result = {
        "n_rows": n_rows,
        "n_days": n_days,
        "n_stations": n_stations,
        "n_fsen": n_fsen,
        "n_station_fsen_pairs": n_stations * n_fsen,
        "expected_rows_per_day": expected_per_day,
        "actual_rows_per_day": actual_per_day,
        "over_factor": actual_per_day / max(1, expected_per_day),
        "n_duplicate_fdia_fest_fsen": n_dups,
        "row_count_distribution_per_key": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "max": int(counts.max()),
            "n_keys": len(counts),
        },
        "sample_station_names": df["FEST"].drop_duplicates().head(10).tolist(),
        "sample_fsen_values": df["FSEN"].drop_duplicates().tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Q1 – 208627 ZIP contents...")
    q1 = q1_208627_zip()
    save_json(q1, OUT / "04_history_208627_zip.json")

    print("Q2 – 208223 KML geometry...")
    q2 = q2_208223_kml()
    save_json(q2, OUT / "04_tramas_kml.json")

    print("Q3 – 203962 KMZ / XLSX...")
    q3 = q3_203962()
    save_json(q3, OUT / "04_imd_203962.json")

    print("Q4 – 202468 tipo_elem...")
    q4 = q4_202468_tipos()
    save_json(q4, OUT / "04_points_202468.json")

    print("Q5 – Dec-2024 anomaly...")
    q5 = q5_dec2024()
    save_json(q5, OUT / "04_dec2024_anomaly.json")

    sections = {
        "Q1 – 208627 ZIP": json.dumps(q1, indent=2, ensure_ascii=False),
        "Q2 – 208223 KML": json.dumps(q2, indent=2, ensure_ascii=False),
        "Q3 – 203962 KMZ/XLSX": json.dumps(q3, indent=2, ensure_ascii=False),
        "Q4 – 202468 tipo_elem": json.dumps(q4, indent=2, ensure_ascii=False),
        "Q5 – Dec-2024 anomaly": json.dumps(q5, indent=2, ensure_ascii=False),
    }
    save_markdown_report(sections, Path("reports/05_phase1_final.md"))
    print("\nDone. See reports/04_phase1_final.md and reports/tables/04_*.json")


if __name__ == "__main__":
    main()