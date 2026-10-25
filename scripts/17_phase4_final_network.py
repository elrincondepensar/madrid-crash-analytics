"""
Phase 4 · Iteration 5 (v2) — Fix IMD (done) and build the definitive road
network from the Geoportal SHP (VIALES_SHP.zip).

Steps:
  1. Fix IMD: already done in v1. Re-verify only.
  2. Download VIALES_SHP.zip from geoportal.madrid.es.
  3. Load the SHP with geopandas. Inspect columns and geometry.
  4. Save network_callejero.parquet (WKT geometry + attributes).
  5. Report structure.

Run:
    python scripts/17_phase4_final_network.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

TIMEOUT = 300
UA = "madrid-road-safety/phase4/0.5"
OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

SHP_URL = ("https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/"
           "CALLEJERO/VIALES/VIALES_SHP.zip")
WFS_URL = ("https://sigma.madrid.es/hosted/services/CALLEJERO/"
           "CALLEJERO_VIALES/MapServer/WFSServer"
           "?request=GetCapabilities&service=WFS")


def download(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def load_shp_from_zip(content: bytes) -> gpd.GeoDataFrame:
    """Extract the first .shp inside a ZIP and load with geopandas."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        shp_names = [n for n in z.namelist() if n.lower().endswith(".shp")]
        if not shp_names:
            raise RuntimeError(
                f"No .shp inside archive. Contents: {z.namelist()[:10]}"
            )
        extract_dir = OUT_DATA / "_callejero_viales"
        extract_dir.mkdir(exist_ok=True)
        z.extractall(extract_dir)
        shp_path = extract_dir / shp_names[0]
        return gpd.read_file(shp_path)


def main() -> None:
    print("1/3 Verifying IMD fixed parquet...")
    imd_df = pd.read_parquet(OUT_DATA / "imd_points.parquet")
    if "TOTAL_fixed" not in imd_df.columns:
        raise RuntimeError("IMD parquet lacks TOTAL_fixed column.")
    imd_check = {
        "median": float(imd_df["TOTAL_fixed"].median()),
        "max": float(imd_df["TOTAL_fixed"].max()),
        "n_null": int(imd_df["TOTAL_fixed"].isna().sum()),
    }
    print(f"  IMD median={imd_check['median']:.0f}, "
          f"max={imd_check['max']:.0f}, nulls={imd_check['n_null']}")
    save_json(imd_check, OUT_TABLES / "17b_imd_check.json")

    print("\n2/3 Downloading VIALES_SHP.zip from Geoportal...")
    try:
        content = download(SHP_URL)
        print(f"  Downloaded {len(content):,} bytes.")
    except Exception as exc:
        print(f"  SHP download failed: {exc}")
        print(f"  WFS fallback URL: {WFS_URL}")
        save_json({"error": str(exc), "shp_url": SHP_URL, "wfs_url": WFS_URL},
                   OUT_TABLES / "17b_shp_error.json")
        return

    print("\n3/3 Loading SHP with geopandas...")
    try:
        gdf = load_shp_from_zip(content)
    except Exception as exc:
        print(f"  Load error: {exc}")
        save_json({"error": str(exc)}, OUT_TABLES / "17b_shp_load_error.json")
        return

    print(f"  Loaded {len(gdf):,} features.")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")
    print(f"  Geometry types: {gdf.geometry.type.value_counts().to_dict()}")

    # Persist as WKT
    gdf_out = gdf.copy()
    gdf_out["geometry_wkt"] = gdf_out["geometry"].apply(
        lambda g: g.wkt if g is not None else None
    )
    gdf_out = gdf_out.drop(columns=["geometry"])
    gdf_out.to_parquet(OUT_DATA / "network_callejero.parquet", index=False)
    print(f"  Saved {OUT_DATA / 'network_callejero.parquet'}")

    structure = {
        "n_features": int(len(gdf)),
        "crs": str(gdf.crs),
        "columns": list(gdf.columns),
        "geometry_types": gdf.geometry.type.value_counts().to_dict(),
        "sample_rows": gdf.drop(columns=["geometry"]).head(3).to_dict(
            orient="records"),
    }
    save_json(structure, OUT_TABLES / "17b_callejero_structure.json")

    sections = {
        "IMD check (already fixed)": dumps_jsonable(imd_check, indent=2),
        "VIALES_SHP structure": dumps_jsonable(structure, indent=2),
        "Notes": (
            "Source: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/"
            "CALLEJERO/VIALES/VIALES_SHP.zip\n"
            "CRS is expected to be EPSG:25830 (UTM 30N).\n"
            "If geometry is Polygon, viales are recintos (blocks); "
            "if LineString, they are axes."
        ),
    }
    save_markdown_report(sections, Path("reports/17b_phase4_network_v2.md"))
    print("\nDone. See reports/17b_phase4_network_v2.md.")


if __name__ == "__main__":
    main()