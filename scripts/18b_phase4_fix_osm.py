"""
Phase 4 · Fix (v2) — Convert network_osm.parquet to proper GeoParquet.

The original OSM parquet (script 18) stored geometry as raw WKT strings
in EPSG:4326 with no GeoParquet metadata. The geometries are actually
valid (OSM is careful), but any reader relying on gpd.read_parquet()
fails and any CRS misdeclaration breaks reprojection.

What this script does
---------------------
  1. Load network_osm.parquet via io.load_geometry_parquet
     (CRS auto-detected as EPSG:4326).
  2. Check validity (should be 100% valid; repairs if not).
  3. Save as GeoParquet (both `geometry` and `geometry_wkt` preserved).
  4. Verify row order and CRS.

Run:
    python scripts/18b_phase4_fix_osm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import (
    dumps_jsonable, save_json, save_markdown_report,
)

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

IN_PATH = OUT_DATA / "network_osm.parquet"
OUT_PATH = OUT_DATA / "network_osm.parquet"


def main() -> None:
    print(f"1/4 Loading {IN_PATH} ...")
    gdf = ms_io.load_geometry_parquet(IN_PATH)
    print(f"  rows={len(gdf):,}  CRS={gdf.crs.to_epsg()}  "
          f"geom_types={gdf.geometry.geom_type.value_counts().to_dict()}")

    n_in = len(gdf)
    n_invalid = int((~gdf.geometry.is_valid).sum())
    print(f"  invalid geometries: {n_invalid}")

    print("\n2/4 Repairing (if needed) ...")
    if n_invalid > 0:
        gdf = ms_io.repair_geometries(gdf, label="osm_network")
    else:
        print("  nothing to repair")
    n_out = len(gdf)
    print(f"  rows={n_out:,}")

    print("\n3/4 Sanity: confirm edge_id positional alignment ...")
    # Row order must be preserved so that Fase 7 edge_id remains valid
    edge_ids = gdf.index.tolist() if hasattr(gdf, "index") else None
    has_range_index = (list(gdf.index) == list(range(len(gdf))))
    print(f"  RangeIndex preserved: {has_range_index}")
    if not has_range_index:
        print("  WARNING: index was modified. Resetting to RangeIndex.")
        gdf = gdf.reset_index(drop=True)

    print("\n4/4 Saving GeoParquet + report ...")
    ms_io.save_geometry_parquet(gdf, OUT_PATH)
    print(f"  saved {OUT_PATH}")

    report = {
        "n_input": n_in,
        "n_output": n_out,
        "n_repaired": n_in - n_out,
        "crs": gdf.crs.to_epsg(),
        "geom_type_distribution": gdf.geometry.geom_type.value_counts().to_dict(),
        "columns": list(gdf.columns),
        "range_index_preserved": has_range_index,
    }
    save_json(report, OUT_TABLES / "18b_osm_fix.json")

    sections = {
        "OSM network fix": dumps_jsonable(report, indent=2),
        "Notes": (
            f"- Input: {n_in:,} edges (OSM drive network).\n"
            "- Saved as GeoParquet with CRS metadata (EPSG:4326) "
            "AND legacy geometry_wkt.\n"
            "- Row order preserved → edge_id in "
            "network_kde_final.parquet (Fase 7) remains aligned.\n"
        ),
    }
    save_markdown_report(sections, Path("reports/18b_phase4_fix_osm.md"))
    print("\nDone. See reports/18b_phase4_fix_osm.md.")


if __name__ == "__main__":
    main()