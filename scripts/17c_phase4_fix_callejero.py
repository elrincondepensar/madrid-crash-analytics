"""
Phase 4 · Fix (v3) — Repair network_callejero.parquet.

Background
----------
The original callejero parquet (script 17) stored geometry as raw WKT
strings with no CRS metadata. Of the 9,408 polygons, ~98% have
self-intersections or other topological invalidities inherited from
the Viales SHP of the Geoportal. Phase 8 crashed trying to read it
because:
  1. _reconstruct_geom hardcoded crs=EPSG:4326 (bug now fixed in io.py),
  2. sjoin_nearest on invalid geometries produced Infinity coords after
     the incorrect reprojection.

What this script does
---------------------
  1. Load network_callejero.parquet via io.load_geometry_parquet
     (CRS auto-detected as EPSG:25830).
  2. Repair every invalid geometry via shapely.make_valid (with buffer(0)
     fallback). Keep only the polygonal component of repaired
     GeometryCollections.
  3. Drop unrecoverable rows.
  4. Save as GeoParquet (both `geometry` and `geometry_wkt` preserved).
  5. Report counts and sample repairs.

Run:
    python scripts/17c_phase4_fix_callejero.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import (
    dumps_jsonable, save_json, save_markdown_report,
)

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

IN_PATH = OUT_DATA / "network_callejero.parquet"
OUT_PATH = OUT_DATA / "network_callejero.parquet"  # in-place


def main() -> None:
    print(f"1/4 Loading {IN_PATH} ...")
    gdf = ms_io.load_geometry_parquet(IN_PATH)
    print(f"  rows={len(gdf):,}  CRS={gdf.crs.to_epsg()}  "
          f"geom_types={gdf.geometry.geom_type.value_counts().to_dict()}")

    n_in = len(gdf)

    print("\n2/4 Repairing geometries ...")
    gdf = ms_io.repair_geometries(gdf, label="callejero")
    n_out = len(gdf)
    print(f"  rows={n_out:,}  (dropped {n_in - n_out})")

    # Detect any remaining corrupt rows (infinite coords)
    n_inf = 0
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        b = g.bounds
        if not all(abs(v) < 1e9 for v in b):
            n_inf += 1
    print(f"  geometries with non-finite bounds: {n_inf}")

    print("\n3/4 Saving GeoParquet ...")
    ms_io.save_geometry_parquet(gdf, OUT_PATH)
    print(f"  saved {OUT_PATH}")

    print("\n4/4 Report ...")
    report = {
        "n_input": n_in,
        "n_output": n_out,
        "n_dropped": n_in - n_out,
        "crs": gdf.crs.to_epsg(),
        "n_non_finite_bounds": n_inf,
        "geom_type_distribution": gdf.geometry.geom_type.value_counts().to_dict(),
        "columns": list(gdf.columns),
    }
    save_json(report, OUT_TABLES / "17c_callejero_fix.json")

    sections = {
        "Callejero repair": dumps_jsonable(report, indent=2),
        "Notes": (
            f"- Input: {n_in:,} polygons from Viales SHP (Geoportal).\n"
            "- Repaired via shapely.make_valid + buffer(0) fallback.\n"
            "- Saved as GeoParquet with CRS metadata AND legacy geometry_wkt.\n"
            "- Downstream point-in-polygon joins can now use "
            "`gpd.read_parquet()` directly without reprojection ambiguity.\n"
        ),
    }
    save_markdown_report(sections, Path("reports/17c_phase4_fix_callejero.md"))
    print("\nDone. See reports/17c_phase4_fix_callejero.md.")


if __name__ == "__main__":
    main()