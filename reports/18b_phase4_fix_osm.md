# Source Audit Report

## OSM network fix

{
  "n_input": 95437,
  "n_output": 95437,
  "n_repaired": 0,
  "crs": 4326,
  "geom_type_distribution": {
    "LineString": 95437
  },
  "columns": [
    "osm_id",
    "u",
    "v",
    "key",
    "highway",
    "name",
    "maxspeed_kmh",
    "lanes_n",
    "oneway",
    "length_m",
    "geometry"
  ],
  "range_index_preserved": true
}

## Notes

- Input: 95,437 edges (OSM drive network).
- Saved as GeoParquet with CRS metadata (EPSG:4326) AND legacy geometry_wkt.
- Row order preserved → edge_id in network_kde_final.parquet (Fase 7) remains aligned.
