# Source Audit Report

## OSM bbox

{
  "bbox_wgs84": [
    -3.8427317306778255,
    40.31231317355152,
    -3.5186484137986533,
    40.63725158288926
  ]
}

## OSM network summary

{
  "n_edges": 95437,
  "crs_original": "EPSG:4326",
  "highway_counts": {
    "residential": 64763,
    "tertiary": 11493,
    "secondary": 5357,
    "primary": 4279,
    "living_street": 3283,
    "motorway_link": 1861,
    "unclassified": 1714,
    "motorway": 991,
    "trunk": 820,
    "trunk_link": 276,
    "tertiary_link": 184,
    "primary_link": 180,
    "secondary_link": 158,
    "busway": 77,
    "road": 1
  },
  "name_null_share": 0.1441,
  "maxspeed_kmh_stats": {
    "n_valid": 35263,
    "n_null": 60174,
    "min": 3.0,
    "median": 40.0,
    "p90": 50.0,
    "max": 120.0
  },
  "maxspeed_kmh_counts": {
    "50.0": 13841,
    "30.0": 12790,
    "40.0": 4673,
    "20.0": 2081,
    "90.0": 378,
    "70.0": 373,
    "60.0": 350,
    "100.0": 322,
    "80.0": 290,
    "120.0": 148
  },
  "lanes_n_stats": {
    "n_valid": 49792,
    "n_null": 45645,
    "min": 1.0,
    "median": 2.0,
    "p90": 4.0,
    "max": 12.0
  },
  "length_m_stats": {
    "n_valid": 95437,
    "n_null": 0,
    "min": 0.5580683796558925,
    "median": 63.75661717214911,
    "p90": 207.31429237994143,
    "max": 6866.439024081492
  },
  "total_network_length_km": 9809.3
}

## Notes

- network_type='drive' (motorways, primary, secondary, residential, service principal).
- Edges are simplified (osmnx simplify=True).
- maxspeed_kmh and lanes_n have nulls where OSM lacks data; Fase 5 imputes or uses missingness indicators.
- length_m is the edge length in metres.
