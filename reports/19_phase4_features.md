# Source Audit Report

## Join summary

{
  "n_accidents_input": 137295,
  "n_accidents_matched_to_edge": 137295,
  "match_share": 1.0,
  "distance_to_edge_stats": {
    "median_m": 2.2,
    "p90_m": 6.5,
    "p99_m": 54.1,
    "max_m": 1747.8
  },
  "accidents_outside_madrid_utm": 10,
  "n_accidents_in_callejero_poly": 131090,
  "callejero_match_share": 0.9548,
  "tipo_via_distribution": {
    "Calle": 76099,
    "Avenida": 30105,
    "Paseo": 9184,
    "NaN": 6205,
    "Plaza": 4320,
    "Carretera": 3286,
    "Autovía": 3020,
    "Glorieta": 2688,
    "Camino": 741,
    "Ronda": 635,
    "Cuesta": 391,
    "Travesía": 164,
    "Bulevar": 129,
    "Puente": 111,
    "Carrera": 59,
    "Cañada": 53,
    "Costanilla": 29,
    "Pasaje": 27,
    "Paso Elevado": 14,
    "Callejón": 13,
    "Autopista": 7,
    "Pista": 7,
    "Acceso": 4,
    "Trasera": 3,
    "Jardín": 1
  },
  "edge_highway_distribution": {
    "residential": 53106,
    "primary": 23446,
    "tertiary": 21869,
    "secondary": 15991,
    "motorway": 8962,
    "motorway_link": 6662,
    "trunk": 3620,
    "living_street": 1324,
    "unclassified": 895,
    "primary_link": 474,
    "secondary_link": 357,
    "tertiary_link": 228,
    "busway": 187,
    "trunk_link": 174
  },
  "edge_maxspeed_imputed_share": 0.3656,
  "edge_lanes_null_share": 0.1736,
  "feature_null_shares": {
    "edge_highway": 0.0,
    "edge_maxspeed": 0.0,
    "edge_lanes": 0.1736,
    "tipo_via": 0.0452,
    "edge_name": 0.072
  }
}

## Notes

- Edges: OSM drive network from Fase 4 iter 6.
- Edges matched by nearest geometry; ties broken by longest edge.
- tipo_via comes from point-in-polygon of callejero rotulación (Calle, Avenida, Paseo...).
- maxspeed_kmh after fill covers ~100%. Raw coverage ~37%.
- maxspeed_is_imputed=True flags the 63% filled with fallback.
- edge_distance_m = distance from accident point to nearest edge.
