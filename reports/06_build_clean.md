# Source Audit Report

## Validation

{
  "n_accidents": 137328,
  "n_persons": 322317,
  "n_duplicate_expedientes": 0,
  "n_rows_year_outside_range": 0,
  "n_missing_coords": 33,
  "severity_counts": {
    "no_injury": 74855,
    "minor": 48106,
    "serious": 14165,
    "fatal": 202
  },
  "severity_share": {
    "no_injury": 0.5451,
    "minor": 0.3503,
    "serious": 0.1031,
    "fatal": 0.0015
  },
  "severity_counts_by_year": {
    "2019": {
      "fatal": 33,
      "minor": 8329,
      "no_injury": 11241,
      "serious": 2331
    },
    "2020": {
      "fatal": 34,
      "minor": 5251,
      "no_injury": 7357,
      "serious": 1530
    },
    "2021": {
      "fatal": 28,
      "minor": 6399,
      "no_injury": 9642,
      "serious": 1883
    },
    "2022": {
      "fatal": 21,
      "minor": 7230,
      "no_injury": 10917,
      "serious": 2044
    },
    "2023": {
      "fatal": 24,
      "minor": 6951,
      "no_injury": 11532,
      "serious": 2165
    },
    "2024": {
      "fatal": 28,
      "minor": 6893,
      "no_injury": 11644,
      "serious": 2133
    },
    "2025": {
      "fatal": 34,
      "minor": 7053,
      "no_injury": 12522,
      "serious": 2079
    }
  },
  "total_fatalities": 207,
  "sum_n_people_equals_n_persons": true,
  "severity_fatal_iff_n_fatalities_ge_1": true,
  "severity_fatal_violations": 0,
  "severity_no_injury_matches_worst_code": true,
  "severity_no_injury_violations": 0,
  "worst_lesividad_code_counts": {
    "-1": 57489,
    "7": 29971,
    "14": 17366,
    "2": 11058,
    "1": 8089,
    "6": 6824,
    "5": 3222,
    "3": 3107,
    "4": 202
  }
}

## Accident columns and dtypes

{
  "num_expediente": "str",
  "year": "Int64",
  "fecha": "datetime64[us]",
  "hora": "str",
  "hour": "Int8",
  "localizacion": "str",
  "numero": "float64",
  "cod_distrito": "float64",
  "distrito": "str",
  "coordenada_x_utm": "float64",
  "coordenada_y_utm": "float64",
  "tipo_accidente": "str",
  "estado_meteorologico": "str",
  "n_people": "int64",
  "n_vehicles": "int64",
  "n_conductores": "int64",
  "n_pasajeros": "int64",
  "n_peatones": "int64",
  "has_bicycle": "bool",
  "has_motorcycle": "bool",
  "has_heavy_vehicle": "bool",
  "has_pedestrian": "bool",
  "any_alcohol_positive": "bool",
  "n_alcohol_positive": "int64",
  "any_drug_positive": "bool",
  "n_fatalities": "int64",
  "n_serious": "int64",
  "n_minor": "int64",
  "worst_lesividad_code": "int64",
  "severity": "str",
  "severity_code": "int8"
}

## Person columns and dtypes

{
  "num_expediente": "str",
  "fecha": "datetime64[us]",
  "hora": "str",
  "localizacion": "str",
  "numero": "float64",
  "cod_distrito": "float64",
  "distrito": "str",
  "tipo_accidente": "str",
  "estado_meteorologico": "str",
  "tipo_vehiculo": "str",
  "tipo_persona": "str",
  "rango_edad": "str",
  "sexo": "str",
  "cod_lesividad": "float64",
  "lesividad": "str",
  "coordenada_x_utm": "float64",
  "coordenada_y_utm": "float64",
  "positiva_alcohol": "str",
  "positiva_droga": "float64",
  "file_year": "int64",
  "year": "Int64",
  "hour": "Int8",
  "is_driver": "bool",
  "is_passenger": "bool",
  "is_pedestrian": "bool",
  "is_bicycle": "bool",
  "is_motorcycle": "bool",
  "is_heavy_vehicle": "bool",
  "is_fatal": "bool",
  "is_serious": "bool",
  "is_minor": "bool",
  "has_known_lesividad": "bool",
  "alcohol_positive": "bool",
  "drug_positive": "bool"
}
