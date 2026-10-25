# Source Audit Report

## Q1 – 208627 ZIP

{
  "url": "https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource/208627-0-transporte-ptomedida-historico-zip/download/208627-0-transporte-ptomedida-historico-zip.zip",
  "size_bytes": 79348668,
  "n_files": 1,
  "sample_names": [
    "12-2022.csv"
  ],
  "schema": {
    "columns": [
      "id",
      "fecha",
      "tipo_elem",
      "intensidad",
      "ocupacion",
      "carga",
      "vmed",
      "error",
      "periodo_integracion"
    ],
    "dtypes": {
      "id": "int64",
      "fecha": "str",
      "tipo_elem": "str",
      "intensidad": "int64",
      "ocupacion": "int64",
      "carga": "int64",
      "vmed": "int64",
      "error": "str",
      "periodo_integracion": "int64"
    },
    "sample_rows": [
      {
        "id": 1001,
        "fecha": "2022-12-01 00:00:00",
        "tipo_elem": "M30",
        "intensidad": 3240,
        "ocupacion": 9,
        "carga": 0,
        "vmed": 59,
        "error": "N",
        "periodo_integracion": 5
      },
      {
        "id": 1001,
        "fecha": "2022-12-01 00:15:00",
        "tipo_elem": "M30",
        "intensidad": 3240,
        "ocupacion": 9,
        "carga": 0,
        "vmed": 59,
        "error": "N",
        "periodo_integracion": 5
      },
      {
        "id": 1001,
        "fecha": "2022-12-01 00:30:00",
        "tipo_elem": "M30",
        "intensidad": 300,
        "ocupacion": 1,
        "carga": 0,
        "vmed": 66,
        "error": "N",
        "periodo_integracion": 5
      }
    ]
  }
}

## Q2 – 208223 KML

{
  "intensidades": {
    "size_bytes": 1765646,
    "n_placemarks": 0,
    "samples": []
  },
  "intens_polig": {
    "size_bytes": 2872438,
    "n_placemarks": 0,
    "samples": []
  }
}

## Q3 – 203962 KMZ/XLSX

{
  "kmz_size_bytes": 293084,
  "kmz_files": [
    "doc.kml"
  ],
  "kmz_kml": {
    "n_placemarks": 1961,
    "samples": [
      {
        "name": "Pelícano - Argüeso",
        "description": "",
        "geometry_types": [
          "Point"
        ],
        "n_coordinate_tuples": 0
      },
      {
        "name": "Viejo de Leganés (Camino)- Besolla",
        "description": "",
        "geometry_types": [
          "Point"
        ],
        "n_coordinate_tuples": 0
      }
    ]
  },
  "xlsx_error": "`Import openpyxl` failed.  Use pip or conda to install the openpyxl package."
}

## Q4 – 202468 tipo_elem

{
  "n_csv_resources": 96,
  "samples": [
    {
      "name": "202468-0-intensidad-trafico-csv",
      "error": "Error tokenizing data. C error: Expected 1 fields in line 19, saw 2\n"
    },
    {
      "name": "202468-1-intensidad-trafico-csv",
      "error": "'utf-8' codec can't decode byte 0xba in position 326: invalid start byte"
    },
    {
      "name": "202468-2-intensidad-trafico-csv",
      "error": "'utf-8' codec can't decode byte 0xba in position 1597: invalid start byte"
    }
  ]
}

## Q5 – Dec-2024 anomaly

{
  "n_rows": 1044731,
  "n_days": 31,
  "n_stations": 59,
  "n_fsen": 4,
  "n_station_fsen_pairs": 236,
  "expected_rows_per_day": 236,
  "actual_rows_per_day": 33701.0,
  "over_factor": 142.80084745762713,
  "n_duplicate_fdia_fest_fsen": 1037414,
  "row_count_distribution_per_key": {
    "min": 1,
    "median": 1.0,
    "max": 1,
    "n_keys": 7316
  },
  "sample_station_names": [
    "ES01",
    "ES02",
    "ES03",
    "ES04",
    "ES05",
    "ES06",
    "ES07",
    "ES08",
    "ES09",
    "ES10"
  ],
  "sample_fsen_values": [
    "1-",
    "1=",
    "2-",
    "2=",
    NaN
  ]
}
