# Source Audit Report

## IMD check (already fixed)

{
  "median": 11054.0,
  "max": 310006.0,
  "n_null": 0
}

## VIALES_SHP structure

{
  "n_features": 9408,
  "crs": "EPSG:25830",
  "columns": [
    "TOP_ID",
    "TOP_ID_COM",
    "TOP_ID_TER",
    "TVIA_ID",
    "TOP_DT_FCH",
    "TOP_DT_F_1",
    "CV_TX_DENO",
    "DEN_TX_PAR",
    "DEN_TX_NOM",
    "geometry"
  ],
  "geometry_types": {
    "Polygon": 8644,
    "MultiPolygon": 641
  },
  "sample_rows": [
    {
      "TOP_ID": 127,
      "TOP_ID_COM": 313425.0,
      "TOP_ID_TER": 313500.0,
      "TVIA_ID": 0.0,
      "TOP_DT_FCH": "1983-08-01 00:00:00",
      "TOP_DT_F_1": null,
      "CV_TX_DENO": "Calle",
      "DEN_TX_PAR": "de",
      "DEN_TX_NOM": "Isabel Colbrand"
    },
    {
      "TOP_ID": 150,
      "TOP_ID_COM": 173900.0,
      "TOP_ID_TER": 789500.0,
      "TVIA_ID": 0.0,
      "TOP_DT_FCH": "1985-12-27 00:00:00",
      "TOP_DT_F_1": null,
      "CV_TX_DENO": "Calle",
      "DEN_TX_PAR": "de",
      "DEN_TX_NOM": "González Dávila"
    },
    {
      "TOP_ID": 200,
      "TOP_ID_COM": 656500.0,
      "TOP_ID_TER": 342950.0,
      "TVIA_ID": 0.0,
      "TOP_DT_FCH": "1835-01-11 00:00:00",
      "TOP_DT_F_1": null,
      "CV_TX_DENO": "Calle",
      "DEN_TX_PAR": "de la",
      "DEN_TX_NOM": "Abada"
    }
  ]
}

## Notes

Source: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/CALLEJERO/VIALES/VIALES_SHP.zip
CRS is expected to be EPSG:25830 (UTM 30N).
If geometry is Polygon, viales are recintos (blocks); if LineString, they are axes.
