# Source Audit Report

## 208223 raw structure

{
  "n_placemarks": 3928,
  "n_placemarks_with_linestring": 3928,
  "n_placemarks_with_coords": 3928,
  "n_placemarks_with_name": 0,
  "first_placemark_xml": "<Placemark xmlns=\"http://earth.google.com/kml/2.2\">\n      <styleUrl>#Color2</styleUrl>\n      <LineString>\n        <coordinates>-3.691893,40.389298,0 -3.690911,40.387566,0 </coordinates>\n      </LineString>\n    </Placemark>\n    \n"
}

## 203962 AÑO and TOTAL parsing

{
  "n_points": 1961,
  "year_counts": {
    "2013": 1460,
    "2012": 501
  },
  "total_parsed_stats": {
    "n_valid": 1961,
    "min": 1.018,
    "median": 11.804,
    "max": 995.0,
    "mean": 30.4
  },
  "rango_inte_counts": {
    "Entre 10.000-20.000 Veh/día": 530,
    "Entre 5.000-10.000 Veh/día": 502,
    "Entre 20.000-40.000 Veh/día": 370,
    "Entre 1.000-5.000 Veh/día": 338,
    "Entre 40.000-60.000 Veh/día": 109,
    "<  1.000 Veh/día": 51,
    "Entre 60.000-80.000 Veh/día": 37,
    "Entre 80.000-100.000 Veh/día": 14,
    "> 100.000 Veh/día": 10
  },
  "sample_TOTAL_values": [
    "7.354000000000000",
    "11.170999999999999",
    "5.860000000000000",
    "16.550000000000001",
    "20.675999999999998"
  ],
  "sample_TOTAL_parsed": [
    7.354,
    11.171,
    5.86,
    16.55,
    20.676
  ]
}

## Spatial join coverage

{
  "n_segments": 3928,
  "n_segments_with_imd": 1353,
  "n_segments_with_accidents": 3880,
  "segments_share_with_imd": 0.3445,
  "segments_share_with_accidents": 0.9878,
  "accidents_matching": {
    "n_matched": 101950,
    "share_matched": 0.7426,
    "dist_m_quantiles": {
      "50": 11.0,
      "90": 62.8,
      "99": 96.1,
      "max": 100.0
    }
  },
  "imd_total_per_segment_stats": {
    "n_notnull": 1353,
    "median": 13.1,
    "p90": 43.8
  },
  "n_accidents_by_segment_stats": {
    "n_with_0": 48,
    "n_with_1_to_5": 508,
    "n_with_6_to_20": 1714,
    "n_with_21_plus": 1658,
    "max": 490
  }
}

## Notes

- Segments and IMD points matched by nearest geometry (no attribute ID match, since 208223 has no attributes).
- Accidents limited to 100 m max distance from a segment.
- IMD year: check 'AÑO' distribution. If all 2013, the intensity is a 2013 snapshot, not 2019-2025.
