# Source Audit Report

## Year summary

file_year  n_rows  n_expedientes  rows_per_expediente
      2019   51811          21934             2.362132
      2020   32433          14172             2.288527
      2021   41783          17952             2.327484
      2022   47053          20212             2.327973
      2023   48830          20672             2.362132
      2024   49340          20698             2.383805
      2025   51067          21688             2.354620

## Year consistency

{
  "n_mismatch_file_vs_fecha": 0,
  "n_mismatch_file_vs_prefix": 21,
  "mismatch_file_vs_prefix_by_file": {
    "2019": 2,
    "2020": 2,
    "2021": 5,
    "2024": 11,
    "2025": 1
  },
  "mismatch_file_vs_prefix_prefix_counts": {
    "2023": 11,
    "2020": 5,
    "2018": 2,
    "2019": 2,
    "2024": 1
  }
}

## num_expediente

{
  "n_rows": 322317,
  "n_unique": 137328,
  "n_null": 0,
  "n_collision_cross_year": 0,
  "inconsistent_accidents": {
    "more_than_one_fecha": 0,
    "more_than_one_localizacion": 0,
    "more_than_one_x": 0,
    "more_than_one_y": 0
  }
}

## Severity counts

{
  "2019": {
    "no_injury": 11241,
    "minor": 8329,
    "serious": 2331,
    "fatal": 33
  },
  "2020": {
    "no_injury": 7357,
    "minor": 5251,
    "serious": 1530,
    "fatal": 34
  },
  "2021": {
    "no_injury": 9642,
    "minor": 6399,
    "serious": 1883,
    "fatal": 28
  },
  "2022": {
    "no_injury": 10917,
    "minor": 7230,
    "serious": 2044,
    "fatal": 21
  },
  "2023": {
    "no_injury": 11532,
    "minor": 6951,
    "serious": 2165,
    "fatal": 24
  },
  "2024": {
    "no_injury": 11644,
    "minor": 6893,
    "serious": 2133,
    "fatal": 28
  },
  "2025": {
    "no_injury": 12522,
    "minor": 7053,
    "serious": 2079,
    "fatal": 34
  }
}

## Severity shares

{
  "2019": {
    "no_injury": 0.5125,
    "minor": 0.3797,
    "serious": 0.1063,
    "fatal": 0.0015
  },
  "2020": {
    "no_injury": 0.5191,
    "minor": 0.3705,
    "serious": 0.108,
    "fatal": 0.0024
  },
  "2021": {
    "no_injury": 0.5371,
    "minor": 0.3565,
    "serious": 0.1049,
    "fatal": 0.0016
  },
  "2022": {
    "no_injury": 0.5401,
    "minor": 0.3577,
    "serious": 0.1011,
    "fatal": 0.001
  },
  "2023": {
    "no_injury": 0.5579,
    "minor": 0.3363,
    "serious": 0.1047,
    "fatal": 0.0012
  },
  "2024": {
    "no_injury": 0.5626,
    "minor": 0.333,
    "serious": 0.1031,
    "fatal": 0.0014
  },
  "2025": {
    "no_injury": 0.5774,
    "minor": 0.3252,
    "serious": 0.0959,
    "fatal": 0.0016
  }
}

## No-injury breakdown

{
  "n_accidents_all_nan": 57489,
  "n_accidents_some_known": 79839,
  "share_all_nan": 0.4186
}

## People per accident

{
  "mean": 2.347059594547361,
  "median": 2.0,
  "p95": 5.0,
  "max": 44,
  "distribution_head": {
    "1": 21789,
    "2": 75166,
    "3": 24223,
    "4": 9208,
    "5": 3873,
    "6": 1688,
    "7": 706,
    "8": 352,
    "9": 155,
    "10": 71
  }
}

## Coordinates

{
  "n_null_x": 62,
  "n_null_y": 62,
  "n_zero_x": 0,
  "n_zero_y": 0,
  "share_in_range_overall": 0.9998,
  "share_in_range_by_year": {
    "2019": 1.0,
    "2020": 0.9998,
    "2021": 0.9997,
    "2022": 0.9997,
    "2023": 0.9997,
    "2024": 0.9998,
    "2025": 0.9997
  }
}
