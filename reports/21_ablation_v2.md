# Source Audit Report

## Comparison (test = 2025)

| model | kappa_q | f1_macro | pr_auc_fatal | pr_auc_macro | Lift@2 | Lift@5 | Lift@10 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4_baseline | 0.5223 | 0.3679 | 0.00826 | 0.4413 | 10.926 | 6.249 | 4.062 | 0.3125 | 0.4062 |
| M4_plus_network | 0.5258 | 0.369 | 0.00893 | 0.444 | 10.926 | 6.249 | 4.999 | 0.3125 | 0.5 |
| M4_plus_network_te | 0.521 | 0.3677 | 0.00932 | 0.4439 | 10.926 | 5.624 | 4.687 | 0.2812 | 0.4688 |

## Bootstrap 95% CI (test, 1000 iters)

{
  "M4_baseline": {
    "Lift@2": {
      "median": 10.926,
      "ci95_low": 4.162,
      "ci95_high": 19.334
    },
    "Lift@5": {
      "median": 6.206,
      "ci95_low": 3.333,
      "ci95_high": 9.999
    },
    "Lift@10": {
      "median": 4.062,
      "ci95_low": 2.4981,
      "ci95_high": 5.937
    },
    "Recall@2": {
      "median": 0.2188,
      "ci95_low": 0.0833,
      "ci95_high": 0.3871
    },
    "Recall@5": {
      "median": 0.3103,
      "ci95_low": 0.1667,
      "ci95_high": 0.5
    },
    "Recall@10": {
      "median": 0.4062,
      "ci95_low": 0.24981,
      "ci95_high": 0.5938
    },
    "PR_AUC_fatal": {
      "median": 0.009010595426847064,
      "ci95_low": 0.004302168526396063,
      "ci95_high": 0.017735702853513856
    }
  },
  "M4_plus_network": {
    "Lift@2": {
      "median": 10.926,
      "ci95_low": 4.162,
      "ci95_high": 19.334
    },
    "Lift@5": {
      "median": 6.198,
      "ci95_low": 3.2,
      "ci95_high": 9.696425
    },
    "Lift@10": {
      "median": 4.999,
      "ci95_low": 3.330925,
      "ci95_high": 6.799
    },
    "Recall@2": {
      "median": 0.2188,
      "ci95_low": 0.0833,
      "ci95_high": 0.3871
    },
    "Recall@5": {
      "median": 0.3099,
      "ci95_low": 0.16,
      "ci95_high": 0.4848225
    },
    "Recall@10": {
      "median": 0.5,
      "ci95_low": 0.3330925,
      "ci95_high": 0.68
    },
    "PR_AUC_fatal": {
      "median": 0.009709485420490983,
      "ci95_low": 0.004572151022100794,
      "ci95_high": 0.019349906821694653
    }
  },
  "M4_plus_network_te": {
    "Lift@2": {
      "median": 10.926,
      "ci95_low": 4.162,
      "ci95_high": 19.334
    },
    "Lift@5": {
      "median": 5.632,
      "ci95_low": 2.758,
      "ci95_high": 9.229
    },
    "Lift@10": {
      "median": 4.999,
      "ci95_low": 3.214,
      "ci95_high": 7.026249999999999
    },
    "Recall@2": {
      "median": 0.2188,
      "ci95_low": 0.0833,
      "ci95_high": 0.3871
    },
    "Recall@5": {
      "median": 0.28165,
      "ci95_low": 0.1379,
      "ci95_high": 0.4615
    },
    "Recall@10": {
      "median": 0.5,
      "ci95_low": 0.3214,
      "ci95_high": 0.7027249999999999
    },
    "PR_AUC_fatal": {
      "median": 0.01006515714602466,
      "ci95_low": 0.004826073538432698,
      "ci95_high": 0.019735946730077503
    }
  }
}

## Deltas vs baseline (with CI)

{
  "M4_plus_network": {
    "Lift@2": {
      "point": 0.0,
      "ci95_delta_low": -15.172,
      "ci95_delta_high": 15.172
    },
    "Lift@5": {
      "point": 0.0,
      "ci95_delta_low": -6.799,
      "ci95_delta_high": 6.3634
    },
    "Lift@10": {
      "point": 0.937,
      "ci95_delta_low": -2.6061,
      "ci95_delta_high": 4.3009
    },
    "Recall@5": {
      "point": 0.0,
      "ci95_delta_low": -0.34,
      "ci95_delta_high": 0.3181
    },
    "Recall@10": {
      "point": 0.0938,
      "ci95_delta_low": -0.2607,
      "ci95_delta_high": 0.4302
    },
    "PR_AUC_fatal": {
      "point": 0.0007,
      "ci95_delta_low": -0.0132,
      "ci95_delta_high": 0.015
    }
  },
  "M4_plus_network_te": {
    "Lift@2": {
      "point": 0.0,
      "ci95_delta_low": -15.172,
      "ci95_delta_high": 15.172
    },
    "Lift@5": {
      "point": -0.625,
      "ci95_delta_low": -7.241,
      "ci95_delta_high": 5.896
    },
    "Lift@10": {
      "point": 0.625,
      "ci95_delta_low": -2.723,
      "ci95_delta_high": 4.5281
    },
    "Recall@5": {
      "point": -0.0313,
      "ci95_delta_low": -0.3621,
      "ci95_delta_high": 0.2948
    },
    "Recall@10": {
      "point": 0.0626,
      "ci95_delta_low": -0.2724,
      "ci95_delta_high": 0.4529
    },
    "PR_AUC_fatal": {
      "point": 0.0011,
      "ci95_delta_low": -0.0129,
      "ci95_delta_high": 0.0154
    }
  }
}

## Notes

- Splits (D081): train 2019-22, val 2023-24, test 2025.
- Test has ~34 fatal events (vs ~60 in original).
- Bootstrap 1000 iterations, resampling with replacement.
- `M4_plus_network_te` adds out-of-fold target encoding of edge_name and nombre_via.
