# Source Audit Report

## Boosting comparison (test = 2025)

| key | kappa_q | f1_macro | mae_ord | pr_auc_fatal | pr_auc_macro | rec_serious | rec_fatal | Lift@2 | Lift@5 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2__lgbm | 0.5415 | 0.3801 | 0.2942 | 0.00491 | 0.4461 | 0.024 | 0.0 | 3.122 | 3.125 | 0.1562 | 0.375 |
| M2__catboost | 0.5435 | 0.3862 | 0.2949 | 0.00582 | 0.4463 | 0.039 | 0.0 | 4.683 | 5.624 | 0.2812 | 0.4375 |
| M2+net__lgbm | 0.5466 | 0.3811 | 0.2903 | 0.0053 | 0.4503 | 0.021 | 0.0 | 3.122 | 4.999 | 0.25 | 0.3438 |
| M2+net__catboost | 0.5489 | 0.3878 | 0.2924 | 0.00633 | 0.4497 | 0.039 | 0.0 | 6.243 | 5.624 | 0.2812 | 0.375 |

## Best by kappa (then PR-AUC fatal)

M2+net__catboost

## Best params

{
  "bagging_temperature": 0.15599452033620265,
  "depth": 6,
  "iterations": 1982,
  "l2_leaf_reg": 10.275417738969418,
  "learning_rate": 0.06054365855469246,
  "random_strength": 0.679657809075816
}

## Bootstrap CI (best)

{
  "Lift@2": {
    "median": 6.445,
    "ci95_low": 1.536325,
    "ci95_high": 13.622
  },
  "Lift@5": {
    "median": 5.599,
    "ci95_low": 2.694975,
    "ci95_high": 9.331249999999999
  },
  "Recall@5": {
    "median": 0.28,
    "ci95_low": 0.1347725,
    "ci95_high": 0.46666249999999987
  },
  "Recall@10": {
    "median": 0.3704,
    "ci95_low": 0.20327750000000003,
    "ci95_high": 0.5762299999999998
  },
  "PR_AUC_fatal": {
    "median": 0.006748874954954396,
    "ci95_low": 0.0035966269267153346,
    "ci95_high": 0.012554506430715356
  }
}

## Baseline LR (from Fase 3, for reference)

M4 (LR, no network): kappa=0.522, PR-AUC=0.0083, Lift@2=10.9, Recall@10=0.406
M4+net (LR): kappa=0.526, PR-AUC=0.0089, Lift@2=10.9, Recall@10=0.500
