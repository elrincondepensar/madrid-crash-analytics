# Source Audit Report

## Comparison (test=2025, 32 fatales) — sorted by PR-AUC(fatal)

| key | kappa_q | f1_macro | mae_ord | pr_auc_fatal | pr_auc_macro | rec_serious | rec_fatal | Lift@2 | Lift@5 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm__borderline | 0.5561 | 0.3952 | 0.2956 | 0.00862 | 0.4391 | 0.065 | 0.0 | 6.243 | 4.999 | 0.25 | 0.4375 |
| catboost__smote | 0.553 | 0.3952 | 0.2959 | 0.00746 | 0.4394 | 0.065 | 0.0 | 9.365 | 6.874 | 0.3438 | 0.5 |
| lgbm__smote | 0.5545 | 0.3947 | 0.296 | 0.00728 | 0.4377 | 0.065 | 0.0 | 6.243 | 4.999 | 0.25 | 0.4688 |
| catboost__adasyn | 0.5533 | 0.3943 | 0.2963 | 0.00682 | 0.4398 | 0.064 | 0.0 | 7.804 | 4.999 | 0.25 | 0.4688 |
| lgbm__adasyn | 0.5508 | 0.395 | 0.2972 | 0.00631 | 0.4376 | 0.066 | 0.0 | 7.804 | 4.374 | 0.2188 | 0.4062 |
| catboost__none | 0.5522 | 0.3922 | 0.2959 | 0.00606 | 0.439 | 0.058 | 0.0 | 4.683 | 5.624 | 0.2812 | 0.4062 |
| catboost__borderline | 0.5547 | 0.3966 | 0.2961 | 0.00535 | 0.439 | 0.07 | 0.0 | 3.122 | 4.374 | 0.2188 | 0.4062 |
| lgbm__none | 0.5509 | 0.3915 | 0.2968 | 0.00534 | 0.4377 | 0.056 | 0.0 | 6.243 | 3.125 | 0.1562 | 0.3438 |

## Best model

lgbm__borderline

## Best params

{
  "n_estimators": 730,
  "learning_rate": 0.011789380292701837,
  "num_leaves": 30,
  "max_depth": 10,
  "min_child_samples": 98,
  "reg_alpha": 0.005801324141997622,
  "reg_lambda": 3.8777011416555904,
  "colsample_bytree": 0.9375980314442116,
  "subsample": 0.7606129372551605
}

## Bootstrap CI (best)

{
  "Lift@2": {
    "median": 6.243,
    "ci95_low": 0.6084750000000018,
    "ci95_high": 12.487
  },
  "Lift@5": {
    "median": 4.848,
    "ci95_low": 1.875,
    "ci95_high": 8.57
  },
  "Recall@5": {
    "median": 0.2424,
    "ci95_low": 0.0938,
    "ci95_high": 0.4286
  },
  "Recall@10": {
    "median": 0.43285,
    "ci95_low": 0.26006,
    "ci95_high": 0.640525
  },
  "PR_AUC_fatal": {
    "median": 0.00948268357763966,
    "ci95_low": 0.003977758343149188,
    "ci95_high": 0.026760629110380452
  }
}

## Tuning time per combo (s)

{
  "lgbm__none": 334.5,
  "lgbm__smote": 736.0,
  "lgbm__borderline": 600.4,
  "lgbm__adasyn": 1399.9,
  "catboost__none": 1330.8,
  "catboost__smote": 2754.2,
  "catboost__borderline": 2167.7,
  "catboost__adasyn": 2915.1
}

## Reference (from iter 21, LR baseline)

LR M4+net: kappa=0.526, PR-AUC(fatal)=0.0089, Lift@2=10.9

## Notes

Augmentation applied inside each fold (no leakage).
SMOTE strategy caps: serious→12k, fatal→2k.
Refit final sin early stopping (usa best params de Optuna).
Tuning time real por combo está en 'Tuning time per combo'.
