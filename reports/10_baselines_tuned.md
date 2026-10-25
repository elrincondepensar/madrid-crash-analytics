# Source Audit Report

## Ablation — features vs. tuning (val and test)

| model | split | kappa_q | f1_macro | bal_acc | mae_ord | pr_auc_macro | pr_auc_fatal | roc_auc_macro | rec_fatal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_minimal_default | val | 0.4951 | 0.361 | 0.3817 | 0.3177 | 0.4415 | 0.01141 | 0.8264 | 0.0 |
| M1_minimal_default | test | 0.5054 | 0.3635 | 0.384 | 0.3074 | 0.4406 | 0.00949 | 0.8285 | 0.0 |
| M2_full_default | val | 0.5021 | 0.3662 | 0.3854 | 0.3155 | 0.4434 | 0.02397 | 0.824 | 0.0 |
| M2_full_default | test | 0.5118 | 0.3673 | 0.3865 | 0.306 | 0.442 | 0.00953 | 0.8228 | 0.0 |
| M3_minimal_tuned | val | 0.4956 | 0.3611 | 0.3818 | 0.3174 | 0.4414 | 0.01156 | 0.8271 | 0.0 |
| M3_minimal_tuned | test | 0.5049 | 0.3634 | 0.3839 | 0.3076 | 0.4404 | 0.00925 | 0.8284 | 0.0 |
| M4_full_tuned | val | 0.502 | 0.3663 | 0.3854 | 0.3155 | 0.4424 | 0.01981 | 0.8244 | 0.0 |
| M4_full_tuned | test | 0.5118 | 0.3671 | 0.3865 | 0.306 | 0.4419 | 0.00927 | 0.8216 | 0.0 |

## Best model by test kappa

**M4_full_tuned**

Best params: {"C": 10.0, "class_weight": null}

CV kappa (LeaveOneGroupOut): 0.505456672571027

## Bootstrap 95% CI (best model, test)

{
  "f1_macro": {
    "median": 0.36708508250102867,
    "ci95_low": 0.3644510776573928,
    "ci95_high": 0.36952592976574483,
    "n_valid_resamples": 500
  },
  "pr_auc_fatal": {
    "median": 0.009819915883213664,
    "ci95_low": 0.006042118325067276,
    "ci95_high": 0.01857382327995238,
    "n_valid_resamples": 500
  }
}

## Confusion matrix (best model, test)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 20946 | 3210 | 10 | 0 |
| minor | 4511 | 9392 | 42 | 1 |
| serious | 864 | 3324 | 24 | 0 |
| fatal | 10 | 50 | 2 | 0 |

## Notes

- `pr_auc_fatal` is one-vs-rest AP for the fatal class (the operationally relevant question).
- `pr_auc_macro` is multiclass macro average of per-class AP.
- `roc_auc_macro` is multiclass macro of ROC-AUC (ovr).
- CV inside GridSearchCV is LeaveOneGroupOut over train years (2019–2022), NOT random KFold, to avoid temporal leakage.
