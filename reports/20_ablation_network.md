# Source Audit Report

## Ablation comparison (test)

| model | kappa_q | f1_macro | mae_ord | pr_auc_fatal | pr_auc_macro | roc_auc_macro | rec_serious | rec_fatal | Lift@2 | Lift@5 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4_baseline | 0.5163 | 0.4004 | 0.3514 | 0.0093 | 0.442 | 0.8244 | 0.158 | 0.0 | 11.658 | 7.331 | 0.367 | 0.483 |
| M4_plus_network | 0.539 | 0.415 | 0.3374 | 0.01342 | 0.445 | 0.8278 | 0.2 | 0.0 | 14.156 | 7.331 | 0.367 | 0.533 |

## Best model by test kappa

M4_plus_network

## Thresholds per class (tuned on val)

{
  "M4_baseline": {
    "1": 0.305,
    "2": 0.148,
    "3": 0.173
  },
  "M4_plus_network": {
    "1": 0.34,
    "2": 0.158,
    "3": 0.217
  }
}

## Notes

- Both models use default LR(C=10) with same temporal splits.
- Threshold per class tuned on val; applied to test.
- Success criterion (D039): PR-AUC(fatal) >= 0.03. Also Lift@5 must increase >= +2 pp absolute over M4 baseline.
- 10 accidents outside Madrid bbox + 1 far outlier removed.
