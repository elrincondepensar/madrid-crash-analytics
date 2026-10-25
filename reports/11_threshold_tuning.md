# Source Audit Report

## Comparison

| decisions | split | f1_macro | kappa_q | mae_ord | rec_no_inj | rec_minor | rec_serious | rec_fatal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| argmax | val | 0.4024 | 0.5052 | 0.3173 | 0.878 | 0.64 | 0.048 | 0.042 |
| argmax | test | 0.3773 | 0.5101 | 0.3103 | 0.874 | 0.647 | 0.033 | 0.0 |
| cost-sensitive | val | 0.104 | 0.076 | 1.2718 | 0.056 | 0.055 | 0.943 | 0.083 |
| cost-sensitive | test | 0.0993 | 0.0762 | 1.2822 | 0.052 | 0.053 | 0.951 | 0.081 |

## Confusion matrix — argmax (test)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 21123 | 2988 | 55 | 0 |
| minor | 4709 | 9019 | 217 | 1 |
| serious | 931 | 3139 | 141 | 1 |
| fatal | 10 | 43 | 9 | 0 |

## Confusion matrix — cost-sensitive (test)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 1261 | 6920 | 14228 | 1757 |
| minor | 6 | 735 | 12982 | 223 |
| serious | 1 | 96 | 4006 | 109 |
| fatal | 0 | 1 | 56 | 5 |

## Fatal threshold sweep (test)

| threshold | tp | fp | fn | precision | recall | n_positive_predictions |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 17 | 1294 | 45 | 0.013 | 0.2742 | 1311 |
| 0.02 | 3 | 323 | 59 | 0.0092 | 0.0484 | 326 |
| 0.05 | 1 | 59 | 61 | 0.0167 | 0.0161 | 60 |
| 0.1 | 0 | 5 | 62 | 0.0 | 0.0 | 5 |
| 0.2 | 0 | 2 | 62 | 0.0 | 0.0 | 2 |
| 0.3 | 0 | 2 | 62 | 0.0 | 0.0 | 2 |
| 0.5 | 0 | 2 | 62 | 0.0 | 0.0 | 2 |

## PR-AUC(fatal) on test (calibrated)

0.00801945739253739

## Notes

Cost matrix (per class): no_injury=0.1, minor=1, serious=10, fatal=100.
Cost-sensitive decision minimizes expected cost.
Threshold sweep shows precision-recall trade-off for the fatal class alone.
