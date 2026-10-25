# Source Audit Report

## Comparison — test (thresholded per class on val)

| variant | kappa_q | f1_macro | mae_ord | rec_minor | rec_serious | rec_fatal | pr_auc_fatal | lift@1 | lift@5 | recall@5 | recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.5082 | 0.3891 | 0.3455 | 0.81 | 0.103 | 0.0 | 0.00927 | 9.674 | 7.417 | 0.371 | 0.5 |
| smote | 0.5142 | 0.3988 | 0.3518 | 0.778 | 0.151 | 0.0 | 0.00937 | 12.899 | 8.062 | 0.4032 | 0.4839 |
| borderline | 0.5058 | 0.391 | 0.3472 | 0.801 | 0.112 | 0.0 | 0.00744 | 4.837 | 5.16 | 0.2581 | 0.4516 |
| adasyn | 0.5146 | 0.3989 | 0.3513 | 0.778 | 0.151 | 0.0 | 0.00958 | 12.899 | 7.739 | 0.3871 | 0.5 |

## Fatal ranking on test (per variant)

| variant | k | tp | precision | recall | lift |
| --- | --- | --- | --- | --- | --- |
| baseline | k=1% | 6 | 0.0142 | 0.0968 | 9.674 |
| baseline | k=2% | 14 | 0.0165 | 0.2258 | 11.287 |
| baseline | k=5% | 23 | 0.0108 | 0.371 | 7.417 |
| baseline | k=10% | 31 | 0.0073 | 0.5 | 5.0 |
| baseline | k=20% | 44 | 0.0052 | 0.7097 | 3.548 |
| smote | k=1% | 8 | 0.0189 | 0.129 | 12.899 |
| smote | k=2% | 11 | 0.013 | 0.1774 | 8.868 |
| smote | k=5% | 25 | 0.0118 | 0.4032 | 8.062 |
| smote | k=10% | 30 | 0.0071 | 0.4839 | 4.838 |
| smote | k=20% | 40 | 0.0047 | 0.6452 | 3.226 |
| borderline | k=1% | 3 | 0.0071 | 0.0484 | 4.837 |
| borderline | k=2% | 5 | 0.0059 | 0.0806 | 4.031 |
| borderline | k=5% | 16 | 0.0075 | 0.2581 | 5.16 |
| borderline | k=10% | 28 | 0.0066 | 0.4516 | 4.516 |
| borderline | k=20% | 44 | 0.0052 | 0.7097 | 3.548 |
| adasyn | k=1% | 8 | 0.0189 | 0.129 | 12.899 |
| adasyn | k=2% | 12 | 0.0142 | 0.1935 | 9.674 |
| adasyn | k=5% | 24 | 0.0113 | 0.3871 | 7.739 |
| adasyn | k=10% | 31 | 0.0073 | 0.5 | 5.0 |
| adasyn | k=20% | 42 | 0.005 | 0.6774 | 3.387 |

## Thresholds selected on val

{
  "baseline": {
    "0": 0.0,
    "1": 0.301,
    "2": 0.168,
    "3": 0.143
  },
  "smote": {
    "0": 0.0,
    "1": 0.301,
    "2": 0.148,
    "3": 0.473
  },
  "borderline": {
    "0": 0.0,
    "1": 0.301,
    "2": 0.163,
    "3": 0.99
  },
  "adasyn": {
    "0": 0.0,
    "1": 0.301,
    "2": 0.148,
    "3": 0.576
  }
}

## Confusion matrices (test, thresholded)

**baseline** (test, thresholded)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 16667 | 7271 | 228 | 0 |
| minor | 1965 | 11301 | 679 | 1 |
| serious | 380 | 3398 | 434 | 0 |
| fatal | 5 | 41 | 16 | 0 |

**smote** (test, thresholded)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 16765 | 7070 | 330 | 1 |
| minor | 1999 | 10853 | 1089 | 5 |
| serious | 395 | 3177 | 637 | 3 |
| fatal | 7 | 33 | 22 | 0 |

**borderline** (test, thresholded)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 16743 | 7194 | 229 | 0 |
| minor | 2037 | 11166 | 742 | 1 |
| serious | 427 | 3315 | 470 | 0 |
| fatal | 7 | 40 | 15 | 0 |

**adasyn** (test, thresholded)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 16789 | 7050 | 327 | 0 |
| minor | 2005 | 10847 | 1092 | 2 |
| serious | 400 | 3174 | 636 | 2 |
| fatal | 7 | 33 | 22 | 0 |

## Notes

- Threshold per class tuned on val to maximize per-class F1 (one-vs-rest).
- Lift@k = (precision in top k%) / base_rate(fatal).
- Base fatal rate in test = 62 / 42386 = 0.00146.
- SMOTE sampling_strategy caps class fatal at 1.160 (1:10 vs majority) to avoid overfit from 116 real samples.
- SMOTE interpolates UTM coordinates linearly; documented as limitation (D054).
