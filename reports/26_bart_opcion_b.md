# Source Audit Report

## Comparación LR vs BART (test=2025)

| model | kappa_q | f1_macro | mae_ord | pr_auc_grave | roc_auc_grave | rec_grave | rec_minor | rec_no_inj | fatal_Lift@2 | fatal_Lift@5 | fatal_Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR (C=10) | 0.0064 | 0.2543 | 0.5192 | 0.1105 | 0.5316 | 0.001 | 0.016 | 0.988 | 0.0 | 1.25 | 0.1875 |
| BART (smoke, solo grave) | nan | nan | nan | 0.1659 | nan | nan | nan | nan | 3.122 | 2.5 | 0.25 |

## Bootstrap CI — LR

{
  "pr_auc_grave": {
    "median": 0.11093867808573044,
    "ci95_low": 0.10314437951861903,
    "ci95_high": 0.11892184523158283
  },
  "recall_grave": {
    "median": 0.0009409551214653312,
    "ci95_low": 0.0,
    "ci95_high": 0.002425475287317392
  },
  "precision_grave": {
    "median": 0.3333333333333333,
    "ci95_low": 0.0,
    "ci95_high": 0.8
  }
}

## Bootstrap CI — BART

N/A (smoke)

## Fatal ranking — LR

{
  "n_fatal": 32,
  "by_k": {
    "k=2%": {
      "k": 434,
      "tp": 0,
      "precision": 0.0,
      "recall": 0.0,
      "lift": 0.0
    },
    "k=5%": {
      "k": 1084,
      "tp": 2,
      "precision": 0.0018,
      "recall": 0.0625,
      "lift": 1.25
    },
    "k=10%": {
      "k": 2168,
      "tp": 6,
      "precision": 0.0028,
      "recall": 0.1875,
      "lift": 1.875
    }
  }
}

## Fatal ranking — BART

{
  "n_fatal": 32,
  "by_k": {
    "k=2%": {
      "k": 434,
      "tp": 2,
      "precision": 0.0046,
      "recall": 0.0625,
      "lift": 3.122
    },
    "k=5%": {
      "k": 1084,
      "tp": 4,
      "precision": 0.0037,
      "recall": 0.125,
      "lift": 2.5
    },
    "k=10%": {
      "k": 2168,
      "tp": 8,
      "precision": 0.0037,
      "recall": 0.25,
      "lift": 2.5
    }
  }
}

## BART fits (por clase)

{
  "grave": {
    "n_positive_train": 7900,
    "fit_seconds": 77.6
  }
}

## Notes

Target: 0=no_injury, 1=minor, 2=grave (serious ∪ fatal).
Mismo preprocesado (M4: imputer + OHE), fit en train.
Mismas features (M2 + numero + red + TE edge_name/nombre_via).
LR usa todas las features. BART usa todas en full, top-K en smoke.
fatal se evalúa como ranking usando P(grave) como score.
