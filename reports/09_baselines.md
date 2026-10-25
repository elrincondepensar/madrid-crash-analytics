# Source Audit Report

## Splits

{
  "train": 74270,
  "val": 20672,
  "test": 42386
}

## Majority class

{
  "code": 0,
  "label": "no_injury"
}

## Comparison — val and test

| model                     | split   |   f1_macro |   balanced_acc |   kappa_q |   mae_ord |   pr_auc_fatal |   recall_fatal |
|:--------------------------|:--------|-----------:|---------------:|----------:|----------:|---------------:|---------------:|
| baseline_0_majority       | val     |     0.179  |         0.25   |    0      |    0.5492 |        0.00116 |          0     |
| baseline_0_majority       | test    |     0.1816 |         0.25   |    0      |    0.5322 |        0.00146 |          0     |
| baseline_1a_lr_balanced   | val     |     0.3716 |         0.5044 |    0.444  |    0.554  |        0.01148 |          0.583 |
| baseline_1a_lr_balanced   | test    |     0.3743 |         0.5002 |    0.4468 |    0.5433 |        0.00762 |          0.548 |
| baseline_1b_lr_unweighted | val     |     0.361  |         0.3817 |    0.4951 |    0.3177 |        0.01141 |          0     |
| baseline_1b_lr_unweighted | test    |     0.3635 |         0.384  |    0.5054 |    0.3074 |        0.00949 |          0     |

## Bootstrap 95% CI (test)

{
  "baseline_0_majority": {
    "f1_macro": {
      "median": 0.18158837712394382,
      "ci95_low": 0.18071038284547747,
      "ci95_high": 0.18251138542665388,
      "n_valid_resamples": 500
    },
    "pr_auc_fatal": {
      "median": 0.0014627471334874723,
      "ci95_low": 0.0011088566979663096,
      "ci95_high": 0.001793044873307224,
      "n_valid_resamples": 500
    }
  },
  "baseline_1a_lr_balanced": {
    "f1_macro": {
      "median": 0.37436793113633776,
      "ci95_low": 0.36976945524993116,
      "ci95_high": 0.37802373774167286,
      "n_valid_resamples": 500
    },
    "pr_auc_fatal": {
      "median": 0.007914735763989437,
      "ci95_low": 0.005158901103706198,
      "ci95_high": 0.012002489959136966,
      "n_valid_resamples": 500
    }
  },
  "baseline_1b_lr_unweighted": {
    "f1_macro": {
      "median": 0.3634947984222985,
      "ci95_low": 0.3613128376594689,
      "ci95_high": 0.3655970254867244,
      "n_valid_resamples": 500
    },
    "pr_auc_fatal": {
      "median": 0.009745711062462624,
      "ci95_low": 0.006471048422478196,
      "ci95_high": 0.014717273097406955,
      "n_valid_resamples": 500
    }
  }
}

## Confusion matrices (test)

**baseline_0_majority** (test)

|           |   no_injury |   minor |   serious |   fatal |
|:----------|------------:|--------:|----------:|--------:|
| no_injury |       24166 |       0 |         0 |       0 |
| minor     |       13946 |       0 |         0 |       0 |
| serious   |        4212 |       0 |         0 |       0 |
| fatal     |          62 |       0 |         0 |       0 |

**baseline_1a_lr_balanced** (test)

|           |   no_injury |   minor |   serious |   fatal |
|:----------|------------:|--------:|----------:|--------:|
| no_injury |       19572 |    2586 |       845 |    1163 |
| minor     |        3744 |    4353 |      3724 |    2125 |
| serious   |         676 |    1075 |      1391 |    1070 |
| fatal     |           8 |       4 |        16 |      34 |

**baseline_1b_lr_unweighted** (test)

|           |   no_injury |   minor |   serious |   fatal |
|:----------|------------:|--------:|----------:|--------:|
| no_injury |       21101 |    3061 |         4 |       0 |
| minor     |        4704 |    9238 |         4 |       0 |
| serious   |         908 |    3303 |         1 |       0 |
| fatal     |          10 |      52 |         0 |       0 |
