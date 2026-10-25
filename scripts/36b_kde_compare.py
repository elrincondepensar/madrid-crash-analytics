"""
Script 36b — Fix of naming inconsistency in Option B, and complete
the C vs B comparison that script 36 failed to run.

Root cause (script 36): Option B columns were saved with the prefix
`kde_b_*`, while Option C uses `kde_*`. The comparison code expects
identical column names in both dataframes, so it found zero matches.

This script:
  1. Loads network_kde_b.parquet (already computed, ~20 min of compute
     that we do NOT re-run).
  2. Renames columns kde_b_* -> kde_* .
  3. Persists the renamed parquet back to disk (consistency).
  4. Loads network_kde_c.parquet.
  5. Runs the C vs B comparison (Spearman rho + Jaccard on top-20%).
  6. Applies the decision rule (D-36-3).
  7. Builds network_kde_final.parquet with the winning option at bw=260.
  8. Writes reports/tables/36b_kde_compare.json and reports/36b_kde_compare.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_TABLES = REPORTS / "tables"

KDE_B_PATH = DATA_PROCESSED / "network_kde_b.parquet"
KDE_C_PATH = DATA_PROCESSED / "network_kde_c.parquet"
OUT_FINAL = DATA_PROCESSED / "network_kde_final.parquet"
OUT_JSON = REPORTS_TABLES / "36b_kde_compare.json"
OUT_MD = REPORTS / "36b_kde_compare.md"

BW_VALUES = (100, 150, 200, 260, 350, 500)
BW_PRIMARY = 260
PERIODS = ("train", "val", "test")
SEVERITIES = ("all", "injured")

log = logging.getLogger("phase7_compare_fix")


# ---------------------------------------------------------------------------
# Load & normalize
# ---------------------------------------------------------------------------

def _normalize_kde_col(name: str) -> str:
    """Normalize a KDE column name to `kde_{period}_{severity}_bw{bw}`.

    Handles three input forms that may appear depending on how the parquet
    was written historically:
      - `kde_b_{period}_{severity}_bw{bw}` (script 36 original)  -> strip `kde_b_`, add `kde_`
      - `kde_{period}_{severity}_bw{bw}`  (correct)               -> unchanged
      - `{period}_{severity}_bw{bw}`      (bug from script 36b)   -> add `kde_`
    `edge_id` passes through unchanged.
    """
    if not isinstance(name, str):
        return name
    if name == "edge_id":
        return name
    if name.startswith("kde_b_"):
        return "kde_" + name[len("kde_b_"):]
    if name.startswith("kde_"):
        return name
    if any(name.startswith(p) for p in ("train_", "val_", "test_")):
        return "kde_" + name
    return name


def load_b_normalized() -> pd.DataFrame:
    """Load network_kde_b.parquet and normalize column names to `kde_*`."""
    log.info("Loading network_kde_b.parquet ...")
    b = pd.read_parquet(KDE_B_PATH)
    old_cols = list(b.columns)
    renamed = b.rename(columns=_normalize_kde_col)
    new_cols = list(renamed.columns)

    n_changed = sum(1 for o, n in zip(old_cols, new_cols) if o != n)
    log.info(f"  normalized {n_changed} columns")
    log.info(f"  sample: {old_cols[1]} -> {new_cols[1]}, "
             f"{old_cols[-1]} -> {new_cols[-1]}")

    # Sanity: no leftover non-`kde_*` / non-`edge_id` columns
    bad = [c for c in renamed.columns
           if c != "edge_id" and not c.startswith("kde_")]
    if bad:
        raise RuntimeError(f"unexpected columns after normalize: {bad[:5]}...")

    # Persist only if something changed
    if n_changed > 0:
        renamed.to_parquet(KDE_B_PATH, index=False)
        log.info(f"  saved normalized {KDE_B_PATH.name}")
    else:
        log.info("  no changes needed")
    return renamed


def load_c() -> pd.DataFrame:
    log.info("Loading network_kde_c.parquet ...")
    return pd.read_parquet(KDE_C_PATH)


def validate_columns(kde_c: pd.DataFrame, kde_b: pd.DataFrame) -> list:
    """Confirm that all required (period, severity, bw) columns exist in both."""
    missing_c, missing_b = [], []
    required = []
    for bw in BW_VALUES:
        for period in PERIODS:
            for sev in SEVERITIES:
                col = f"kde_{period}_{sev}_bw{bw}"
                required.append(col)
                if col not in kde_c.columns:
                    missing_c.append(col)
                if col not in kde_b.columns:
                    missing_b.append(col)
    if missing_c or missing_b:
        raise RuntimeError(
            f"column mismatch —\n"
            f"  missing in C: {missing_c}\n"
            f"  missing in B: {missing_b}"
        )
    log.info(f"  validated: all {len(required)} required columns present in both")
    return required


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() < 100:
        return None
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def _jaccard_top20(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() < 100:
        return None
    aa = a[mask]
    bb = b[mask]
    thr_a = np.percentile(aa, 80)
    thr_b = np.percentile(bb, 80)
    top_a = aa >= thr_a
    top_b = bb >= thr_b
    union = (top_a | top_b).sum()
    if union == 0:
        return None
    return float((top_a & top_b).sum() / union)


def compare(kde_c: pd.DataFrame, kde_b: pd.DataFrame,
            columns: list) -> dict:
    per_combo = {}
    rhos, jacs = [], []
    for col in columns:
        a = kde_c[col].to_numpy(dtype=np.float64)
        b = kde_b[col].to_numpy(dtype=np.float64)
        rho = _spearman(a, b)
        jac = _jaccard_top20(a, b)
        per_combo[col] = {"rho": rho, "jaccard_top20": jac}
        if rho is not None:
            rhos.append(rho)
        if jac is not None:
            jacs.append(jac)
        log.info(f"  {col:32s}  rho={_fmt(rho)}  jaccard={_fmt(jac)}")
    return {
        "per_combo": per_combo,
        "n_comparisons": len(rhos),
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "min_rho": float(np.min(rhos)) if rhos else None,
        "max_rho": float(np.max(rhos)) if rhos else None,
        "mean_jaccard_top20": float(np.mean(jacs)) if jacs else None,
        "min_jaccard_top20": float(np.min(jacs)) if jacs else None,
        "max_jaccard_top20": float(np.max(jacs)) if jacs else None,
    }


def decide(c_vs_b: dict) -> dict:
    mean_rho = c_vs_b["mean_rho"]
    mean_jac = c_vs_b["mean_jaccard_top20"]
    if mean_rho is None or mean_jac is None:
        return {"winner": "unknown", "reason": "missing metrics"}
    if mean_rho >= 0.95 and mean_jac >= 0.70:
        return {"winner": "C",
                "reason": (f"C and B agree strongly "
                           f"(mean rho={mean_rho:.3f}, "
                           f"mean jaccard={mean_jac:.3f}); "
                           f"C is simpler (raw accident points).")}
    return {"winner": "B",
            "reason": (f"C and B diverge "
                       f"(mean rho={mean_rho:.3f}, "
                       f"mean jaccard={mean_jac:.3f}); "
                       f"B uses the network-aligned source.")}


# ---------------------------------------------------------------------------
# Build final parquet
# ---------------------------------------------------------------------------

def build_final(kde_c: pd.DataFrame, kde_b: pd.DataFrame,
                winner: str) -> pd.DataFrame:
    src = kde_c if winner == "C" else kde_b
    out = src[["edge_id"]].copy()
    for period in PERIODS:
        for sev in SEVERITIES:
            col = f"kde_{period}_{sev}_bw{BW_PRIMARY}"
            if col not in src.columns:
                raise RuntimeError(f"missing {col} in winning option {winner}")
            out[col] = src[col].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(results: dict) -> str:
    lines = ["# Phase 7 · Iter 3b — C vs B comparison (fixed)\n"]
    lines.append(
        "_Fix of naming inconsistency in script 36: Option B columns now "
        "share names with Option C (`kde_*`). `network_kde_b.parquet` was "
        "renamed in place; no Option B recomputation._\n"
    )

    c = results["c_vs_b"]
    lines.append("## C vs B comparison\n")
    lines.append(
        f"Compared over **{c['n_comparisons']}** (bw, period, severity) "
        f"combinations.\n"
    )
    lines.append("| metric | mean | min | max |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| Spearman rho (C, B) | {_fmt(c['mean_rho'])} | "
                 f"{_fmt(c['min_rho'])} | {_fmt(c['max_rho'])} |")
    lines.append(f"| Jaccard (top-20% dense) | "
                 f"{_fmt(c['mean_jaccard_top20'])} | "
                 f"{_fmt(c['min_jaccard_top20'])} | "
                 f"{_fmt(c['max_jaccard_top20'])} |")
    lines.append(f"\n**Winner: `{results['decision']['winner']}`** — "
                 f"{results['decision']['reason']}\n")

    lines.append("## Per-combination table\n")
    lines.append("| col | rho | jaccard_top20 |")
    lines.append("| --- | ---: | ---: |")
    for col, v in sorted(c["per_combo"].items()):
        lines.append(f"| `{col}` | {_fmt(v['rho'])} | "
                     f"{_fmt(v['jaccard_top20'])} |")
    lines.append("")

    lines.append("## Decision rule (D-36-3)\n")
    lines.append(
        "- `mean_rho ≥ 0.95` **AND** `mean_jaccard_top20 ≥ 0.70` → "
        "**C wins** (simpler)."
    )
    lines.append("- Otherwise → **B wins** (network-aligned source).")
    lines.append(
        f"- `network_kde_final.parquet` contains only `{results['decision']['winner']}` "
        f"at bw={BW_PRIMARY} m (6 columns: 3 periods × 2 severities)."
    )

    lines.append("\n## Notes\n")
    lines.append(
        "- **Option B is 20 min of compute** (36 KDE jobs with sample_weight, "
        "ball_tree, 95k eval points each). Saved to disk in script 36; this "
        "script only renamed columns."
    )
    lines.append(
        "- **Jaccard** = fraction of top-20% dense edges the two options "
        "agree on. High = identical hotspot rankings at the top."
    )
    lines.append(
        "- **Both options are 2D KDE with Gaussian kernel**, differing only "
        "in the source representation: C uses 137k raw accident points, "
        "B uses ~10-20k edge midpoints weighted by their accident count. "
        "Total mass is comparable; B loses within-edge dispersion."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    kde_b = load_b_normalized()
    kde_c = load_c()
    columns = validate_columns(kde_c, kde_b)

    log.info("Comparing ...")
    c_vs_b = compare(kde_c, kde_b, columns)
    log.info(f"  mean rho={_fmt(c_vs_b['mean_rho'])}  "
             f"mean jaccard={_fmt(c_vs_b['mean_jaccard_top20'])}")

    decision = decide(c_vs_b)
    log.info(f"  DECISION: {decision['winner']} — {decision['reason']}")

    log.info(f"Building network_kde_final.parquet (option {decision['winner']}, "
             f"bw={BW_PRIMARY}) ...")
    final = build_final(kde_c, kde_b, decision["winner"])
    final.to_parquet(OUT_FINAL, index=False)
    log.info(f"  saved {OUT_FINAL.relative_to(ROOT)} "
             f"({len(final):,} rows × {len(final.columns)} cols)")

    results_json = {
        "bandwidths_m": list(BW_VALUES),
        "bw_primary": BW_PRIMARY,
        "periods": list(PERIODS),
        "severities": list(SEVERITIES),
        "n_edges": int(len(kde_c)),
        "c_vs_b": c_vs_b,
        "decision": decision,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results_json), encoding="utf-8")
    log.info(f"saved {OUT_JSON.relative_to(ROOT)}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(results_json), encoding="utf-8")
    log.info(f"saved {OUT_MD.relative_to(ROOT)}")

    log.info("=" * 72)
    log.info(f"WINNER: {decision['winner']}")
    log.info(f"  mean rho={_fmt(c_vs_b['mean_rho'])}  "
             f"mean jaccard={_fmt(c_vs_b['mean_jaccard_top20'])}")


if __name__ == "__main__":
    main()