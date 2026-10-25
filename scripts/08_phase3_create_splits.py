"""
Phase 3 · Iteration 2a — Persist train/val/test splits.

Loads accidents_clean.parquet, assigns each accident to train (2019-2022),
val (2023) or test (2024-2025) and saves splits.parquet. This file is the
single source of truth for splits across all subsequent phases (D030, D034).

Run from the repository root:
    python scripts/08_phase3_create_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.config import year_to_split
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

DATA = Path("data/processed/accidents_clean.parquet")
OUT = Path("data/processed")
REPORTS_TABLES = Path("reports/tables")
REPORTS_TABLES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Loading {DATA}...")
    df = pd.read_parquet(DATA, columns=[
        "num_expediente", "year", "severity", "severity_code",
    ])
    print(f"  {len(df):,} accidents.")

    df["split"] = df["year"].astype(int).map(year_to_split)

    counts = df["split"].value_counts().to_dict()
    severity_by_split = (
        df.groupby(["split", "severity"]).size()
          .unstack(fill_value=0)
          .reindex(["train", "val", "test"])
          .to_dict(orient="index")
    )
    severity_share_by_split = (
        df.groupby(["split", "severity"]).size()
          .unstack(fill_value=0)
          .div(df.groupby("split").size(), axis=0)
          .round(4)
          .reindex(["train", "val", "test"])
          .to_dict(orient="index")
    )

    checks = {
        "counts_by_split": counts,
        "severity_counts_by_split": severity_by_split,
        "severity_share_by_split": severity_share_by_split,
        "n_duplicates": int(df["num_expediente"].duplicated().sum()),
        "n_out_of_range_years": int(
            (~df["split"].isin(["train", "val", "test"])).sum()
        ),
        "total": int(len(df)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    df[["num_expediente", "year", "severity", "severity_code", "split"]] \
        .to_parquet(OUT / "splits.parquet", index=False)

    save_json(checks, REPORTS_TABLES / "08_splits.json")

    sections = {
        "Counts by split": dumps_jsonable(counts, indent=2),
        "Severity counts by split": dumps_jsonable(severity_by_split, indent=2),
        "Severity share by split": dumps_jsonable(severity_share_by_split, indent=2),
        "Checks": dumps_jsonable(
            {k: v for k, v in checks.items()
             if k not in ("counts_by_split", "severity_counts_by_split",
                          "severity_share_by_split")}, indent=2),
    }
    save_markdown_report(sections, Path("reports/08_splits.md"))
    print(f"Done. Wrote {OUT / 'splits.parquet'} and reports/08_splits.md")


if __name__ == "__main__":
    main()