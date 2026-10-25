"""
Phase 2 · Iteration 1 (v2) — Diagnostic of the accidents base.

Fixes the severity derivation (worst victim wins with the correct mapping)
and adds three diagnostics: year-of-file vs year-of-date, NaN-only accidents,
and the post-correction severity distribution.

Run:
    python scripts/05_phase2_accidents_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from madrid_safety.accidents import load_all_accidents, derive_severity
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

OUT = Path("reports/tables")
OUT.mkdir(parents=True, exist_ok=True)

UTM_X_MIN, UTM_X_MAX = 425_000, 460_000
UTM_Y_MIN, UTM_Y_MAX = 4_460_000, 4_490_000


def diag_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("file_year")
             .agg(n_rows=("num_expediente", "size"),
                  n_expedientes=("num_expediente", "nunique"))
             .reset_index())
    out["rows_per_expediente"] = out["n_rows"] / out["n_expedientes"]
    return out


def diag_year_consistency(df: pd.DataFrame) -> dict:
    """File year vs fecha-year vs expedition-number prefix."""
    prefix = df["num_expediente"].astype(str).str[:4].astype(int)
    d = df.assign(prefix_year=prefix)
    return {
        "n_mismatch_file_vs_fecha": int((d["file_year"] != d["year"]).sum()),
        "n_mismatch_file_vs_prefix": int((d["file_year"] != d["prefix_year"]).sum()),
        "mismatch_file_vs_prefix_by_file":
            d.loc[d["file_year"] != d["prefix_year"]]
             .groupby("file_year").size().to_dict(),
        "mismatch_file_vs_prefix_prefix_counts":
            d.loc[d["file_year"] != d["prefix_year"]]
             ["prefix_year"].value_counts().to_dict(),
    }


def diag_num_expediente(df: pd.DataFrame) -> dict:
    s = df["num_expediente"].astype(str)
    per_year = df.groupby("num_expediente")["year"].nunique()
    consist = (df.groupby("num_expediente")
                 .agg(n_dates=("fecha", "nunique"),
                      n_locs=("localizacion", "nunique"),
                      n_x=("coordenada_x_utm", "nunique"),
                      n_y=("coordenada_y_utm", "nunique")))
    return {
        "n_rows": int(len(df)),
        "n_unique": int(s.nunique()),
        "n_null": int(df["num_expediente"].isna().sum()),
        "n_collision_cross_year": int((per_year > 1).sum()),
        "inconsistent_accidents": {
            "more_than_one_fecha": int((consist["n_dates"] > 1).sum()),
            "more_than_one_localizacion": int((consist["n_locs"] > 1).sum()),
            "more_than_one_x": int((consist["n_x"] > 1).sum()),
            "more_than_one_y": int((consist["n_y"] > 1).sum()),
        },
    }


def diag_lesividad(df: pd.DataFrame) -> dict:
    return {
        "crosstab": pd.crosstab(
            df["cod_lesividad"].fillna(-1),
            df["lesividad"].fillna("__NaN__"),
        ).reset_index().to_dict(orient="records"),
        "cod_lesividad_by_year": (
            df.assign(cod_lesividad=df["cod_lesividad"].fillna(-1).astype(int))
              .groupby(["year", "cod_lesividad"]).size()
              .unstack(fill_value=0).to_dict(orient="index")),
    }


def diag_people_per_accident(df: pd.DataFrame) -> dict:
    n = df.groupby("num_expediente").size()
    return {
        "mean": float(n.mean()),
        "median": float(n.median()),
        "p95": float(n.quantile(0.95)),
        "max": int(n.max()),
        "distribution_head": n.value_counts().sort_index().head(10).to_dict(),
    }


def diag_severity(sev: pd.DataFrame) -> dict:
    order = ["no_injury", "minor", "serious", "fatal", "unknown"]
    counts = (sev.groupby(["year", "severity"]).size()
                .unstack(fill_value=0)
                .reindex(columns=[c for c in order if c in sev["severity"].unique()],
                         fill_value=0))
    return {
        "counts_by_year": counts.to_dict(orient="index"),
        "share_by_year": (counts.div(counts.sum(axis=1), axis=0)
                                .round(4).to_dict(orient="index")),
    }


def diag_no_injury_breakdown(df: pd.DataFrame) -> dict:
    """How many accidents have all cod_lesividad = NaN (no injury recorded)."""
    all_nan = df.groupby("num_expediente")["cod_lesividad"].apply(
        lambda s: s.isna().all())
    return {
        "n_accidents_all_nan": int(all_nan.sum()),
        "n_accidents_some_known": int((~all_nan).sum()),
        "share_all_nan": round(float(all_nan.mean()), 4),
    }


def diag_coordinates(df: pd.DataFrame) -> dict:
    x = df["coordenada_x_utm"]
    y = df["coordenada_y_utm"]
    in_range = x.between(UTM_X_MIN, UTM_X_MAX) & y.between(UTM_Y_MIN, UTM_Y_MAX)
    return {
        "n_null_x": int(x.isna().sum()),
        "n_null_y": int(y.isna().sum()),
        "n_zero_x": int((x == 0).sum()),
        "n_zero_y": int((y == 0).sum()),
        "share_in_range_overall": round(float(in_range.mean()), 4),
        "share_in_range_by_year": (df.assign(in_range=in_range)
                                     .groupby("year")["in_range"]
                                     .mean().round(4).to_dict()),
    }


def diag_nulls(df: pd.DataFrame) -> dict:
    return {
        "n_rows_by_year": df.groupby("year").size().to_dict(),
        "nulls_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
    }


def diag_categorical(df: pd.DataFrame) -> dict:
    cols = ["tipo_persona", "tipo_vehiculo", "tipo_accidente",
            "positiva_alcohol", "positiva_droga", "sexo", "rango_edad"]
    return {c: df[c].value_counts(dropna=False).head(15).to_dict()
            for c in cols if c in df.columns}


def main() -> None:
    print("Loading accidents 2019-2025 (cached from previous run)...")
    df = load_all_accidents()
    print(f"  {len(df):,} rows, {df['num_expediente'].nunique():,} unique accidents.")

    print("Diagnosing...")
    year_summary = diag_year_summary(df)
    year_consistency = diag_year_consistency(df)
    num_exp = diag_num_expediente(df)
    lesiv = diag_lesividad(df)
    people = diag_people_per_accident(df)
    sev = derive_severity(df)
    sev_stats = diag_severity(sev)
    no_inj = diag_no_injury_breakdown(df)
    coords = diag_coordinates(df)
    nulls = diag_nulls(df)
    cats = diag_categorical(df)

    report = {
        "year_summary": year_summary.to_dict(orient="records"),
        "year_consistency": year_consistency,
        "num_expediente": num_exp,
        "lesividad": lesiv,
        "people_per_accident": people,
        "severity": sev_stats,
        "no_injury_breakdown": no_inj,
        "coordinates": coords,
        "nulls": nulls,
        "categoricals": cats,
    }
    save_json(report, OUT / "05_accidents_diag.json")

    sections = {
        "Year summary": year_summary.to_string(index=False),
        "Year consistency": dumps_jsonable(year_consistency, indent=2),
        "num_expediente": dumps_jsonable(num_exp, indent=2),
        "Severity counts": dumps_jsonable(sev_stats["counts_by_year"], indent=2),
        "Severity shares": dumps_jsonable(sev_stats["share_by_year"], indent=2),
        "No-injury breakdown": dumps_jsonable(no_inj, indent=2),
        "People per accident": dumps_jsonable(people, indent=2),
        "Coordinates": dumps_jsonable(coords, indent=2),
    }
    save_markdown_report(sections, Path("reports/05_accidents_diag.md"))
    print("Done. See reports/05_accidents_diag.md and reports/tables/05_accidents_diag.json")


if __name__ == "__main__":
    main()