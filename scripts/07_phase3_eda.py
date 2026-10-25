"""
Phase 3 · Iteration 1 — EDA and problem design.

Focus: severity prior, COVID rupture, temporal patterns, categorical patterns,
        missingness of candidate features, and leakage proof.

Run from the repository root:
    python scripts/07_phase3_eda.py

Outputs:
    reports/tables/07_eda.json
    reports/07_eda.md
    reports/figures/phase3/01_*.png … 04_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

DATA = Path("data/processed/accidents_clean.parquet")
FIG = Path("reports/figures/phase3")
OUT = Path("reports/tables")
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["no_injury", "minor", "serious", "fatal"]
SEVERITY_COLORS = {"no_injury": "#9ecae1", "minor": "#fdd0a2",
                   "serious": "#fd8d3c", "fatal": "#a50f15"}


def _savefig(fig, name: str) -> str:
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# 1. Severity prior by year
# ---------------------------------------------------------------------------
def prior_by_year(df: pd.DataFrame) -> dict:
    counts = (df.groupby(["year", "severity"]).size()
                .unstack(fill_value=0)
                .reindex(columns=SEVERITY_ORDER, fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for sev in SEVERITY_ORDER:
        ax.plot(share.index, share[sev], marker="o", label=sev,
                color=SEVERITY_COLORS[sev])
    ax.set_title("Severity share by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of accidents")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_path = _savefig(fig, "01_severity_by_year")
    return {
        "counts": counts.to_dict(orient="index"),
        "shares": share.round(4).to_dict(orient="index"),
        "figure": fig_path,
    }


# ---------------------------------------------------------------------------
# 2. COVID rupture
# ---------------------------------------------------------------------------
def covid_rupture(df: pd.DataFrame) -> dict:
    groups = {
        "pre_2019": df[df["year"] == 2019],
        "covid_2020_21": df[df["year"].isin([2020, 2021])],
        "post_2022_25": df[df["year"].isin([2022, 2023, 2024, 2025])],
    }
    out = {}
    for name, sub in groups.items():
        n = len(sub)
        vc = sub["severity"].value_counts(normalize=True).round(4).to_dict()
        fatal_rate = float((sub["severity"] == "fatal").mean())
        serious_rate = float(sub["severity"].isin(["fatal", "serious"]).mean())
        out[name] = {
            "n_accidents": int(n),
            "severity_share": vc,
            "fatal_rate": round(fatal_rate, 5),
            "serious_or_fatal_rate": round(serious_rate, 5),
        }
    return out


# ---------------------------------------------------------------------------
# 3. Temporal patterns (hour, weekday, month)
# ---------------------------------------------------------------------------
def _share_plot(ax, share_df, title, xlabel):
    for sev in SEVERITY_ORDER:
        if sev in share_df.columns:
            ax.plot(share_df.index, share_df[sev], marker="o", label=sev,
                    color=SEVERITY_COLORS[sev])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share within period")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def temporal_patterns(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["month"] = d["fecha"].dt.month
    d["weekday"] = d["fecha"].dt.weekday  # 0=Mon
    d["hour_int"] = d["hour"].astype("Int64")
    d = d.dropna(subset=["hour_int"])
    d["hour_int"] = d["hour_int"].astype(int)

    def _agg(col):
        c = (d.groupby([col, "severity"]).size()
               .unstack(fill_value=0)
               .reindex(columns=SEVERITY_ORDER, fill_value=0))
        return c, c.div(c.sum(axis=1), axis=0)

    hour_c, hour_s = _agg("hour_int")
    wd_c, wd_s = _agg("weekday")
    month_c, month_s = _agg("month")

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    _share_plot(axes[0], hour_s, "Severity share by hour of day", "Hour")
    axes[0].set_xticks(range(0, 24, 2))
    _share_plot(axes[1], wd_s, "Severity share by weekday (0=Mon)", "Weekday")
    axes[1].set_xticks(range(0, 7))
    _share_plot(axes[2], month_s, "Severity share by month", "Month")
    axes[2].set_xticks(range(1, 13))
    fig.tight_layout()
    fig_path = _savefig(fig, "02_temporal_patterns")

    return {
        "hour": {"counts": hour_c.to_dict(orient="index"),
                 "shares": hour_s.round(4).to_dict(orient="index")},
        "weekday": {"counts": wd_c.to_dict(orient="index"),
                    "shares": wd_s.round(4).to_dict(orient="index")},
        "month": {"counts": month_c.to_dict(orient="index"),
                  "shares": month_s.round(4).to_dict(orient="index")},
        "figure": fig_path,
    }


# ---------------------------------------------------------------------------
# 4. Categorical features vs severity
# ---------------------------------------------------------------------------
def categorical_patterns(df: pd.DataFrame) -> dict:
    def _agg(col, top_n=10):
        c = (df.groupby([col, "severity"]).size()
               .unstack(fill_value=0)
               .reindex(columns=SEVERITY_ORDER, fill_value=0))
        c = c.sort_values(by=SEVERITY_ORDER, ascending=False).head(top_n)
        s = c.div(c.sum(axis=1), axis=0)
        return c, s

    acc_c, acc_s = _agg("tipo_accidente", top_n=10)
    met_c, met_s = _agg("estado_meteorologico", top_n=8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, s, title in [
        (axes[0], acc_s, "Severity share by tipo_accidente (top 10)"),
        (axes[1], met_s, "Severity share by estado_meteorologico"),
    ]:
        s[SEVERITY_ORDER].plot(kind="barh", stacked=True, ax=ax,
                                color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER])
        ax.set_title(title)
        ax.set_xlabel("Share within category")
        ax.set_xlim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig_path = _savefig(fig, "03_categorical_severity")

    return {
        "tipo_accidente": {
            "counts": acc_c.to_dict(orient="index"),
            "shares": acc_s.round(4).to_dict(orient="index"),
        },
        "estado_meteorologico": {
            "counts": met_c.to_dict(orient="index"),
            "shares": met_s.round(4).to_dict(orient="index"),
        },
        "figure": fig_path,
    }


# ---------------------------------------------------------------------------
# 5. District-level (rough, no maps yet)
# ---------------------------------------------------------------------------
def district_severity(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["distrito"])
    counts = (d.groupby(["distrito", "severity"]).size()
                .unstack(fill_value=0)
                .reindex(columns=SEVERITY_ORDER, fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0)
    # Injury rate (anything above no_injury)
    counts["injury_rate"] = (
        (counts["minor"] + counts["serious"] + counts["fatal"])
        / counts[SEVERITY_ORDER].sum(axis=1)
    )
    top = counts.sort_values("injury_rate", ascending=False).head(10)
    return {
        "per_district": counts.round(4).to_dict(orient="index"),
        "top10_by_injury_rate": top.round(4).to_dict(orient="index"),
    }


# ---------------------------------------------------------------------------
# 6. Feature availability and missingness (candidate features only)
# ---------------------------------------------------------------------------
def feature_candidates(df: pd.DataFrame) -> dict:
    # M2 features (proposed; see design section)
    candidate_features = [
        "year", "hour", "localizacion", "cod_distrito", "distrito",
        "coordenada_x_utm", "coordenada_y_utm",
        "tipo_accidente", "estado_meteorologico",
        "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
        "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
    ]
    report = {}
    for f in candidate_features:
        if f not in df.columns:
            continue
        s = df[f]
        report[f] = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "missing_share": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique()),
            "is_categorical": bool(s.dtype == "object" or s.dtype == "bool"),
        }
    return report


# ---------------------------------------------------------------------------
# 7. Leakage proof
# ---------------------------------------------------------------------------
def leakage_check(df: pd.DataFrame) -> dict:
    leaky = ["n_fatalities", "n_serious", "n_minor",
             "any_alcohol_positive", "n_alcohol_positive",
             "any_drug_positive",
             "cod_lesividad_max", "worst_lesividad_code"]
    out = {}
    for f in leaky:
        if f not in df.columns:
            continue
        try:
            s = df[[f, "severity_code"]].dropna()
            corr = s.corr(method="spearman").iloc[0, 1]
            out[f] = {"spearman_with_severity_code": round(float(corr), 3),
                      "n": int(len(s))}
        except Exception as exc:
            out[f] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {DATA}...")
    df = pd.read_parquet(DATA)
    print(f"  {len(df):,} accidents.")

    print("1/7 Severity prior by year...")
    prior = prior_by_year(df)

    print("2/7 COVID rupture...")
    covid = covid_rupture(df)

    print("3/7 Temporal patterns...")
    temporal = temporal_patterns(df)

    print("4/7 Categorical patterns...")
    cats = categorical_patterns(df)

    print("5/7 District severity...")
    districts = district_severity(df)

    print("6/7 Feature candidates...")
    features = feature_candidates(df)

    print("7/7 Leakage proof...")
    leakage = leakage_check(df)

    report = {
        "prior_by_year": prior,
        "covid_rupture": covid,
        "temporal": temporal,
        "categoricals": cats,
        "districts": districts,
        "feature_candidates": features,
        "leakage": leakage,
    }
    save_json(report, OUT / "07_eda.json")

    sections = {
        "Prior by year": dumps_jsonable(prior["shares"], indent=2),
        "COVID rupture": dumps_jsonable(covid, indent=2),
        "Hour patterns": dumps_jsonable(temporal["hour"]["shares"], indent=2),
        "Weekday patterns": dumps_jsonable(temporal["weekday"]["shares"], indent=2),
        "Month patterns": dumps_jsonable(temporal["month"]["shares"], indent=2),
        "Tipo_accidente": dumps_jsonable(cats["tipo_accidente"]["shares"], indent=2),
        "Estado_meteorologico": dumps_jsonable(
            cats["estado_meteorologico"]["shares"], indent=2),
        "Top10 districts by injury rate": dumps_jsonable(
            districts["top10_by_injury_rate"], indent=2),
        "Feature candidates": dumps_jsonable(features, indent=2),
        "Leakage proof": dumps_jsonable(leakage, indent=2),
    }
    save_markdown_report(sections, Path("reports/07_eda.md"))
    print("Done. See reports/07_eda.md and reports/figures/phase3/.")


if __name__ == "__main__":
    main()