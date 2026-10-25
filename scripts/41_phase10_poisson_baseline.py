"""
Script 41 — Phase 10 iter 1: Poisson/NB baseline with exposure offset.

Model:
    log(E[accidents_hex_month]) = α + log(exposure_hex_month)

Where exposure_hex_month = intensity_day_mean_mean × days_in_month
(vehículos totales que circulan por el hex en el mes).

For both responses:
    - n_injured (severity_code >= 1)  -- primary
    - n_all     (all accidents)       -- secondary

Diagnostics:
    - Overdispersion: Pearson chi2 / df.
    - If > 2.0 → refit with Negative Binomial (NB2).
    - Moran on Pearson residuals to check spatial autocorrelation
      of the residuals. If I_residuals << I_counts, exposure
      explains most of the spatial structure.

Output:
    reports/tables/41_poisson_baseline.json
    reports/41_poisson_baseline.md
    data/processed/panel_hex_month.parquet (the panel, for iter 2)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

import libpysal
import esda
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/41_poisson_baseline.md")
OUT_PANEL = DATA / "panel_hex_month.parquet"

log = logging.getLogger("phase10_41")


# ---------------------------------------------------------------------------
# Build panel
# ---------------------------------------------------------------------------

def build_panel() -> pd.DataFrame:
    """Join accidents (aggregated hex×month) with exposure."""

    # 1. Load accidents
    log.info("Loading accidents ...")
    acc = pd.read_parquet(
        DATA / "accidents_clean.parquet",
        columns=["num_expediente", "year", "fecha",
                 "severity_code",
                 "coordenada_x_utm", "coordenada_y_utm"],
    )
    acc["year"] = acc["year"].astype(int)
    acc["month"] = pd.to_datetime(acc["fecha"], errors="coerce").dt.month
    acc = acc.dropna(subset=["month", "coordenada_x_utm",
                             "coordenada_y_utm"]).copy()
    acc["month"] = acc["month"].astype(int)
    log.info(f"  {len(acc):,} accidents with valid date+coords")

    # 2. Assign to hex (sjoin within)
    log.info("Loading hex grid ...")
    hexes = ms_io.load_geometry_parquet(
        DATA / "moran_hex250m.parquet",
        expected_crs="EPSG:25830",
    )
    hexes = hexes[["hex_id", "geometry"]].copy()
    log.info(f"  {len(hexes):,} hexes")

    log.info("Assigning accidents to hexes ...")
    pts = gpd.GeoDataFrame(
        acc,
        geometry=gpd.points_from_xy(acc["coordenada_x_utm"],
                                     acc["coordenada_y_utm"]),
        crs="EPSG:25830",
    )
    joined = gpd.sjoin(pts, hexes, how="inner", predicate="within")
    log.info(f"  {len(joined):,} accidents in a hex")

    # 3. Aggregate to hex × year × month
    log.info("Aggregating accidents to hex × year × month ...")
    joined["n_all"] = 1
    joined["n_injured"] = (joined["severity_code"] >= 1).astype(int)
    panel_acc = joined.groupby(
        ["hex_id", "year", "month"], as_index=False
    ).agg(n_all=("n_all", "sum"),
          n_injured=("n_injured", "sum"))
    log.info(f"  {len(panel_acc):,} hex-month rows with accidents")

    # 4. Load exposure
    log.info("Loading exposure ...")
    exp = pd.read_parquet(DATA / "exposure_hex250m_monthly.parquet")
    exp = exp[[
        "hex_id", "year", "month",
        "intensity_day_mean_mean", "intensity_day_median",
        "intensity_day_sum", "n_sensors", "coverage_mean",
        "n_urb", "n_m30_c30",
    ]].copy()
    log.info(f"  {len(exp):,} hex-month rows with exposure")

    # 5. Merge: inner keeps only hex-months with both accidents + exposure
    #    But we also need hex-months with exposure and 0 accidents!
    #    So: start from exp, left-join panel_acc, fill 0.
    log.info("Building final panel (exposure-driven, accidents left-joined) ...")
    panel = exp.merge(
        panel_acc,
        on=["hex_id", "year", "month"],
        how="left",
    )
    panel["n_all"] = panel["n_all"].fillna(0).astype(int)
    panel["n_injured"] = panel["n_injured"].fillna(0).astype(int)

    # Days in month → exposure per month (veh × days)
    import calendar
    panel["days_in_month"] = [
        calendar.monthrange(int(y), int(m))[1]
        for y, m in zip(panel["year"], panel["month"])
    ]
    panel["exposure_month"] = (
        panel["intensity_day_mean_mean"] * panel["days_in_month"]
    )

    # Drop rows with invalid exposure
    n_before = len(panel)
    panel = panel[panel["exposure_month"] > 0].copy()
    log.info(f"  kept {len(panel):,}/{n_before:,} with valid exposure")

    # Log transform for offset
    panel["log_exposure"] = np.log(panel["exposure_month"])

    # Add year-month string for potential fixed effects
    panel["ym"] = (panel["year"].astype(str) + "-" +
                    panel["month"].astype(str).str.zfill(2))

    return panel


# ---------------------------------------------------------------------------
# Fit Poisson + NB
# ---------------------------------------------------------------------------

def fit_models(panel: pd.DataFrame, y_col: str) -> dict:
    log.info(f"=== Fitting models for {y_col} ===")

    # Poisson with offset
    formula = f"{y_col} ~ 1"
    model_pois = smf.glm(
        formula=formula,
        data=panel,
        family=sm.families.Poisson(),
        offset=panel["log_exposure"],
    ).fit()
    log.info(f"  Poisson AIC={model_pois.aic:.1f}  "
             f"alpha={model_pois.params.iloc[0]:.6f}")

    # Overdispersion
    pearson_chi2 = float(model_pois.pearson_chi2)
    df_resid = float(model_pois.df_resid)
    overdisp = pearson_chi2 / df_resid
    log.info(f"  overdispersion (Pearson chi2/df) = {overdisp:.2f}")

    # NB if overdispersed
    model_nb = None
    if overdisp > 2.0:
        log.info("  overdispersion > 2, fitting NB2 ...")
        try:
            model_nb = smf.negativebinomial(
                formula=formula,
                data=panel,
                offset=panel["log_exposure"],
            ).fit(disp=0)
            log.info(f"  NB AIC={model_nb.aic:.1f}  "
                     f"alpha_param={model_nb.params.iloc[1]:.4f}")
        except Exception as e:  # noqa: BLE001
            log.warning(f"  NB failed: {type(e).__name__}: {e}")

    # Choose best
    best_model = model_pois
    best_name = "poisson"
    if model_nb is not None and model_nb.aic < model_pois.aic:
        best_model = model_nb
        best_name = "neg_binomial"

    # Pearson residuals from best model
    mu = best_model.fittedvalues
    y = panel[y_col].to_numpy()
    residuals = (y - mu) / np.sqrt(mu + 1e-10)
    panel_copy = panel.copy()
    panel_copy["pearson_resid"] = residuals
    panel_copy["fitted"] = mu

    # Moran on Pearson residuals (using same W as Phase 8)
    log.info("  Computing Moran on Pearson residuals ...")
    hexes_resid = panel_copy.groupby("hex_id").agg(
        mean_resid=("pearson_resid", "mean"),
    ).reset_index()
    hexes_geo = ms_io.load_geometry_parquet(
        DATA / "moran_hex250m.parquet",
        expected_crs="EPSG:25830",
    )
    merged = hexes_geo.merge(hexes_resid, on="hex_id", how="inner")
    merged = merged[merged["mean_resid"].notna()].reset_index(drop=True)

    moran_resid = None
    try:
        W = libpysal.weights.Queen.from_dataframe(
            merged, use_index=False, silence_warnings=True)
        if len(W.islands) > 0:
            keep = np.array([i for i in range(len(merged))
                             if i not in W.islands])
            merged = merged.iloc[keep].reset_index(drop=True)
            W = libpysal.weights.Queen.from_dataframe(
                merged, use_index=False, silence_warnings=True)
        W.transform = "r"
        moran = Moran(merged["mean_resid"].to_numpy(), W,
                       permutations=999)
        moran_resid = {
            "I": float(moran.I),
            "p_sim": float(moran.p_sim),
            "n_units": int(len(merged)),
        }
        log.info(f"  Moran on residuals: I={moran.I:.4f} (p={moran.p_sim:.4f})")
    except Exception as e:  # noqa: BLE001
        log.warning(f"  Moran on residuals failed: {e}")

    # Aggregate fit by year
    panel_copy["year_int"] = panel_copy["year"].astype(int)
    by_year = []
    for y in sorted(panel_copy["year_int"].unique()):
        s = panel_copy[panel_copy["year_int"] == y]
        by_year.append({
            "year": int(y),
            "actual_total": int(s[y_col].sum()),
            "predicted_total": float(s["fitted"].sum()),
            "ratio": float(s["fitted"].sum() / max(s[y_col].sum(), 1)),
            "n_hex_month": int(len(s)),
        })

    # Save panel for iter 2
    panel_copy["model_used"] = best_name
    panel_copy["y_col"] = y_col
    if y_col == "n_injured":  # save only once
        panel_copy.to_parquet(
            DATA / f"panel_hex_month_{y_col}.parquet", index=False)

    return {
        "response": y_col,
        "n_obs": int(len(panel)),
        "n_hexes": int(panel["hex_id"].nunique()),
        "poisson": {
            "aic": float(model_pois.aic),
            "alpha": float(model_pois.params.iloc[0]),
            "overdispersion": float(overdisp),
        },
        "nb": {
            "aic": float(model_nb.aic) if model_nb is not None else None,
            "alpha_param": (float(model_nb.params.iloc[1])
                            if model_nb is not None else None),
        } if model_nb is not None else None,
        "best_model": best_name,
        "moran_on_residuals": moran_resid,
        "by_year": by_year,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_md(results: dict) -> str:
    lines = ["# Phase 10 · Iter 1 — Poisson baseline with exposure offset\n"]
    lines.append(
        "**Model:** `E[accidents] = exp(α) × exposure`. Null model "
        "with offset = log(exposure_month). Fit via Poisson GLM; NB2 if "
        "overdispersed.\n\n"
    )

    for resp in ("n_injured", "n_all"):
        r = results[resp]
        lines.append(f"## Response: `{resp}`\n")
        lines.append(f"- n_obs = {r['n_obs']:,}, "
                     f"n_hexes = {r['n_hexes']:,}\n")
        p = r["poisson"]
        lines.append(f"- **Poisson**: AIC={p['aic']:.1f}, "
                     f"α={p['alpha']:.2e}, "
                     f"overdispersion={p['overdispersion']:.2f}\n")
        if r["nb"]:
            lines.append(
                f"- **NB2**: AIC={r['nb']['aic']:.1f}, "
                f"alpha_param={r['nb']['alpha_param']:.3f}\n")
        lines.append(f"- **Best**: `{r['best_model']}`\n")

        if r["moran_on_residuals"]:
            m = r["moran_on_residuals"]
            lines.append(f"- **Moran on residuals**: I={m['I']:.4f} "
                         f"(p={m['p_sim']:.4f}, n={m['n_units']})\n")
            lines.append(
                "  - Fase 8 counts I=0.677. "
                f"Reduction: {0.677 - m['I']:.3f}. "
                "If large → exposure explains spatial pattern. "
                "If small → residual spatial structure remains.\n")

        lines.append("\n### By year fit\n")
        lines.append("| year | actual | predicted | ratio |\n")
        lines.append("| --- | ---: | ---: | ---: |\n")
        for yv in r["by_year"]:
            lines.append(f"| {yv['year']} | {yv['actual_total']:,} | "
                         f"{yv['predicted_total']:.0f} | "
                         f"{yv['ratio']:.3f} |\n")
        lines.append("\n")

    lines.append("## Interpretation\n")
    lines.append(results["interpretation"])
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build panel
    log.info("Building panel ...")
    panel = build_panel()
    log.info(f"Final panel: {len(panel):,} rows, "
             f"{panel['hex_id'].nunique():,} unique hexes, "
             f"{panel['year'].nunique()} unique years")
    log.info(f"  n_all total: {panel['n_all'].sum():,}")
    log.info(f"  n_injured total: {panel['n_injured'].sum():,}")

    # Fit for both responses
    results = {}
    results["n_injured"] = fit_models(panel, "n_injured")
    results["n_all"] = fit_models(panel, "n_all")

    # Interpretation
    interp_lines = []
    for resp in ("n_injured", "n_all"):
        r = results[resp]
        moran_res = r["moran_on_residuals"]
        if moran_res:
            reduction = 0.677 - moran_res["I"]
            if reduction > 0.30:
                interp_lines.append(
                    f"- **{resp}**: exposure explains most of the "
                    f"spatial autocorrelation (I: 0.677 → "
                    f"{moran_res['I']:.3f}, Δ={reduction:+.3f}). "
                    f"Los hotspots de Fase 8 son esencialmente "
                    f"consecuencia del volumen de tráfico.")
            elif reduction > 0.10:
                interp_lines.append(
                    f"- **{resp}**: exposure explains part of the "
                    f"spatial structure (I: 0.677 → {moran_res['I']:.3f}). "
                    f"Hay heterogeneidad espacial real más allá del "
                    f"tráfico → justifica covariables en iter 2.")
            else:
                interp_lines.append(
                    f"- **{resp}**: exposure does NOT explain the "
                    f"spatial structure (I: 0.677 → "
                    f"{moran_res['I']:.3f}). Los accidentes tienen "
                    f"patrones espaciales propios independientes del "
                    f"volumen de tráfico.")

        interp_lines.append(
            f"- **{resp}**: overdispersion={r['poisson']['overdispersion']:.2f}. "
            f"Best model: `{r['best_model']}` "
            f"(AIC Poisson={r['poisson']['aic']:.1f}"
            + (f", NB2={r['nb']['aic']:.1f}" if r['nb'] else "")
            + ").")

    results["interpretation"] = "\n".join(interp_lines)

    # Save
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "41_poisson_baseline.json").write_text(
        dumps_jsonable(results), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(results), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '41_poisson_baseline.json'}")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    for resp in ("n_injured", "n_all"):
        r = results[resp]
        log.info(f"[{resp}] best={r['best_model']} "
                 f"overdisp={r['poisson']['overdispersion']:.2f} "
                 f"moran_resid={r['moran_on_residuals']['I'] if r['moran_on_residuals'] else 'n/a'}")


if __name__ == "__main__":
    main()