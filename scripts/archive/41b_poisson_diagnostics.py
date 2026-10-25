"""
Script 41b — Phase 10 iter 1b: diagnostics and model comparison.

Fixes NB2 (sm.NegativeBinomial with start_params + Nelder-Mead fallback),
adds three models for comparison, bootstrap CI, and residual hotspot analysis.

Models:
    M0: n ~ 1                          (null, no offset)
    M1: n ~ 1 + offset(log_exposure)   (exposure-only, done in 41)
    M2: n ~ 1 + log_exposure           (exposure free coef, tests α=1)
    M3: n ~ 1 + log_exposure + kde + highway_ordinal  (adds covariates)
    M4: n ~ 1 + kde + highway_ordinal  (covariates only, no exposure)

Diagnostics:
    - Cameron-Trivedi overdispersion test.
    - Bootstrap 95% CI on α (500 resamples).
    - Top-10 positive/negative residuals (residual hotspots).
    - Model comparison via ΔAIC and Moran on residuals.

Outputs:
    reports/tables/41b_diagnostics.json
    reports/41b_diagnostics.md
    data/processed/residual_hotspots.parquet
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
from scipy import stats

import libpysal
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/41b_diagnostics.md")
OUT_HOTSPOTS = DATA / "residual_hotspots.parquet"

log = logging.getLogger("phase10_41b")


# ---------------------------------------------------------------------------
# Load panel + add covariates
# ---------------------------------------------------------------------------

def load_panel_with_covariates() -> pd.DataFrame:
    log.info("Loading panel ...")
    panel = pd.read_parquet(DATA / "panel_hex_month_n_injured.parquet")
    log.info(f"  {len(panel):,} rows, {panel['hex_id'].nunique()} hexes")

    # Add KDE density (from Fase 8)
    moran_hex = pd.read_parquet(DATA / "moran_hex250m.parquet")
    kde_cols = [c for c in moran_hex.columns if "kde_all_mean" in c]
    # Fase 8 didn't persist kde_all_mean in the parquet; recompute from
    # Fase 7 network_kde_final. We join per-hex mean of kde_all on train.
    log.info("Loading KDE (train mean per hex) ...")
    kde_final = pd.read_parquet(DATA / "network_kde_final.parquet")
    net = ms_io.load_geometry_parquet(DATA / "network_osm.parquet",
                                       expected_crs="EPSG:4326").to_crs(25830)
    net["edge_id"] = np.arange(len(net), dtype=np.int64)
    net["kde_all"] = kde_final["kde_train_all_bw260"].to_numpy()
    net["len_m"] = net.geometry.length

    hexes = ms_io.load_geometry_parquet(DATA / "moran_hex250m.parquet",
                                         expected_crs="EPSG:25830")
    net_in_hex = gpd.sjoin(net[["geometry", "kde_all", "len_m"]],
                            hexes[["hex_id", "geometry"]],
                            how="inner", predicate="intersects")
    # Length-weighted mean kde per hex
    def wmean(g):
        w = g["len_m"].to_numpy()
        v = g["kde_all"].to_numpy()
        ok = np.isfinite(v) & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    kde_per_hex = net_in_hex.groupby("hex_id").apply(wmean).rename("kde_all_mean")
    kde_per_hex = kde_per_hex.reset_index()

    # Add highway ordinal mean (from Phase 4 network)
    log.info("Loading highway_ordinal ...")
    # For each hex, mean of highway_ordinal over edges that pass through it
    acc_feat = pd.read_parquet(
        DATA / "accidents_with_network_features.parquet",
        columns=["coordenada_x_utm", "coordenada_y_utm",
                 "edge_highway_ordinal"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"])
    pts = gpd.GeoDataFrame(
        acc_feat,
        geometry=gpd.points_from_xy(acc_feat["coordenada_x_utm"],
                                     acc_feat["coordenada_y_utm"]),
        crs="EPSG:25830",
    )
    joined = gpd.sjoin(pts, hexes[["hex_id", "geometry"]],
                       how="inner", predicate="within")
    hi_per_hex = joined.groupby("hex_id")["edge_highway_ordinal"].mean().rename(
        "highway_ordinal_mean").reset_index()

    panel = panel.merge(kde_per_hex, on="hex_id", how="left")
    panel = panel.merge(hi_per_hex, on="hex_id", how="left")

    # Drop NaN in covariates (a few hexes)
    n0 = len(panel)
    panel = panel.dropna(subset=["kde_all_mean", "highway_ordinal_mean"]).copy()
    log.info(f"  {len(panel):,} rows after covariate join "
             f"(dropped {n0 - len(panel):,})")

    return panel


# ---------------------------------------------------------------------------
# Cameron-Trivedi overdispersion test
# ---------------------------------------------------------------------------

def cameron_trivedi_test(y: np.ndarray, mu: np.ndarray) -> dict:
    """Regress (y-mu)^2 - y on mu (without intercept). Slope > 0 implies
    overdispersion. Uses OLS with alpha factor."""
    z = ((y - mu) ** 2 - y) / mu  # standard formulation
    # Simple regression: z ~ mu (no intercept)
    X = sm.add_constant(mu)
    model = sm.OLS(z, X).fit()
    alpha_hat = float(model.params.iloc[1])
    se = float(model.bse.iloc[1])
    z_stat = alpha_hat / se if se > 0 else np.nan
    return {
        "alpha": alpha_hat,
        "se": se,
        "z_stat": float(z_stat) if np.isfinite(z_stat) else None,
        "p_value": float(2 * (1 - stats.norm.cdf(abs(z_stat))))
                    if np.isfinite(z_stat) else None,
        "interpretation": "alpha > 0 → overdispersed; use NB",
    }


# ---------------------------------------------------------------------------
# Robust NB2 with fallbacks
# ---------------------------------------------------------------------------

def fit_nb2_robust(y: np.ndarray, exposure: np.ndarray,
                   max_attempts: int = 3) -> dict:
    """Multiple strategies to fit NB2. Returns diagnostics."""
    log_exposure = np.log(exposure + 1e-10)
    X = sm.add_constant(log_exposure)
    attempts = []

    # Attempt 1: sm.NegativeBinomial with default Newton
    try:
        model = sm.NegativeBinomial(y, X, loglike_method="nb2")
        res = model.fit(disp=0, method="newton", maxiter=200)
        attempts.append({"method": "newton", "ok": True,
                         "aic": float(res.aic)})
        log.info(f"  NB2 (newton): AIC={res.aic:.1f}")
        return {"success": True, "result": res, "attempts": attempts,
                "coef_log_exposure": float(res.params.iloc[1]),
                "alpha_param": float(res.params.iloc[2])}
    except Exception as e:  # noqa: BLE001
        attempts.append({"method": "newton", "ok": False,
                         "error": str(e)[:120]})
        log.info(f"  NB2 (newton) failed: {type(e).__name__}")

    # Attempt 2: Nelder-Mead
    try:
        model = sm.NegativeBinomial(y, X, loglike_method="nb2")
        res = model.fit(disp=0, method="nm", maxiter=5000,
                        start_params=[0.0, 1.0, 0.5])
        attempts.append({"method": "nm", "ok": True, "aic": float(res.aic)})
        log.info(f"  NB2 (nm): AIC={res.aic:.1f}")
        return {"success": True, "result": res, "attempts": attempts,
                "coef_log_exposure": float(res.params.iloc[1]),
                "alpha_param": float(res.params.iloc[2])}
    except Exception as e:  # noqa: BLE001
        attempts.append({"method": "nm", "ok": False,
                         "error": str(e)[:120]})
        log.info(f"  NB2 (nm) failed: {type(e).__name__}")

    # Attempt 3: GLM with NB family and fixed alpha=1.0
    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0))
        res = model.fit()
        attempts.append({"method": "glm_nb_fixed_alpha", "ok": True,
                         "aic": float(res.aic)})
        log.info(f"  GLM-NB (alpha=1): AIC={res.aic:.1f}")
        return {"success": True, "result": res, "attempts": attempts,
                "coef_log_exposure": float(res.params.iloc[1]),
                "alpha_param": 1.0, "alpha_fixed": True}
    except Exception as e:  # noqa: BLE001
        attempts.append({"method": "glm_nb_fixed_alpha", "ok": False,
                         "error": str(e)[:120]})
        log.info(f"  GLM-NB failed: {type(e).__name__}")

    return {"success": False, "attempts": attempts}


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def fit_all_models(panel: pd.DataFrame, y_col: str) -> dict:
    """Fit M0-M4 with Poisson and NB where useful."""
    log.info(f"\n=== Fitting models for {y_col} ===")
    results = {}

    y = panel[y_col].to_numpy(dtype=np.float64)
    exposure = panel["exposure_month"].to_numpy(dtype=np.float64)
    log_exposure = np.log(exposure + 1e-10)

    # M0: null, no offset, Poisson
    m0_pois = smf.glm(f"{y_col} ~ 1", data=panel,
                       family=sm.families.Poisson()).fit()
    results["M0_poisson_null"] = {
        "formula": "n ~ 1",
        "aic": float(m0_pois.aic),
        "overdispersion": float(m0_pois.pearson_chi2 / m0_pois.df_resid),
        "params": {k: float(v) for k, v in m0_pois.params.items()},
    }
    log.info(f"  M0 (Poisson, no offset): AIC={m0_pois.aic:.1f}")

    # M1: exposure offset (done in 41, redo here for consistency)
    m1_pois = smf.glm(f"{y_col} ~ 1", data=panel,
                       family=sm.families.Poisson(),
                       offset=log_exposure).fit()
    results["M1_poisson_offset"] = {
        "formula": "n ~ 1 + offset(log_exposure)",
        "aic": float(m1_pois.aic),
        "overdispersion": float(m1_pois.pearson_chi2 / m1_pois.df_resid),
        "params": {k: float(v) for k, v in m1_pois.params.items()},
    }
    log.info(f"  M1 (Poisson, offset): AIC={m1_pois.aic:.1f}")

    # M2: exposure as free covariate (tests whether coef ≈ 1)
    m2_pois = smf.glm(f"{y_col} ~ log_exposure", data=panel,
                       family=sm.families.Poisson()).fit()
    coef = float(m2_pois.params["log_exposure"])
    se = float(m2_pois.bse["log_exposure"])
    z = coef / se if se > 0 else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else None
    results["M2_poisson_free_exposure"] = {
        "formula": "n ~ log_exposure",
        "aic": float(m2_pois.aic),
        "coef_log_exposure": coef,
        "se": se,
        "z_stat": float(z) if np.isfinite(z) else None,
        "p_value": p,
        "test_coef_equals_1": {
            "z_stat": float((coef - 1) / se) if se > 0 else None,
            "p_value": (float(2 * (1 - stats.norm.cdf(abs((coef - 1) / se))))
                        if se > 0 else None),
        },
    }
    log.info(f"  M2 (free exposure): coef={coef:.4f} (α=1 test: p={results['M2_poisson_free_exposure']['test_coef_equals_1']['p_value']})")

    # M3: exposure + covariates (Poisson, offset)
    m3_formula = f"{y_col} ~ kde_all_mean + highway_ordinal_mean"
    m3_pois = smf.glm(m3_formula, data=panel,
                       family=sm.families.Poisson(),
                       offset=log_exposure).fit()
    results["M3_poisson_offset_plus_covariates"] = {
        "formula": m3_formula + " + offset(log_exposure)",
        "aic": float(m3_pois.aic),
        "params": {k: float(v) for k, v in m3_pois.params.items()},
        "pvalues": {k: float(v) for k, v in m3_pois.pvalues.items()},
    }
    log.info(f"  M3 (offset + covariates): AIC={m3_pois.aic:.1f}")

    # M4: covariates only (no exposure offset)
    m4_formula = f"{y_col} ~ kde_all_mean + highway_ordinal_mean"
    m4_pois = smf.glm(m4_formula, data=panel,
                       family=sm.families.Poisson()).fit()
    results["M4_poisson_covariates_only"] = {
        "formula": m4_formula,
        "aic": float(m4_pois.aic),
        "params": {k: float(v) for k, v in m4_pois.params.items()},
    }
    log.info(f"  M4 (covariates only): AIC={m4_pois.aic:.1f}")

    # Cameron-Trivedi overdispersion test on M1
    mu_m1 = m1_pois.fittedvalues
    ct = cameron_trivedi_test(y, mu_m1)
    results["cameron_trivedi_on_M1"] = ct
    log.info(f"  Cameron-Trivedi: alpha={ct['alpha']:.4f}, "
             f"p={ct['p_value']}")

    # NB2 robust fitting
    log.info("  Attempting NB2 (robust) ...")
    nb2 = fit_nb2_robust(y, exposure)
    if nb2["success"]:
        results["NB2_robust"] = {
            "aic": float(nb2["result"].aic),
            "coef_log_exposure": nb2["coef_log_exposure"],
            "alpha_param": nb2["alpha_param"],
            "attempts": nb2["attempts"],
        }
        results["best_model"] = "nb2"
    else:
        results["NB2_robust"] = {"success": False,
                                  "attempts": nb2["attempts"]}
        results["best_model"] = "poisson"

    # Moran on residuals of M1 (Poisson with offset)
    panel_copy = panel.copy()
    panel_copy["pearson_resid"] = (
        (y - mu_m1) / np.sqrt(mu_m1 + 1e-10))
    residuals_per_hex = panel_copy.groupby("hex_id").agg(
        mean_resid=("pearson_resid", "mean")).reset_index()

    hexes_geo = ms_io.load_geometry_parquet(
        DATA / "moran_hex250m.parquet", expected_crs="EPSG:25830")
    merged = hexes_geo.merge(residuals_per_hex, on="hex_id", how="inner")
    merged = merged[merged["mean_resid"].notna()].reset_index(drop=True)

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
        moran = Moran(merged["mean_resid"].to_numpy(), W, permutations=999)
        results["moran_on_M1_residuals"] = {
            "I": float(moran.I),
            "p_sim": float(moran.p_sim),
            "n_units": int(len(merged)),
        }
        log.info(f"  Moran on M1 residuals: I={moran.I:.4f} (p={moran.p_sim:.4f})")
    except Exception as e:  # noqa: BLE001
        log.warning(f"  Moran failed: {e}")
        results["moran_on_M1_residuals"] = {"error": str(e)}

    # ΔAIC comparisons
    results["delta_aic"] = {
        "M0_minus_M1": results["M0_poisson_null"]["aic"] -
                        results["M1_poisson_offset"]["aic"],
        "M1_minus_M2": results["M1_poisson_offset"]["aic"] -
                        results["M2_poisson_free_exposure"]["aic"],
        "M1_minus_M3": results["M1_poisson_offset"]["aic"] -
                        results["M3_poisson_offset_plus_covariates"]["aic"],
        "M1_minus_M4": results["M1_poisson_offset"]["aic"] -
                        results["M4_poisson_covariates_only"]["aic"],
    }

    # Residual hotspots (top +5 / -5 per mean residual)
    top_pos = merged.nlargest(5, "mean_resid")[
        ["hex_id", "mean_resid"]].to_dict(orient="records")
    top_neg = merged.nsmallest(5, "mean_resid")[
        ["hex_id", "mean_resid"]].to_dict(orient="records")
    results["residual_hotspots"] = {
        "top_5_positive": top_pos,
        "top_5_negative": top_neg,
    }

    return results


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_alpha(panel: pd.DataFrame, y_col: str,
                    n_boot: int = 500) -> dict:
    """Bootstrap CI on alpha (Poisson with offset)."""
    log.info(f"Bootstrap ({n_boot} resamples) ...")
    rng = np.random.default_rng(42)
    alphas = []
    n = len(panel)
    log_exp = panel["log_exposure"].to_numpy()
    y_full = panel[y_col].to_numpy()

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        panel_boot = panel.iloc[idx].copy()
        try:
            m = smf.glm(f"{y_col} ~ 1", data=panel_boot,
                         family=sm.families.Poisson(),
                         offset=panel_boot["log_exposure"]).fit()
            alphas.append(float(m.params.iloc[0]))
        except Exception:
            continue

    alphas = np.array(alphas)
    return {
        "n_successful": len(alphas),
        "mean": float(alphas.mean()),
        "ci_2.5": float(np.percentile(alphas, 2.5)),
        "ci_97.5": float(np.percentile(alphas, 97.5)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_md(results: dict) -> str:
    lines = ["# Phase 10 · Iter 1b — Diagnostics and model comparison\n"]

    for y_col in ("n_injured", "n_all"):
        r = results[y_col]
        lines.append(f"\n## Response: `{y_col}`\n")

        lines.append("\n### Model comparison (Poisson AIC)\n")
        lines.append("| model | AIC | ΔAIC vs M1 |\n")
        lines.append("| --- | ---: | ---: |\n")
        m1_aic = r["M1_poisson_offset"]["aic"]
        for name in ("M0_poisson_null", "M1_poisson_offset",
                     "M2_poisson_free_exposure",
                     "M3_poisson_offset_plus_covariates",
                     "M4_poisson_covariates_only"):
            aic = r[name]["aic"]
            lines.append(f"| `{name}` | {aic:.1f} | {m1_aic - aic:+.1f} |\n")

        lines.append("\n### Is exposure proportional? (M2 test α=1)\n")
        m2 = r["M2_poisson_free_exposure"]
        lines.append(f"- coefficient on `log_exposure`: "
                     f"**{m2['coef_log_exposure']:.4f}** "
                     f"(SE={m2['se']:.4f})\n")
        t = m2["test_coef_equals_1"]
        lines.append(f"- Test H0: coef=1 → z={t['z_stat']:.2f}, "
                     f"p={t['p_value']}\n")
        if t["p_value"] and t["p_value"] < 0.05:
            lines.append(
                "  - **Reject proportionality**. Accidents scale with "
                "exposure sublinearly or superlinearly. Offset not "
                "appropriate; keep exposure as free covariate.\n")
        else:
            lines.append(
                "  - **Cannot reject proportionality**. Offset is "
                "statistically justified.\n")

        lines.append("\n### NB2 robust fitting\n")
        nb = r["NB2_robust"]
        if "aic" in nb:
            lines.append(f"- Success: AIC={nb['aic']:.1f}, "
                         f"α_param={nb['alpha_param']:.4f}\n")
        else:
            lines.append(f"- Failed all attempts: {nb['attempts']}\n")

        lines.append("\n### Overdispersion\n")
        ct = r["cameron_trivedi_on_M1"]
        lines.append(f"- Cameron-Trivedi alpha={ct['alpha']:.4f}, "
                     f"p={ct['p_value']}\n")
        lines.append(f"- Pearson chi2/df = "
                     f"{r['M1_poisson_offset']['overdispersion']:.2f}\n")

        lines.append("\n### Moran on M1 residuals\n")
        mor = r.get("moran_on_M1_residuals", {})
        if "I" in mor:
            lines.append(f"- I={mor['I']:.4f} (p={mor['p_sim']:.4f}, "
                         f"n={mor['n_units']})\n")
            lines.append(f"- ΔI from Fase 8 (0.677): "
                         f"{0.677 - mor['I']:+.3f}\n")

        lines.append("\n### Residual hotspots (mean per hex over 84 months)\n")
        lines.append("**Top 5 positive (más accidentes de lo esperado):**\n")
        for h in r["residual_hotspots"]["top_5_positive"]:
            lines.append(f"- hex {h['hex_id']}: resid={h['mean_resid']:.3f}\n")
        lines.append("\n**Top 5 negative (menos accidentes de lo esperado):**\n")
        for h in r["residual_hotspots"]["top_5_negative"]:
            lines.append(f"- hex {h['hex_id']}: resid={h['mean_resid']:.3f}\n")

        lines.append("\n### Bootstrap CI on α\n")
        b = r["bootstrap_alpha"]
        lines.append(f"- mean α={b['mean']:.4f}, "
                     f"95% CI [{b['ci_2.5']:.4f}, {b['ci_97.5']:.4f}], "
                     f"n={b['n_successful']}\n")

    lines.append("\n## Conclusions\n")
    lines.append(results["conclusions"])
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
    panel = load_panel_with_covariates()

    all_results = {}
    for y_col in ("n_injured", "n_all"):
        r = fit_all_models(panel, y_col)
        r["bootstrap_alpha"] = bootstrap_alpha(panel, y_col, n_boot=500)
        all_results[y_col] = r

    # Conclusions
    conclusions = []
    for y_col in ("n_injured", "n_all"):
        r = all_results[y_col]
        d1 = r["delta_aic"]["M0_minus_M1"]
        d3 = r["delta_aic"]["M1_minus_M3"]
        d4 = r["delta_aic"]["M1_minus_M4"]
        conclusions.append(
            f"- **{y_col}**: exposure aporta ΔAIC M0→M1 = {d1:.0f} "
            f"(cuanto mayor, más útil exposure). Covariables añaden "
            f"ΔAIC M1→M3 = {d3:.0f} (si > 10, aportan). "
            f"Sin exposure, covariables dan ΔAIC M1→M4 = {d4:.0f} "
            f"(si < 0, covariables empeoran sin exposure).")

    all_results["conclusions"] = "\n".join(conclusions)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "41b_diagnostics.json").write_text(
        dumps_jsonable(all_results), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(all_results), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '41b_diagnostics.json'}")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    for y_col in ("n_injured", "n_all"):
        r = all_results[y_col]
        log.info(f"[{y_col}] best={r['best_model']} "
                 f"ΔAIC(M0→M1)={r['delta_aic']['M0_minus_M1']:.0f} "
                 f"ΔAIC(M1→M3)={r['delta_aic']['M1_minus_M3']:.0f} "
                 f"α={r['M1_poisson_offset']['params']}")


if __name__ == "__main__":
    main()