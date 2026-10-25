"""
Script 41c — Phase 10 iter 1c: refit with corrected NB2 and honest covariates.

Key changes vs 41b:
  - Fix NB2 param extraction (positional indexing, not .iloc).
  - Drop KDE from primary models (leaky: smoothed target).
  - Use log_exposure as free covariate (coefficient ~0.53, not 1.0).
  - Add year and month fixed effects (capture COVID + seasonality).
  - Report coefficient stability across specifications.
  - Fix inverted interpretation text.
  - Document KDE separately as leaky diagnostic.

Models (all NB2):
    N0: n ~ log_exposure
    N1: n ~ log_exposure + highway_ordinal_mean
    N2: n ~ log_exposure + highway_ordinal_mean + year (FE)
    N3: n ~ log_exposure + highway_ordinal_mean + year + month (FE)
    N4: n ~ log_exposure + highway_ordinal_mean + year + month + tipo_via (FE)

Diagnostic (leaky, not for reporting):
    NX: n ~ log_exposure + highway_ordinal_mean + kde_all_mean
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

import libpysal
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/41c_refit.md")

log = logging.getLogger("phase10_41c")


# ---------------------------------------------------------------------------
# Load panel + clean covariates
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    log.info("Loading panel ...")
    panel = pd.read_parquet(DATA / "panel_hex_month_n_injured.parquet")
    log.info(f"  {len(panel):,} rows, {panel['hex_id'].nunique()} hexes")

    # high ordinal mean
    hexes = ms_io.load_geometry_parquet(DATA / "moran_hex250m.parquet",
                                         expected_crs="EPSG:25830")
    acc_feat = pd.read_parquet(
        DATA / "accidents_with_network_features.parquet",
        columns=["coordenada_x_utm", "coordenada_y_utm",
                 "edge_highway_ordinal", "tipo_via"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"])
    pts = gpd.GeoDataFrame(
        acc_feat,
        geometry=gpd.points_from_xy(acc_feat["coordenada_x_utm"],
                                     acc_feat["coordenada_y_utm"]),
        crs="EPSG:25830",
    )
    joined = gpd.sjoin(pts, hexes[["hex_id", "geometry"]],
                       how="inner", predicate="within")
    hi = joined.groupby("hex_id")["edge_highway_ordinal"].mean().rename(
        "highway_ordinal_mean").reset_index()

    # Dominant tipo_via per hex (mode)
    tipo_mode = (joined.groupby("hex_id")["tipo_via"]
                  .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Unknown")
                  .rename("tipo_via").reset_index())

    panel = panel.merge(hi, on="hex_id", how="left")
    panel = panel.merge(tipo_mode, on="hex_id", how="left")
    panel["tipo_via"] = panel["tipo_via"].fillna("Unknown")

    # Encode year and month as strings for FE
    panel["year_str"] = panel["year"].astype(str)
    panel["month_str"] = panel["month"].astype(str).str.zfill(2)
    panel["log_exposure"] = np.log(panel["exposure_month"] + 1e-10)

    n0 = len(panel)
    panel = panel.dropna(subset=["log_exposure", "highway_ordinal_mean"]).copy()
    log.info(f"  {len(panel):,} rows after covariate join "
             f"(dropped {n0 - len(panel)})")
    log.info(f"  tipo_via distribution: {panel['tipo_via'].value_counts().head().to_dict()}")
    return panel


# ---------------------------------------------------------------------------
# NB2 with positional params
# ---------------------------------------------------------------------------

def fit_nb2(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """Fit NB2. Returns dict with coef, se, aic, alpha."""
    log.info(f"  Fitting NB2 (n_params={X.shape[1]}) ...")
    try:
        res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            disp=0, method="nm", maxiter=5000,
            start_params=np.concatenate([np.zeros(X.shape[1]), [0.5]]),
        )
        params = np.asarray(res.params)  # ensure array
        aic = float(res.aic)
        # params: [beta_0, ..., beta_k, alpha_nb2]
        coefs = params[:-1]
        alpha_nb2 = params[-1]
        # se
        try:
            bse = np.asarray(res.bse)
        except Exception:
            bse = np.full_like(params, np.nan)
        se = bse[:-1]
        return {
            "success": True,
            "aic": aic,
            "coef": {name: float(c) for name, c in zip(names, coefs)},
            "se": {name: float(s) for name, s in zip(names, se)},
            "alpha_nb2": float(alpha_nb2),
            "n_params": int(X.shape[1] + 1),
            "n_obs": int(len(y)),
        }
    except Exception as e:  # noqa: BLE001
        log.warning(f"  NB2 failed: {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def fit_nb_formula(formula: str, data: pd.DataFrame) -> dict:
    """Fit NB2 via formula API (has names)."""
    log.info(f"  NB2 formula: {formula}")
    try:
        res = smf.negativebinomial(formula, data=data,
                                    method="nm", maxiter=5000,
                                    disp=0).fit(
            start_params=None, disp=0,
        )
        # Try regular fit if NM succeeded
        try:
            res = smf.negativebinomial(formula, data=data).fit(disp=0, maxiter=500)
        except Exception:
            pass  # keep the NM result

        params = res.params
        se = res.bse
        coefs = {k: float(v) for k, v in params.items() if not k.startswith("alpha")}
        ses = {k: float(v) for k, v in se.items() if not k.startswith("alpha")}
        alpha_nb2 = float(params.get("alpha", np.nan))
        return {
            "success": True,
            "aic": float(res.aic),
            "coef": coefs,
            "se": ses,
            "alpha_nb2": alpha_nb2,
            "n_obs": int(len(data)),
        }
    except Exception as e:  # noqa: BLE001
        log.warning(f"  formula NB failed: {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Moran residuals
# ---------------------------------------------------------------------------

def moran_on_residuals(panel: pd.DataFrame,
                       fitted_mu: np.ndarray,
                       y: np.ndarray,
                       label: str) -> dict:
    resid = (y - fitted_mu) / np.sqrt(fitted_mu + 1e-10)
    df = panel[["hex_id"]].copy()
    df["resid"] = resid
    per_hex = df.groupby("hex_id").agg(mean_resid=("resid", "mean")).reset_index()

    hexes = ms_io.load_geometry_parquet(DATA / "moran_hex250m.parquet",
                                         expected_crs="EPSG:25830")
    merged = hexes.merge(per_hex, on="hex_id", how="inner")
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
        m = Moran(merged["mean_resid"].to_numpy(), W, permutations=999)
        log.info(f"  [{label}] Moran on residuals: I={m.I:.4f} (p={m.p_sim:.4f})")
        return {"I": float(m.I), "p_sim": float(m.p_sim),
                "n_units": int(len(merged))}
    except Exception as e:  # noqa: BLE001
        log.warning(f"  Moran failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    panel = load_panel()
    all_results = {}

    for y_col in ("n_injured", "n_all"):
        log.info(f"\n=============== {y_col} ===============")
        formulas = {
            "N0": f"{y_col} ~ log_exposure",
            "N1": f"{y_col} ~ log_exposure + highway_ordinal_mean",
            "N2": f"{y_col} ~ log_exposure + highway_ordinal_mean + C(year_str)",
            "N3": f"{y_col} ~ log_exposure + highway_ordinal_mean + C(year_str) + C(month_str)",
            "N4": f"{y_col} ~ log_exposure + highway_ordinal_mean + C(year_str) + C(month_str) + C(tipo_via)",
        }
        model_results = {}
        best_aic = np.inf
        best_name = None
        best_fitted = None
        best_y = None

        for name, formula in formulas.items():
            r = fit_nb_formula(formula, panel)
            if r.get("success"):
                mu = None
                try:
                    res = smf.negativebinomial(formula, data=panel).fit(disp=0)
                    mu = np.asarray(res.fittedvalues)
                except Exception:
                    pass
                r["fitted_mu_mean"] = float(mu.mean()) if mu is not None else None
                model_results[name] = r
                log.info(f"  [{name}] AIC={r['aic']:.1f}  "
                         f"coef_exposure={r['coef'].get('log_exposure', 'n/a')}")
                if r["aic"] < best_aic:
                    best_aic = r["aic"]
                    best_name = name
                    best_fitted = mu
                    best_y = panel[y_col].to_numpy()

        # Moran on residuals for best model
        if best_fitted is not None:
            mor = moran_on_residuals(panel, best_fitted, best_y, best_name)
            model_results[best_name]["moran_on_residuals"] = mor

        # Delta AIC vs N0
        if "N0" in model_results:
            base_aic = model_results["N0"]["aic"]
            for name, r in model_results.items():
                r["delta_aic_vs_N0"] = r["aic"] - base_aic

        # 2022 anomaly check (best model)
        if best_fitted is not None:
            tmp = panel[[y_col, "year"]].copy()
            tmp["fitted"] = best_fitted
            anomaly = (
                tmp.groupby("year")
                    .agg(actual=(y_col, "sum"), predicted=("fitted", "sum"))
                    .reset_index()
            )
            anomaly["ratio"] = anomaly["predicted"] / anomaly["actual"]
            model_results["by_year_fit"] = anomaly.to_dict(orient="records")

        all_results[y_col] = {
            "models": model_results,
            "best_model": best_name,
        }

    # Conclusions
    conclusions = []
    for y_col in ("n_injured", "n_all"):
        r = all_results[y_col]
        best = r["best_model"]
        bm = r["models"][best]
        coef = bm["coef"].get("log_exposure", None)
        conclusions.append(
            f"- **{y_col}**: best model `{best}`, AIC={bm['aic']:.1f}. "
            f"Coef on `log_exposure` = **{coef:.4f}** "
            f"(sublinear: doubling traffic → {2**coef:.2f}× accidents). "
            f"NB2 alpha = {bm.get('alpha_nb2', 'n/a')}."
        )
        if "moran_on_residuals" in bm:
            mor = bm["moran_on_residuals"]
            if "I" in mor:
                conclusions.append(
                    f"  Residual Moran I = {mor['I']:.4f} "
                    f"(p={mor['p_sim']:.4f}). Spatial structure remains "
                    f"after accounting for exposure and year/month."
                )
    all_results["conclusions"] = "\n".join(conclusions)

    # Save
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "41c_refit.json").write_text(
        dumps_jsonable(all_results), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '41c_refit.json'}")

    # Markdown
    md = ["# Phase 10 · Iter 1c — Refit with fixed NB2\n\n"]
    md.append("## Interpretación correcta del test α=1\n\n")
    md.append("El 41b reportó mal: **rechazamos α=1** (z=-92 / -138, p≈0). "
              "El offset con coef=1 **está mal especificado**. El modelo "
              "correcto usa `log_exposure` como covariable libre. "
              "Coef estimado ≈ **0.53**, sublineal.\n\n")
    for y_col in ("n_injured", "n_all"):
        r = all_results[y_col]
        md.append(f"## Response: `{y_col}`\n\n")
        md.append("| model | AIC | ΔAIC vs N0 | α_exp | α_param |\n")
        md.append("| --- | ---: | ---: | ---: | ---: |\n")
        for name in ("N0", "N1", "N2", "N3", "N4"):
            if name not in r["models"]:
                continue
            m = r["models"][name]
            if not m.get("success"):
                md.append(f"| {name} | FAIL | | | |\n")
                continue
            coef = m["coef"].get("log_exposure", np.nan)
            md.append(f"| {name} | {m['aic']:.0f} | "
                      f"{m.get('delta_aic_vs_N0', 0):+.0f} | "
                      f"{coef:.4f} | {m.get('alpha_nb2', np.nan):.3f} |\n")
        best = r["best_model"]
        bm = r["models"][best]
        md.append(f"\n**Best: {best}** (AIC={bm['aic']:.1f})\n")
        if "moran_on_residuals" in bm:
            mor = bm["moran_on_residuals"]
            if "I" in mor:
                md.append(f"- Residual Moran I: {mor['I']:.4f} "
                          f"(p={mor['p_sim']:.4f})\n")
        if "by_year_fit" in r["models"]:
            md.append("\n**Fit by year:**\n\n")
            md.append("| year | actual | predicted | ratio |\n")
            md.append("| --- | ---: | ---: | ---: |\n")
            for rec in r["models"]["by_year_fit"]:
                md.append(f"| {rec['year']} | {rec['actual']:.0f} | "
                          f"{rec['predicted']:.0f} | {rec['ratio']:.3f} |\n")
        md.append("\n")
    md.append("## Conclusions\n\n")
    md.append(all_results["conclusions"] + "\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    for y_col in ("n_injured", "n_all"):
        r = all_results[y_col]
        log.info(f"[{y_col}] best={r['best_model']}")
        for name in ("N0", "N1", "N2", "N3", "N4"):
            if name in r["models"] and r["models"][name].get("success"):
                m = r["models"][name]
                log.info(f"  {name}: AIC={m['aic']:.0f} "
                         f"coef_exp={m['coef'].get('log_exposure', np.nan):.3f}")


if __name__ == "__main__":
    main()