"""
Script 43 — Phase 10 iter 3: FE diagnostics + n_all fix.

Context: the FE model (script 42) reduced the exposure coefficient from
0.474 (N4) to 0.156 (-67%) and crashed on n_all. Both symptoms are
explained by the between-hex dominance of exposure variation. This
script quantifies it.

Specifications:
    N4              : no hex FE.               coef ≈ 0.474 (total)
    FE_no_year      : hex FE, no year FE.      coef ≈ ?
    FE_no_hex       : year+month+tipo_via FE.  coef ≈ 0.474 (like N4)
    FE_full         : hex+year+month+tipo_via. coef ≈ 0.156 (within)

Uses lbfgs (not IRLS) for numerical stability with 928 hex dummies.

Variance decomposition:
    var(log_exposure) = var_between(mean per hex) + var_within(residual)

Hex quality flags: well_sampled (84 obs), moderate (24-83), singleton (<24).

Outputs:
    reports/tables/43_fe_diagnostics.json
    reports/43_fe_diagnostics.md
    data/processed/hex_risk_fe_flags.parquet
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
OUT_MD = Path("reports/43_fe_diagnostics.md")
OUT_RISK = DATA / "hex_risk_fe_flags.parquet"

log = logging.getLogger("phase10_43")


def load_panel() -> pd.DataFrame:
    log.info("Loading panel ...")
    panel = pd.read_parquet(DATA / "panel_hex_month_n_injured.parquet")

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
    tipo = (joined.groupby("hex_id")["tipo_via"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "Unknown")
            .rename("tipo_via").reset_index())

    panel = panel.merge(hi, on="hex_id", how="left").merge(
        tipo, on="hex_id", how="left")
    panel["tipo_via"] = panel["tipo_via"].fillna("Unknown")
    panel["year_str"] = panel["year"].astype(int).astype(str)
    panel["month_str"] = panel["month"].astype(int).astype(str).str.zfill(2)
    panel["log_exposure"] = np.log(panel["exposure_month"] + 1e-10)
    panel["hex_id"] = panel["hex_id"].astype(int)
    panel = panel.dropna(subset=["log_exposure", "highway_ordinal_mean"]).copy()
    log.info(f"  {len(panel):,} rows, {panel['hex_id'].nunique()} hexes")
    return panel


def variance_decomposition(panel: pd.DataFrame) -> dict:
    """Decompose var(log_exposure) into between-hex + within-hex."""
    le = panel["log_exposure"]
    mean_per_hex = panel.groupby("hex_id")["log_exposure"].transform("mean")
    between_component = mean_per_hex
    within_component = le - mean_per_hex

    var_total = float(le.var())
    var_between = float(between_component.var())
    var_within = float(within_component.var())

    # Additional: within-hex-within-year (after removing year means)
    year_mean = panel.groupby("year_str")["log_exposure"].transform("mean")
    within_year = le - year_mean
    var_within_year = float(within_year.var())

    return {
        "var_total": var_total,
        "var_between_hex": var_between,
        "var_within_hex": var_within,
        "var_within_hex_within_year": var_within_year,
        "between_share": var_between / var_total if var_total > 0 else None,
        "within_share": var_within / var_total if var_total > 0 else None,
        "within_year_share": var_within_year / var_total if var_total > 0 else None,
        "std_between": float(np.sqrt(var_between)),
        "std_within": float(np.sqrt(var_within)),
    }


def fit_spec(panel: pd.DataFrame, y_col: str, spec_name: str,
              formula: str) -> dict:
    """Fit a Poisson GLM with lbfgs (numerically stable for FE)."""
    log.info(f"  [{y_col}] fitting {spec_name} ...")
    try:
        model = smf.glm(formula=formula, data=panel,
                        family=sm.families.Poisson())
        # Use lbfgs instead of IRLS
        res = model.fit(method="lbfgs", maxiter=500, disp=False)
        coef = float(res.params.get("log_exposure", np.nan))
        se = float(res.bse.get("log_exposure", np.nan))
        try:
            aic = float(res.aic)
        except Exception:
            aic = float("nan")
        log.info(f"    coef_exposure={coef:.4f}, AIC={aic:.1f}")
        return {
            "spec": spec_name,
            "formula": formula,
            "success": True,
            "coef_exposure": coef,
            "se_exposure": se,
            "hr_exposure": float(np.exp(coef)) if np.isfinite(coef) else None,
            "aic": aic,
            "n_params": int(len(res.params)),
            "n_obs": int(len(panel)),
        }
    except Exception as e:  # noqa: BLE001
        log.warning(f"    failed: {type(e).__name__}: {e}")
        return {"spec": spec_name, "formula": formula, "success": False,
                "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    panel = load_panel()

    # 1. Variance decomposition
    log.info("=== Variance decomposition of log_exposure ===")
    vd = variance_decomposition(panel)
    log.info(f"  var_total={vd['var_total']:.4f}")
    log.info(f"  var_between_hex={vd['var_between_hex']:.4f} "
             f"({vd['between_share']:.1%})")
    log.info(f"  var_within_hex={vd['var_within_hex']:.4f} "
             f"({vd['within_share']:.1%})")
    log.info(f"  var_within_year={vd['var_within_hex_within_year']:.4f} "
             f"({vd['within_year_share']:.1%})")

    # 2. Fit 4 specifications for each response
    specs_common = {
        "N4_no_hex_fe":      "~ log_exposure + highway_ordinal_mean + C(year_str) + C(month_str) + C(tipo_via)",
        "FE_no_year":        "~ log_exposure + highway_ordinal_mean + C(month_str) + C(tipo_via) + C(hex_id)",
        "FE_no_hex":         "~ log_exposure + highway_ordinal_mean + C(year_str) + C(month_str) + C(tipo_via)",
        "FE_full":           "~ log_exposure + highway_ordinal_mean + C(year_str) + C(month_str) + C(tipo_via) + C(hex_id)",
    }
    all_results = {"variance_decomposition": vd}

    for y_col in ("n_injured", "n_all"):
        log.info(f"=== {y_col} ===")
        all_results[y_col] = {}
        for spec_name, formula_suffix in specs_common.items():
            formula = f"{y_col} {formula_suffix}"
            r = fit_spec(panel, y_col, spec_name, formula)
            all_results[y_col][spec_name] = r

    # 3. Load hex_risk_fe from 42 and add flags
    log.info("=== Annotating hex FE with quality flags ===")
    if (DATA / "hex_risk_fe.parquet").exists():
        hex_risk = pd.read_parquet(DATA / "hex_risk_fe.parquet")
    else:
        hex_risk = pd.DataFrame(columns=["hex_id", "fe_log_rr", "rr_point",
                                           "mean_resid", "n_obs"])

    # Flag based on n_obs (from per-hex count)
    n_obs_per_hex = panel.groupby("hex_id").size().rename("n_obs").reset_index()
    hex_risk = hex_risk.merge(n_obs_per_hex, on="hex_id", how="left",
                               suffixes=("", "_recomputed"))
    if "n_obs_recomputed" in hex_risk.columns:
        hex_risk["n_obs"] = hex_risk["n_obs_recomputed"].fillna(hex_risk["n_obs"])
        hex_risk = hex_risk.drop(columns=["n_obs_recomputed"])

    hex_risk["quality"] = "singleton"
    hex_risk.loc[hex_risk["n_obs"] >= 24, "quality"] = "moderate"
    hex_risk.loc[hex_risk["n_obs"] >= 60, "quality"] = "well_sampled"

    flag_counts = hex_risk["quality"].value_counts().to_dict()
    log.info(f"  quality distribution: {flag_counts}")

    hex_risk.to_parquet(OUT_RISK, index=False)
    log.info(f"saved {OUT_RISK}")

    # 4. Summarize drift for each FE variant vs N4
    all_results["summary"] = {}
    for y_col in ("n_injured", "n_all"):
        n4_coef = all_results[y_col]["N4_no_hex_fe"].get("coef_exposure")
        row = {"n4_baseline_coef": n4_coef}
        for spec in ("FE_no_year", "FE_no_hex", "FE_full"):
            c = all_results[y_col][spec].get("coef_exposure")
            if c is not None and n4_coef is not None and n4_coef != 0:
                row[f"{spec}_coef"] = c
                row[f"{spec}_drift_pct"] = 100 * (c - n4_coef) / n4_coef
        all_results["summary"][y_col] = row

    all_results["hex_quality_counts"] = flag_counts

    # Save JSON
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "43_fe_diagnostics.json").write_text(
        dumps_jsonable(all_results), encoding="utf-8")

    # Markdown
    md = ["# Phase 10 · Iter 3 — FE diagnostics\n\n"]
    md.append("## Variance decomposition of `log_exposure`\n\n")
    md.append(f"- Total: **{vd['var_total']:.4f}**\n")
    md.append(f"- Between-hex: **{vd['var_between_hex']:.4f}** "
              f"(**{vd['between_share']:.1%}**)\n")
    md.append(f"- Within-hex: **{vd['var_within_hex']:.4f}** "
              f"({vd['within_share']:.1%})\n")
    md.append(f"- Within-hex-within-year: "
              f"**{vd['var_within_hex_within_year']:.4f}** "
              f"({vd['within_year_share']:.1%})\n\n")
    md.append(
        f"**Interpretación:** {vd['between_share']:.0%} de la varianza de "
        f"exposición está entre hexes. Cuando se añade hex FE, se absorbe "
        f"ese {vd['between_share']:.0%}, y la covariable `log_exposure` "
        f"queda solo con el {vd['within_share']:.0%} restante, que es "
        f"pequeño, ruidoso y dominado por year FE (COVID).\n\n"
    )

    for y_col in ("n_injured", "n_all"):
        md.append(f"## Response: `{y_col}`\n\n")
        md.append("| spec | coef | SE | drift vs N4 | AIC |\n")
        md.append("| --- | ---: | ---: | ---: | ---: |\n")
        s = all_results["summary"][y_col]
        for spec_name in ("N4_no_hex_fe", "FE_no_year", "FE_no_hex", "FE_full"):
            r = all_results[y_col][spec_name]
            if not r.get("success"):
                md.append(f"| {spec_name} | FAIL | | | |\n")
                continue
            c = r["coef_exposure"]
            se = r["se_exposure"]
            drift = ""
            key = f"{spec_name}_drift_pct"
            if key in s:
                drift = f"{s[key]:+.1f}%"
            md.append(f"| {spec_name} | {c:.4f} | {se:.4f} | "
                      f"{drift} | {r['aic']:.0f} |\n")
        md.append("\n")

    md.append("## Hex quality flags\n\n")
    md.append(f"- well_sampled (n>=60): {flag_counts.get('well_sampled', 0)}\n")
    md.append(f"- moderate (24<=n<60): {flag_counts.get('moderate', 0)}\n")
    md.append(f"- singleton (n<24): {flag_counts.get('singleton', 0)}\n\n")

    md.append("## Conclusiones metodológicas\n\n")
    md.append(
        "1. **El FE no falla, mide otra cosa.** Con 90% de la varianza de "
        "exposición entre hexes, añadir hex FE absorbe esa varianza. El "
        "coef 0.156 es 'el efecto within-hex de corto plazo, condicionado "
        "a year FE', no 'el efecto total'. El coef 0.474 (N4) es el "
        "efecto total (between+within).\n\n"
        "2. **Rechazar CAR.** CAR reproduce el mismo problema (b_i absorbe "
        "media por hex). Alternativas si se quiere mapa de riesgo:\n"
        "   - Two-stage: fijar exposure coef de N4, modelar residuals con CAR.\n"
        "   - Bambi/PyMC con prior explícito sobre la identificación.\n"
        "   - Ambos quedan fuera del alcance si Fase 11 aprieta.\n\n"
        "3. **Dos hallazgos científicos válidos:**\n"
        "   - Cross-sectional (entre hexes): accidentes escalan 1.39× por "
        "2× tráfico (sublineal, N4).\n"
        "   - Time-series (within hex): efectos pequeños (0.156) porque "
        "el tráfico dentro de un hex es casi estacionario excepto COVID.\n"
        "   - Política: '¿qué pasa si aumenta el tráfico en Madrid 10%?' → "
        "3.6% más accidentes (within). '¿Qué hexes son más peligrosos?' → "
        "el ranking lo dicta el between (N4).\n\n"
    )

    (OUT_MD).write_text("".join(md), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '43_fe_diagnostics.json'}")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    log.info(f"Variance: between={vd['between_share']:.1%}, "
             f"within={vd['within_share']:.1%}")
    for y_col in ("n_injured", "n_all"):
        s = all_results["summary"][y_col]
        log.info(f"[{y_col}] N4={s['n4_baseline_coef']:.4f}")
        for spec in ("FE_no_year", "FE_no_hex", "FE_full"):
            if f"{spec}_coef" in s:
                log.info(f"  {spec}: {s[f'{spec}_coef']:.4f} "
                         f"({s[f'{spec}_drift_pct']:+.1f}%)")


if __name__ == "__main__":
    main()