"""
Script 41d — Fix `fitted_mu` bug in 41c and recompute Moran residuals.

Bug: `sm.NegativeBinomial.fit().fittedvalues` returns X·β (linear
predictor), not exp(X·β) (mean). Residuals based on it are wrong:
sqrt(negative) → NaN, per-hex Moran computed on 365/854 surviving
hexes instead of ~922.

Fix: reconstruct mu = exp(X @ beta) from the saved coefficients of N4,
recompute Pearson residuals, Moran on residuals, and by-year fits.

Does NOT refit models (AIC and coefs from 41c are valid).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import libpysal
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_JSON = Path("reports/tables/41d_fix_residuals.json")
OUT_MD = Path("reports/41d_fix_residuals.md")

log = logging.getLogger("phase10_41d")


def load_panel_with_design() -> tuple[pd.DataFrame, dict, dict]:
    """Load panel and pre-built design matrices for N4 per response."""
    import json
    panel = pd.read_parquet(DATA / "panel_hex_month_n_injured.parquet")

    # Add highway_ordinal_mean and tipo_via from accidents
    import geopandas as gpd
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
    panel = panel.dropna(subset=["log_exposure", "highway_ordinal_mean"]).copy()

    # Load coefficients from 41c
    refit = json.loads((Path("reports/tables/41c_refit.json")).read_text())
    return panel, refit


def build_design(panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Reconstruct the N4 design matrix manually (intercept + continuous + dummies)."""
    cols = []
    names = []

    cols.append(np.ones(len(panel)))
    names.append("Intercept")

    # year dummies (baseline 2019)
    for y in ["2020", "2021", "2022", "2023", "2024", "2025"]:
        cols.append((panel["year_str"] == y).astype(float).values)
        names.append(f"C(year_str)[T.{y}]")

    # month dummies (baseline 01)
    for m in ["02","03","04","05","06","07","08","09","10","11","12"]:
        cols.append((panel["month_str"] == m).astype(float).values)
        names.append(f"C(month_str)[T.{m}]")

    # tipo_via dummies (baseline = first alphabetical = "Calle"? no.
    # statsmodels uses first level alphabetically. The 41c coef list
    # shows: Avenida, Calle, Camino, Carretera, Cuesta, Glorieta,
    # Paseo, Plaza, Ronda, Travesía, Unknown. Baseline must be "Callejón"
    # or similar missing category. We'll reconstruct with the exact
    # categories seen and use "Callejón" as baseline (or whatever
    # statsmodels picked). Simpler: use 41c params dict directly.
    tv_levels = ["Avenida","Calle","Camino","Carretera","Cuesta",
                 "Glorieta","Paseo","Plaza","Ronda","Travesía","Unknown"]
    for level in tv_levels:
        cols.append((panel["tipo_via"] == level).astype(float).values)
        names.append(f"C(tipo_via)[T.{level}]")

    # continuous
    cols.append(panel["log_exposure"].values)
    names.append("log_exposure")
    cols.append(panel["highway_ordinal_mean"].values)
    names.append("highway_ordinal_mean")

    X = np.column_stack(cols)
    return X, names


def compute_mu(panel: pd.DataFrame, refit: dict, y_col: str) -> np.ndarray:
    """Reconstruct mu = exp(X @ beta) using saved coefficients."""
    X, names = build_design(panel)
    coef = refit[y_col]["models"]["N4"]["coef"]
    beta = np.array([coef.get(n, 0.0) for n in names])
    linpred = X @ beta
    # Clip to avoid extreme values
    linpred = np.clip(linpred, -20, 5)
    return np.exp(linpred)


def moran_on_residuals(panel: pd.DataFrame, y: np.ndarray, mu: np.ndarray) -> dict:
    # Pearson residuals
    resid = (y - mu) / np.sqrt(mu + 1e-10)
    df = panel[["hex_id"]].copy()
    df["resid"] = resid
    per_hex = df.groupby("hex_id").agg(mean_resid=("resid", "mean")).reset_index()

    hexes = ms_io.load_geometry_parquet(DATA / "moran_hex250m.parquet",
                                         expected_crs="EPSG:25830")
    merged = hexes.merge(per_hex, on="hex_id", how="inner")
    merged = merged[merged["mean_resid"].notna()].reset_index(drop=True)

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
    return {"I": float(m.I), "p_sim": float(m.p_sim),
            "n_units": int(len(merged))}


def by_year_fit(panel: pd.DataFrame, y: np.ndarray, mu: np.ndarray) -> list[dict]:
    df = panel[["year"]].copy()
    df["y"] = y
    df["mu"] = mu
    out = (df.groupby("year")
              .agg(actual=("y", "sum"), predicted=("mu", "sum"))
              .reset_index())
    out["ratio"] = out["predicted"] / out["actual"]
    return out.to_dict(orient="records")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    panel, refit = load_panel_with_design()
    log.info(f"panel: {len(panel):,} rows, {panel['hex_id'].nunique()} hexes")

    results = {}
    for y_col in ("n_injured", "n_all"):
        log.info(f"=== {y_col} ===")
        y = panel[y_col].to_numpy(dtype=np.float64)
        mu = compute_mu(panel, refit, y_col)
        log.info(f"  mu: mean={mu.mean():.4f}, min={mu.min():.4f}, max={mu.max():.2f}")
        log.info(f"  actual y: mean={y.mean():.4f}, sum={y.sum():.0f}")
        log.info(f"  predicted sum: {mu.sum():.0f}")

        mor = moran_on_residuals(panel, y, mu)
        log.info(f"  Moran on residuals: I={mor['I']:.4f} (p={mor['p_sim']:.4f}, n={mor['n_units']})")

        by_year = by_year_fit(panel, y, mu)
        log.info(f"  by year ratio:")
        for rec in by_year:
            log.info(f"    {rec['year']}: actual={rec['actual']:.0f}, predicted={rec['predicted']:.0f}, ratio={rec['ratio']:.3f}")

        results[y_col] = {
            "mu_stats": {"mean": float(mu.mean()), "min": float(mu.min()),
                         "max": float(mu.max()), "sum": float(mu.sum())},
            "moran_on_residuals": mor,
            "by_year": by_year,
            "delta_from_41b_M1": (
                {"I_41b_M1": 0.2159, "reduction": 0.2159 - mor["I"]}
                if y_col == "n_injured" else
                {"I_41b_M1": 0.2158, "reduction": 0.2158 - mor["I"]}
            ),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")

    md = ["# Phase 10 · Iter 1d — Residuals fix\n\n"]
    md.append("Reconstruye `mu = exp(X·β)` desde los coeficientes de N4 en 41c.\n\n")
    for y_col in ("n_injured", "n_all"):
        r = results[y_col]
        md.append(f"## `{y_col}`\n\n")
        md.append(f"- mu: mean={r['mu_stats']['mean']:.3f}, "
                  f"sum={r['mu_stats']['sum']:.0f} (vs actual sum en la tabla by-year)\n")
        md.append(f"- **Moran on residuals: I={r['moran_on_residuals']['I']:.4f}** "
                  f"(p={r['moran_on_residuals']['p_sim']:.4f}, "
                  f"n={r['moran_on_residuals']['n_units']})\n")
        md.append(f"  - Reducción vs M1 (41b): {r['delta_from_41b_M1']['reduction']:+.4f}\n\n")
        md.append("| year | actual | predicted | ratio |\n")
        md.append("| --- | ---: | ---: | ---: |\n")
        for rec in r["by_year"]:
            md.append(f"| {rec['year']} | {rec['actual']:.0f} | "
                      f"{rec['predicted']:.0f} | {rec['ratio']:.3f} |\n")
        md.append("\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    log.info(f"saved {OUT_JSON}")
    log.info(f"saved {OUT_MD}")


if __name__ == "__main__":
    main()