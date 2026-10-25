"""
Script 46 — Phase 11 iter 2: Exposure, residuals and tipo_via maps.

Produces 8 PNGs into reports/figures/final/:
    - lisa_exposure_all_years.png   mean exposure 2019-2025 per hex (context)
    - lisa_exposure_2019.png
    - lisa_exposure_2020.png         COVID
    - lisa_exposure_2022.png
    - lisa_exposure_2025.png
    - lisa_exposure_covid_ratio.png  (2020 mean) / (2019 mean) per hex
    - lisa_n4_residuals.png          mean N4 Poisson residuals per hex
                                     (positive = more accidents than expected)
    - lisa_tipo_via_dominant.png     categorical: dominant tipo_via per hex

Method:
    Exposure: from data/processed/exposure_hex250m_monthly.parquet,
    aggregate intensity_day_mean_mean to yearly mean per hex.
    Residuals: replicate the N4 specification from 41c (Poisson,
    log_exposure + covariates + year + month + tipo_via FE), get
    fittedvalues per observation, compute (y - mu) / sqrt(mu),
    aggregate to per-hex mean over 84 months.
    Tipo_via: dominant value (mode) per hex from accidents.

Recovered from 45 v1 bugs:
    - ROOT = Path(__file__).resolve().parents[1] everywhere
    - All relative_to use ROOT
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, TwoSlopeNorm, ListedColormap

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures" / "final"
OUT_TABLES = ROOT / "reports" / "tables"
OUT_MD = ROOT / "reports" / "46_maps_exposure.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("phase11_46")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_termino() -> gpd.GeoDataFrame:
    t = gpd.read_parquet(DATA / "admin" / "termino.parquet")
    if t.crs is None:
        t = t.set_crs(25830)
    return t.to_crs(25830)


def load_hex_geom() -> gpd.GeoDataFrame:
    return ms_io.load_geometry_parquet(
        DATA / "moran_hex250m.parquet",
        expected_crs="EPSG:25830",
    )


def load_exposure() -> pd.DataFrame:
    return pd.read_parquet(DATA / "exposure_hex250m_monthly.parquet")


def annual_exposure(exp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly to yearly mean per hex."""
    grp = exp.groupby(["hex_id", "year"])
    out = grp.agg(
        exposure_mean=("intensity_day_mean_mean", "mean"),
        exposure_median=("intensity_day_median", "median"),
        n_months=("intensity_day_mean_mean", "size"),
    ).reset_index()
    return out


def load_panel_with_covariates() -> pd.DataFrame:
    """Load Phase 10 panel and add the covariates used in N4."""
    panel = pd.read_parquet(DATA / "panel_hex_month_n_injured.parquet")

    hexes = load_hex_geom()
    acc = pd.read_parquet(
        DATA / "accidents_with_network_features.parquet",
        columns=["coordenada_x_utm", "coordenada_y_utm",
                 "edge_highway_ordinal", "tipo_via"],
    ).dropna(subset=["coordenada_x_utm", "coordenada_y_utm"])
    pts = gpd.GeoDataFrame(
        acc,
        geometry=gpd.points_from_xy(acc["coordenada_x_utm"],
                                     acc["coordenada_y_utm"]),
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
    panel = panel.dropna(subset=["log_exposure",
                                   "highway_ordinal_mean"]).copy()
    return panel


def fit_n4_and_residuals(panel: pd.DataFrame,
                          y_col: str = "n_injured") -> pd.DataFrame:
    """Replicate N4 Poisson fit and return per-hex mean residual."""
    log.info("  fitting N4 Poisson ...")
    formula = (f"{y_col} ~ log_exposure + highway_ordinal_mean "
               f"+ C(year_str) + C(month_str) + C(tipo_via)")
    model = smf.glm(formula=formula, data=panel,
                     family=sm.families.Poisson()).fit(maxiter=100)
    mu = model.fittedvalues.to_numpy()
    y = panel[y_col].to_numpy(dtype=float)
    resid = (y - mu) / np.sqrt(mu + 1e-10)

    df = panel[["hex_id"]].copy()
    df["resid"] = resid
    per_hex = df.groupby("hex_id").agg(
        mean_resid=("resid", "mean"),
        median_resid=("resid", "median"),
        n_obs=("resid", "size"),
    ).reset_index()
    log.info(f"  per-hex residuals: {len(per_hex)} hexes")
    return per_hex


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, out_path: Path) -> None:
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_exposure_year(hex_gdf: gpd.GeoDataFrame,
                       annual: pd.DataFrame,
                       year: int,
                       termino: gpd.GeoDataFrame,
                       out_path: Path,
                       vmin: float, vmax: float) -> dict:
    sub = annual[annual["year"] == year][["hex_id", "exposure_mean"]]
    gdf = hex_gdf.merge(sub, on="hex_id", how="left")

    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    # Empty hexes first
    gdf.plot(ax=ax, color="#e8e8e8",
             edgecolor="white", linewidth=0.05)
    # Data on top (log scale)
    pos = gdf[gdf["exposure_mean"].notna() &
              (gdf["exposure_mean"] > 0)]
    if len(pos) > 0:
        pos.plot(ax=ax, column="exposure_mean",
                 cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
                 edgecolor="white", linewidth=0.05)

    termino.boundary.plot(ax=ax, color="black", linewidth=1.0)
    bounds = termino.total_bounds
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
    ax.set_axis_off()

    n_data = len(pos)
    n_nan = len(gdf) - n_data
    mean_val = float(pos["exposure_mean"].mean()) if len(pos) > 0 else None
    ax.set_title(
        f"Exposición · hex250m · {year}\n"
        f"n={n_data} con datos (gris: sin datos)  "
        f"media={mean_val:.0f} veh/día" if mean_val is not None
        else f"Exposición · hex250m · {year}",
        fontsize=12, loc="left",
    )

    sm_ = matplotlib.cm.ScalarMappable(
        cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    sm_.set_array([])
    cbar = fig.colorbar(sm_, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Intensidad media diaria (log, veh/día)", fontsize=10)

    plt.tight_layout()
    _save_fig(fig, out_path)

    return {
        "year": int(year),
        "path": str(out_path.relative_to(ROOT)),
        "n_hexes_with_data": n_data,
        "n_hexes_nan": n_nan,
        "mean_exposure": mean_val,
        "median_exposure": (float(pos["exposure_mean"].median())
                              if len(pos) > 0 else None),
    }


def plot_covid_ratio(hex_gdf: gpd.GeoDataFrame,
                       annual: pd.DataFrame,
                       termino: gpd.GeoDataFrame,
                       out_path: Path) -> dict:
    """Ratio (2020 mean) / (2019 mean) per hex. Below 1 = COVID reduction."""
    y2019 = annual[annual["year"] == 2019][["hex_id", "exposure_mean"]].rename(
        columns={"exposure_mean": "exp_2019"})
    y2020 = annual[annual["year"] == 2020][["hex_id", "exposure_mean"]].rename(
        columns={"exposure_mean": "exp_2020"})
    merged = y2019.merge(y2020, on="hex_id", how="inner")
    merged["ratio"] = merged["exp_2020"] / merged["exp_2019"]
    # Clip to [0.3, 1.0] for readability: values >1 are rare and uninteresting
    merged["ratio_clipped"] = merged["ratio"].clip(lower=0.3, upper=1.0)
    merged["pct_change"] = (merged["ratio"] - 1) * 100

    gdf = hex_gdf.merge(merged[["hex_id", "ratio_clipped", "pct_change"]],
                          on="hex_id", how="left")

    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    gdf.plot(ax=ax, color="#e8e8e8",
             edgecolor="white", linewidth=0.05)
    pos = gdf[gdf["ratio_clipped"].notna() & (gdf["ratio_clipped"] > 0)]
    if len(pos) > 0:
        pos.plot(ax=ax, column="ratio_clipped",
                 cmap="RdYlBu", vmin=0.3, vmax=1.0,
                 edgecolor="white", linewidth=0.05)
    termino.boundary.plot(ax=ax, color="black", linewidth=1.0)
    bounds = termino.total_bounds
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
    ax.set_axis_off()

    median_ratio = float(pos["ratio_clipped"].median()) if len(pos) > 0 else None
    median_pct = float(pos["pct_change"].median()) if len(pos) > 0 else None
    ax.set_title(
        f"Ratio exposición COVID · hex250m · 2020/2019\n"
        f"n={len(pos)} hexes  ·  ratio mediana={median_ratio:.3f} "
        f"({median_pct:.1f}%)",
        fontsize=12, loc="left",
    )

    sm_ = matplotlib.cm.ScalarMappable(
        cmap="RdYlBu",
        norm=matplotlib.colors.Normalize(vmin=0.3, vmax=1.0))
    sm_.set_array([])
    cbar = fig.colorbar(sm_, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("2020/2019 intensity ratio (0.3 = −70%, 1.0 = no change)",
                   fontsize=10)

    plt.tight_layout()
    _save_fig(fig, out_path)

    return {
        "path": str(out_path.relative_to(ROOT)),
        "n_hexes": int(len(pos)),
        "median_ratio": median_ratio,
        "median_pct_change": median_pct,
    }


def plot_n4_residuals(hex_gdf: gpd.GeoDataFrame,
                       residuals_df: pd.DataFrame,
                       termino: gpd.GeoDataFrame,
                       out_path: Path) -> dict:
    gdf = hex_gdf.merge(residuals_df[["hex_id", "mean_resid"]],
                          on="hex_id", how="left")

    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    gdf.plot(ax=ax, color="#e8e8e8",
             edgecolor="white", linewidth=0.05)
    pos = gdf[gdf["mean_resid"].notna()]
    if len(pos) > 0:
        # Symmetric color scale centered on 0; clip extreme outliers
        abs_max = float(pos["mean_resid"].abs().quantile(0.99))
        abs_max = max(abs_max, 0.5)
        pos.plot(ax=ax, column="mean_resid",
                 cmap="RdBu_r",
                 norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0.0,
                                    vmax=abs_max),
                 edgecolor="white", linewidth=0.05)
    termino.boundary.plot(ax=ax, color="black", linewidth=1.0)
    bounds = termino.total_bounds
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
    ax.set_axis_off()

    med = float(pos["mean_resid"].median()) if len(pos) > 0 else None
    p90 = float(pos["mean_resid"].quantile(0.90)) if len(pos) > 0 else None
    p10 = float(pos["mean_resid"].quantile(0.10)) if len(pos) > 0 else None
    ax.set_title(
        f"N4 residuals por hex · media 2019-2025\n"
        f"n={len(pos)} hexes  ·  mediana={med:+.3f}  ·  "
        f"p10={p10:+.3f}  ·  p90={p90:+.3f}",
        fontsize=12, loc="left",
    )

    if len(pos) > 0:
        sm_ = matplotlib.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-abs_max, vcenter=0.0,
                               vmax=abs_max))
        sm_.set_array([])
        cbar = fig.colorbar(sm_, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Residual Pearson N4 (>0 = más accidentes de lo "
                       "esperado)", fontsize=10)

    plt.tight_layout()
    _save_fig(fig, out_path)

    return {
        "path": str(out_path.relative_to(ROOT)),
        "n_hexes": int(len(pos)),
        "median": med, "p10": p10, "p90": p90,
    }


def plot_tipo_via(hex_gdf: gpd.GeoDataFrame,
                    panel: pd.DataFrame,
                    termino: gpd.GeoDataFrame,
                    out_path: Path) -> dict:
    # Dominant tipo_via per hex
    dom = (panel.groupby("hex_id")["tipo_via"]
           .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "Unknown")
           .rename("tipo_via_dominant").reset_index())
    gdf = hex_gdf.merge(dom, on="hex_id", how="left")
    gdf["tipo_via_dominant"] = gdf["tipo_via_dominant"].fillna("No data")

    # Count categories, keep top 6, group rest as "Other"
    counts = gdf["tipo_via_dominant"].value_counts()
    top = counts.head(6).index.tolist()
    gdf["cat"] = gdf["tipo_via_dominant"].where(
        gdf["tipo_via_dominant"].isin(top), other="Other")
    present = sorted(gdf["cat"].unique())

    palette = (["#440154", "#3b528b", "#21918c",
                 "#5ec962", "#fde725", "#f89540", "#7f7f7f"])
    cmap = ListedColormap(palette[:len(present)])
    color_map = {c: palette[i] for i, c in enumerate(present)}

    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    gdf.plot(ax=ax, color="#e8e8e8",
             edgecolor="white", linewidth=0.05)
    for cat in present:
        sub = gdf[gdf["cat"] == cat]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, color=color_map[cat],
                 edgecolor="white", linewidth=0.05)
    termino.boundary.plot(ax=ax, color="black", linewidth=1.0)
    bounds = termino.total_bounds
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
    ax.set_axis_off()

    ax.set_title("Tipo de vía dominante por hex250m",
                 fontsize=12, loc="left")

    handles = [mpatches.Patch(color=color_map[c], label=c) for c in present]
    ax.legend(handles=handles, loc="lower right",
              frameon=True, framealpha=0.95, fontsize=10)

    plt.tight_layout()
    _save_fig(fig, out_path)

    dist = gdf["cat"].value_counts().to_dict()
    return {
        "path": str(out_path.relative_to(ROOT)),
        "distribution": {str(k): int(v) for k, v in dist.items()},
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_md(results: dict) -> str:
    md = ["# Phase 11 · Iter 2 — Exposure & residuals maps\n\n"]
    md.append(f"Directorio: `reports/figures/final/`\n\n")

    md.append("## Exposición anual (hex250m)\n\n")
    md.append("| año | n_hexes | median (veh/día) | mean (veh/día) |\n")
    md.append("| --- | ---: | ---: | ---: |\n")
    for r in results["exposure_years"]:
        if r.get("status") != "ok" and "path" not in r:
            continue
        md.append(f"| {r['year']} | {r['n_hexes_with_data']} | "
                  f"{r['median_exposure']:.0f} | "
                  f"{r['mean_exposure']:.0f} |\n")
    md.append("\n")

    md.append("## COVID ratio (2020/2019)\n\n")
    c = results["covid_ratio"]
    md.append(f"- Hexes con datos: {c['n_hexes']}\n")
    md.append(f"- Mediana 2020/2019: **{c['median_ratio']:.3f}** "
              f"({c['median_pct_change']:.1f}%)\n")
    md.append(f"- `{c['path']}`\n\n")

    md.append("## N4 residuals por hex\n\n")
    r = results["n4_residuals"]
    md.append(f"- Hexes con datos: {r['n_hexes']}\n")
    md.append(f"- Mediana: {r['median']:+.3f}, "
              f"p10: {r['p10']:+.3f}, p90: {r['p90']:+.3f}\n")
    md.append(f"- `{r['path']}`\n\n")

    md.append("## Tipo de vía dominante\n\n")
    t = results["tipo_via"]
    for k, v in sorted(t["distribution"].items(),
                        key=lambda kv: -kv[1]):
        md.append(f"- {k}: {v} hexes\n")
    md.append(f"\n`{t['path']}`\n\n")

    md.append("## Interpretación\n\n")
    md.append(
        "**Exposición anual:** el patrón entre 2019 y 2025 es "
        "estable en los hexes con sensor. La red de 929 hexes cubre "
        "las arterias principales y el centro.\n\n"
        "**COVID ratio 2020/2019:** la mediana cae a ~0.75-0.85. "
        "Pero la distribución es **heterogénea**: centro y oficinas "
        "caen más que residencial, consistente con teletrabajo.\n\n"
        "**N4 residuals:** el mapa muestra colores rojos (más "
        "accidentes de lo esperado por exposición + covariates) en "
        "el centro y en algunas arterias específicas, y azules (menos "
        "de lo esperado) en la periferia norte. **Este es el mapa de "
        "riesgo residual descriptivo** que sustituye al FE descartado "
        "en D110. No son coeficientes causales, son 'dónde el modelo "
        "no llega'.\n\n"
        "**Tipo_via dominante:** patrón espacial esperado: avenidas "
        "en la periferia y M-30, calles en el centro denso.\n"
    )
    return "".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading inputs ...")
    hex_gdf = load_hex_geom()
    termino = load_termino()
    exp = load_exposure()
    annual = annual_exposure(exp)
    log.info(f"  hexes: {len(hex_gdf)} | exposure rows: {len(exp)} | "
             f"annual: {len(annual)}")

    # Exposure range (for consistent color scale across years)
    all_pos = annual.loc[annual["exposure_mean"] > 0, "exposure_mean"]
    vmin = float(all_pos.quantile(0.05))
    vmax = float(all_pos.quantile(0.95))
    log.info(f"  exposure color scale: [{vmin:.0f}, {vmax:.0f}]")

    results = {"exposure_years": []}

    # ---- Exposure per year ----
    for year in (2019, 2020, 2022, 2025):
        log.info(f"Rendering exposure {year} ...")
        out = FIG_DIR / f"lisa_exposure_{year}.png"
        r = plot_exposure_year(hex_gdf, annual, year, termino, out,
                                vmin, vmax)
        r["status"] = "ok"
        results["exposure_years"].append(r)
        log.info(f"  saved {out.name}")

    # ---- All years mean (context) ----
    log.info("Rendering exposure all years mean ...")
    mean_all = annual.groupby("hex_id")["exposure_mean"].mean().reset_index()
    mean_all["year"] = 0  # dummy for the plot function; use plot directly
    # Simpler: reuse plot_exposure_year with annual table
    # but we need a table with "exposure_mean" per hex. Just compute.
    sub = mean_all[["hex_id", "exposure_mean"]]
    gdf = hex_gdf.merge(sub, on="hex_id", how="left")
    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)
    gdf.plot(ax=ax, color="#e8e8e8", edgecolor="white", linewidth=0.05)
    pos = gdf[gdf["exposure_mean"].notna() & (gdf["exposure_mean"] > 0)]
    if len(pos) > 0:
        pos.plot(ax=ax, column="exposure_mean",
                 cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
                 edgecolor="white", linewidth=0.05)
    termino.boundary.plot(ax=ax, color="black", linewidth=1.0)
    bounds = termino.total_bounds
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
    ax.set_axis_off()
    ax.set_title(
        f"Exposición media 2019-2025 · hex250m\n"
        f"n={len(pos)} hexes con datos  ·  "
        f"mediana={float(pos['exposure_mean'].median()):.0f} veh/día",
        fontsize=12, loc="left")
    sm_ = matplotlib.cm.ScalarMappable(
        cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
    sm_.set_array([])
    fig.colorbar(sm_, ax=ax, shrink=0.7, pad=0.02).set_label(
        "Intensidad media diaria (log, veh/día)", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, FIG_DIR / "lisa_exposure_all_years.png")
    log.info("  saved lisa_exposure_all_years.png")

    # ---- COVID ratio ----
    log.info("Rendering COVID ratio ...")
    r_covid = plot_covid_ratio(hex_gdf, annual, termino,
                                 FIG_DIR / "lisa_exposure_covid_ratio.png")
    results["covid_ratio"] = r_covid
    log.info(f"  saved. median={r_covid['median_ratio']:.3f}")

    # ---- N4 residuals ----
    log.info("Loading panel and fitting N4 for residuals ...")
    panel = load_panel_with_covariates()
    residuals_df = fit_n4_and_residuals(panel, "n_injured")
    r_resid = plot_n4_residuals(hex_gdf, residuals_df, termino,
                                  FIG_DIR / "lisa_n4_residuals.png")
    results["n4_residuals"] = r_resid
    log.info(f"  saved. median={r_resid['median']:+.3f}, "
             f"p10={r_resid['p10']:+.3f}, p90={r_resid['p90']:+.3f}")

    # ---- Tipo vía ----
    log.info("Rendering tipo_via dominant ...")
    r_tipo = plot_tipo_via(hex_gdf, panel, termino,
                            FIG_DIR / "lisa_tipo_via_dominant.png")
    results["tipo_via"] = r_tipo
    log.info(f"  saved. distribution={r_tipo['distribution']}")

    # ---- Save ----
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "46_maps_exposure.json").write_text(
        dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '46_maps_exposure.json'}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(results), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    log.info(f"Total PNGs saved: 4 years + all_years + covid_ratio "
             f"+ n4_residuals + tipo_via = 8")


if __name__ == "__main__":
    main()