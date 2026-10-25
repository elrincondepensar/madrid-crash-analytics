"""
Script 45 — Phase 11 iter 1 (v2): LISA cluster maps.

Fixes vs v1:
  - load_unit_geometry now loads the moran_*.parquet (which is self-
    sufficient: contains geometry + *_cluster + *_p_adj columns).
    No separate geometry load, no merge step in render_single.
  - relative_to uses Path(__file__).resolve().parents[1] (project root),
    not Path.cwd().parent (parent of project dir).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures" / "final"
OUT_TABLES = ROOT / "reports" / "tables"
OUT_MD = ROOT / "reports" / "45_maps_lisa.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("phase11_45")

LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "NS": "#f0f0f0",
    "NA": "#7f7f7f",
}
LISA_LABELS = {
    "HH": "HH · hotspot",
    "LL": "LL · coldspot",
    "HL": "HL · outlier alto",
    "LH": "LH · outlier bajo",
    "NS": "NS · no significativo",
    "NA": "NA · sin datos",
}
LISA_ORDER = ["HH", "LL", "HL", "LH", "NS", "NA"]

VAR_LABELS = {
    "counts_all": "Recuentos · todos",
    "counts_injured": "Recuentos · heridos",
    "kde_all": "KDE · todos",
    "injury_rate_eb": "Tasa heridos (EB)",
}
UNIT_LABELS = {
    "hex250m": "Hexágonos 250m",
    "barrios": "Barrios",
    "distritos": "Distritos",
    "callejero": "Callejero (KNN)",
}

# Each entry: path to the moran parquet (self-sufficient: geometry +
# clusters + p_adj). id_col is the unit identifier.
UNITS = {
    "hex250m": {
        "moran": DATA / "moran_hex250m.parquet",
        "id_col": "hex_id",
        "vars": ["counts_all", "counts_injured", "kde_all",
                 "injury_rate_eb"],
    },
    "barrios": {
        "moran": DATA / "moran_barrios.parquet",
        "id_col": "COD_BAR",
        "vars": ["counts_all", "counts_injured", "kde_all",
                 "injury_rate_eb"],
    },
    "distritos": {
        "moran": DATA / "moran_distritos.parquet",
        "id_col": "COD_DIS",
        "vars": ["counts_all", "counts_injured", "kde_all",
                 "injury_rate_eb"],
    },
    "callejero": {
        "moran": DATA / "moran_callejero.parquet",
        "id_col": "TOP_ID",
        "vars": ["counts_all", "kde_all"],
    },
}


def load_moran_json() -> dict:
    p = ROOT / "reports" / "tables" / "37_moran.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_moran_gdf(unit: str) -> gpd.GeoDataFrame:
    """Load the moran_*.parquet as a GeoDataFrame (it is self-sufficient)."""
    cfg = UNITS[unit]
    gdf = ms_io.load_geometry_parquet(cfg["moran"],
                                        expected_crs="EPSG:25830")
    return gdf


def get_moran_I(moran_json: dict, unit: str, var: str) -> dict:
    u = moran_json.get(unit, {})
    v = u.get("variables", {}).get(var, {})
    return {
        "I": v.get("moran_I"),
        "pct_HH": v.get("pct_HH"),
        "pct_LL": v.get("pct_LL"),
        "n": v.get("n_units"),
    }


def prep_cluster_col(gdf: gpd.GeoDataFrame, cluster_col: str) -> gpd.GeoDataFrame:
    """Normalize the cluster column to a LISA category name."""
    out = gdf.copy()
    if cluster_col not in out.columns:
        out[cluster_col] = "NA"
    out[cluster_col] = out[cluster_col].fillna("NA")
    out.loc[~out[cluster_col].isin(LISA_COLORS.keys()), cluster_col] = "NA"
    return out


def render_single(unit: str, var: str,
                   unit_gdf: gpd.GeoDataFrame,
                   moran_json: dict,
                   termino: gpd.GeoDataFrame,
                   out_path: Path) -> dict:
    cfg = UNITS[unit]
    id_col = cfg["id_col"]
    cluster_col = f"{var}_cluster"

    if cluster_col not in unit_gdf.columns:
        return {"unit": unit, "var": var,
                "status": f"missing_column:{cluster_col}"}

    gdf = prep_cluster_col(unit_gdf, cluster_col)

    info = get_moran_I(moran_json, unit, var)
    I_str = f"I={info['I']:.3f}" if info["I"] is not None else "I=n/a"
    HH_str = (f"HH={info['pct_HH']*100:.1f}%"
              if info["pct_HH"] is not None else "")
    LL_str = (f"LL={info['pct_LL']*100:.1f}%"
              if info["pct_LL"] is not None else "")

    fig, ax = plt.subplots(figsize=(11, 10), dpi=110)

    # Base fill: NA (so empty polygons aren't white)
    gdf.plot(ax=ax, color=LISA_COLORS["NA"],
             edgecolor="white", linewidth=0.1)

    # Overlay LISA categories (NA last so it stays on top of an empty layer)
    for cat in ["NS", "LL", "LH", "HL", "HH", "NA"]:
        sub = gdf[gdf[cluster_col] == cat]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, color=LISA_COLORS[cat],
                 edgecolor="white", linewidth=0.2)

    termino.boundary.plot(ax=ax, color="black", linewidth=1.0)

    bounds = termino.total_bounds
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
    ax.set_axis_off()

    ax.set_title(
        f"LISA · {UNIT_LABELS[unit]} · {VAR_LABELS[var]}\n"
        f"{I_str}  ·  {HH_str}  ·  {LL_str}",
        fontsize=12, loc="left",
    )

    handles = [mpatches.Patch(color=LISA_COLORS[k], label=LISA_LABELS[k])
               for k in LISA_ORDER]
    ax.legend(handles=handles, loc="lower right",
              frameon=True, framealpha=0.95, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    cats = gdf[cluster_col].value_counts().to_dict()
    return {
        "unit": unit, "var": var, "status": "ok",
        "path": str(out_path.relative_to(ROOT)),
        "moran_I": info["I"],
        "n_units": int(len(gdf)),
        "counts": {k: int(v) for k, v in cats.items()},
    }


def render_composite(unit: str,
                      unit_gdf: gpd.GeoDataFrame,
                      moran_json: dict,
                      termino: gpd.GeoDataFrame,
                      out_path: Path) -> dict:
    cfg = UNITS[unit]
    vars_4 = ["counts_all", "counts_injured", "kde_all", "injury_rate_eb"]
    vars_4 = [v for v in vars_4 if f"{v}_cluster" in unit_gdf.columns]
    if not vars_4:
        return {"unit": unit, "status": "no_columns"}

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=100)
    axes = axes.flatten()
    bounds = termino.total_bounds

    for i, var in enumerate(vars_4):
        ax = axes[i]
        cluster_col = f"{var}_cluster"
        gdf = prep_cluster_col(unit_gdf, cluster_col)

        gdf.plot(ax=ax, color=LISA_COLORS["NA"],
                 edgecolor="white", linewidth=0.1)
        for cat in ["NS", "LL", "LH", "HL", "HH", "NA"]:
            sub = gdf[gdf[cluster_col] == cat]
            if len(sub) == 0:
                continue
            sub.plot(ax=ax, color=LISA_COLORS[cat],
                     edgecolor="white", linewidth=0.2)
        termino.boundary.plot(ax=ax, color="black", linewidth=0.8)
        ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
        ax.set_ylim(bounds[1] - 500, bounds[3] + 500)
        ax.set_axis_off()

        info = get_moran_I(moran_json, unit, var)
        I_str = f"I={info['I']:.3f}" if info["I"] is not None else "I=n/a"
        HH_str = (f" HH={info['pct_HH']*100:.1f}%"
                  if info["pct_HH"] is not None else "")
        LL_str = (f" LL={info['pct_LL']*100:.1f}%"
                  if info["pct_LL"] is not None else "")
        ax.set_title(f"{VAR_LABELS[var]} · {I_str}{HH_str}{LL_str}",
                     fontsize=11)

    handles = [mpatches.Patch(color=LISA_COLORS[k], label=LISA_LABELS[k])
               for k in LISA_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"LISA · {UNIT_LABELS[unit]} · 4 variables",
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "unit": unit, "status": "ok",
        "path": str(out_path.relative_to(ROOT)),
        "vars": vars_4,
    }


def render_md(results: dict) -> str:
    md = ["# Phase 11 · Iter 1 — LISA maps\n\n"]
    n_ok = sum(1 for m in results["maps"] if m.get("status") == "ok")
    n_comp = sum(1 for c in results["composites"]
                 if c.get("status") == "ok")
    md.append(f"**PNGs generados:** {n_ok} mapas individuales + "
              f"{n_comp} composites.\n\n")
    md.append(f"Directorio: `reports/figures/final/`\n\n")

    md.append("## Mapas individuales\n\n")
    md.append("| unit | variable | I | HH | LL | file |\n")
    md.append("| --- | --- | ---: | ---: | ---: | --- |\n")
    for m in results["maps"]:
        if m.get("status") != "ok":
            md.append(f"| {m['unit']} | {m.get('var', '?')} | - | - | - | "
                      f"*{m.get('status', 'error')}* |\n")
            continue
        c = m.get("counts", {})
        md.append(
            f"| {m['unit']} | {m['var']} | "
            f"{m.get('moran_I', 0):.3f} | "
            f"{c.get('HH', 0)} | {c.get('LL', 0)} | "
            f"`{m['path']}` |\n"
        )
    md.append("\n")

    md.append("## Composites 2×2\n\n")
    for c in results["composites"]:
        if c.get("status") == "ok":
            md.append(f"- `{c['path']}`\n")
    md.append("\n")

    md.append("## Interpretación\n\n")
    md.append(
        "**hex250m:** `counts_all` con I=0.677 (señal concentrada en "
        "pocos hotspots operativos). `counts_injured` muestra HH + LL "
        "bimodal. `kde_all` es el más informativo (I=0.934).\n\n"
        "**barrios:** `counts_*` mayoritariamente NS (I=0.09-0.15). "
        "`kde_all` retiene I=0.80 con HH=16.8%. Confirma D108: la "
        "agregación a barrio diluye la señal de recuentos.\n\n"
        "**distritos:** casi todo NS salvo `kde_all` (HH=4.8%).\n\n"
        "**callejero:** `counts_all` vacío (I=0.013). `kde_all` "
        "preserva estructura (I=0.959). MAUP del callejero.\n"
    )
    return "".join(md)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading municipal boundary ...")
    termino = gpd.read_parquet(DATA / "admin" / "termino.parquet")
    if termino.crs is None:
        termino = termino.set_crs(25830)
    termino = termino.to_crs(25830)
    log.info(f"  termino bounds: {termino.total_bounds}")

    moran_json = load_moran_json()
    log.info(f"  moran json keys: {list(moran_json.keys())[:5]}...")

    results = {"maps": [], "composites": []}

    for unit, cfg in UNITS.items():
        log.info(f"=== Unit: {unit} ===")
        try:
            unit_gdf = load_moran_gdf(unit)
            log.info(f"  loaded: {len(unit_gdf)} rows, "
                     f"columns: {[c for c in unit_gdf.columns if 'cluster' in c][:4]}")
        except Exception as e:  # noqa: BLE001
            log.error(f"  load failed: {type(e).__name__}: {e}")
            results["maps"].append({"unit": unit, "status": "load_failed"})
            continue

        for var in cfg["vars"]:
            out_path = FIG_DIR / f"lisa_{unit}_{var}.png"
            try:
                r = render_single(unit, var, unit_gdf, moran_json,
                                   termino, out_path)
                results["maps"].append(r)
                if r.get("status") == "ok":
                    log.info(f"  saved {out_path.name}")
                else:
                    log.warning(f"  {var}: {r.get('status')}")
            except Exception as e:  # noqa: BLE001
                log.error(f"  [{unit}/{var}] failed: "
                          f"{type(e).__name__}: {e}")
                results["maps"].append({"unit": unit, "var": var,
                                        "status": f"error: {e}"})

        if unit in ("hex250m", "barrios", "distritos"):
            out_path = FIG_DIR / f"lisa_{unit}_4panel.png"
            try:
                r = render_composite(unit, unit_gdf, moran_json,
                                      termino, out_path)
                results["composites"].append(r)
                if r.get("status") == "ok":
                    log.info(f"  saved {out_path.name}")
            except Exception as e:  # noqa: BLE001
                log.error(f"  composite {unit} failed: "
                          f"{type(e).__name__}: {e}")
                results["composites"].append({"unit": unit,
                                                "status": f"error: {e}"})

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "45_maps_lisa.json").write_text(
        dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '45_maps_lisa.json'}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(results), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    n_ok = sum(1 for m in results["maps"] if m.get("status") == "ok")
    n_comp = sum(1 for c in results["composites"]
                 if c.get("status") == "ok")
    log.info("=" * 72)
    log.info(f"Maps OK: {n_ok}/{len(results['maps'])}")
    log.info(f"Composites OK: {n_comp}/{len(results['composites'])}")


if __name__ == "__main__":
    main()