"""
Script 44b — Clean block bootstrap of FE with 400 hexes × 20 iterations.

Fixes the design bug in 44:
  - 44 sampled 100 hexes from 876 well-sampled → each hex appears ~5.7
    times on average across 50 iterations → only 23 survived the >=10
    filter, and those were biased toward lucky draws.
  - 44b samples 400 hexes from 876 → each appears ~9.1 times on average
    across 20 iterations → most hexes survive the >=5 filter.

If per-hex fe_log_rr sd remains large (>0.5) with this design, the
instability is real (rank-deficiency effect), not a sampling artifact.
If sd drops below 0.15, the FE risk map is defensible.

Computational cost: ~6-9 min (20 fits of ~400 dummies via lbfgs).

Outputs:
    reports/tables/44b_bootstrap_clean.json
    reports/44b_bootstrap_clean.md
    data/processed/hex_risk_bootstrap_v2.parquet
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/44b_bootstrap_clean.md")
OUT_BOOT = DATA / "hex_risk_bootstrap_v2.parquet"

log = logging.getLogger("phase10_44b")

N_BOOT = 20
N_HEX_SAMPLE = 400
MIN_BOOT_OCCURRENCES = 5
RNG_SEED = 42


def load_panel() -> pd.DataFrame:
    log.info("Loading panel ...")
    panel = pd.read_parquet(DATA / "panel_hex_month_n_injured.parquet")
    panel["year_str"] = panel["year"].astype(int).astype(str)
    panel["month_str"] = panel["month"].astype(int).astype(str).str.zfill(2)
    panel["log_exposure"] = np.log(panel["exposure_month"] + 1e-10)
    panel["hex_id"] = panel["hex_id"].astype(int)
    panel = panel.dropna(subset=["log_exposure"]).copy()
    log.info(f"  {len(panel):,} rows, {panel['hex_id'].nunique()} hexes")
    return panel


def bootstrap_fe(panel: pd.DataFrame, y_col: str = "n_injured",
                 n_hex_sample: int = N_HEX_SAMPLE,
                 n_boot: int = N_BOOT,
                 min_occ: int = MIN_BOOT_OCCURRENCES,
                 seed: int = RNG_SEED) -> dict:
    log.info(f"=== Bootstrap FE: n_hex_sample={n_hex_sample}, "
             f"n_boot={n_boot}, min_occ={min_occ} ===")

    n_obs_per_hex = panel.groupby("hex_id").size()
    well_sampled = n_obs_per_hex[n_obs_per_hex >= 60].index.tolist()
    log.info(f"  well-sampled hexes (>=60 obs): {len(well_sampled)}")

    rng = np.random.default_rng(seed)
    coef_log_exp = []
    fe_log_rr_per_hex = {}  # hex_orig -> list of fe_log_rr per iter
    fit_failures = 0
    t_start = time.time()

    for it in range(n_boot):
        t_it = time.time()
        sampled = rng.choice(well_sampled,
                              size=min(n_hex_sample, len(well_sampled)),
                              replace=True)

        # Build per-occurrence boot IDs
        occ_count = {}
        for h in sampled:
            k = occ_count.get(h, 0)
            occ_count[h] = k + 1
        ordered_hexes = list(occ_count.keys())
        block_ids = []
        for h in ordered_hexes:
            for k in range(occ_count[h]):
                block_ids.append(f"{h}_{k}")

        # Concatenate per-hex blocks; expand boot id to rows
        pieces = [panel[panel["hex_id"] == h] for h in ordered_hexes]
        row_boot_ids = []
        for bid, piece in zip(block_ids, pieces):
            row_boot_ids.extend([bid] * len(piece))
        panel_boot = pd.concat(pieces, ignore_index=True)
        panel_boot["hex_boot"] = row_boot_ids

        formula = (f"{y_col} ~ log_exposure + C(year_str) + C(month_str) "
                   f"+ C(hex_boot)")
        try:
            res = smf.glm(formula=formula, data=panel_boot,
                           family=sm.families.Poisson()).fit(
                method="lbfgs", maxiter=300, disp=False)
            coef = float(res.params.get("log_exposure", np.nan))
            if np.isfinite(coef):
                coef_log_exp.append(coef)

            params = res.params
            hex_coefs = {k: float(v) for k, v in params.items()
                         if k.startswith("C(hex_boot)[T.")}
            used_ids = set(k.split("[T.")[1].rstrip("]")
                           for k in hex_coefs.keys())
            baseline = next((bid for bid in block_ids
                             if bid not in used_ids), None)

            # Aggregate per-hex across occurrences
            per_hex_vals = {}
            if baseline is not None:
                orig_baseline = baseline.rsplit("_", 1)[0]
                try:
                    per_hex_vals.setdefault(int(orig_baseline), []).append(0.0)
                except ValueError:
                    pass
            for key, v in hex_coefs.items():
                bid = key.split("[T.")[1].rstrip("]")
                orig = bid.rsplit("_", 1)[0]
                try:
                    per_hex_vals.setdefault(int(orig), []).append(v)
                except ValueError:
                    continue

            for h, vals in per_hex_vals.items():
                fe_log_rr_per_hex.setdefault(h, []).append(
                    float(np.mean(vals)))
        except Exception as e:  # noqa: BLE001
            fit_failures += 1
            log.warning(f"  iter {it}: {type(e).__name__}: {e}")

        if (it + 1) % 5 == 0:
            log.info(f"  {it+1}/{n_boot}  "
                     f"elapsed={time.time() - t_start:.0f}s  "
                     f"last_iter={time.time() - t_it:.1f}s")

    n_success = n_boot - fit_failures
    log.info(f"  finished: {n_success}/{n_boot} successful, {fit_failures} failed, "
             f"{time.time() - t_start:.0f}s")

    # Per-hex summary (relaxed filter)
    rows = []
    for h, vals in fe_log_rr_per_hex.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_occ:
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({
            "hex_id": int(h),
            "boot_n": int(len(arr)),
            "boot_mean": float(arr.mean()),
            "boot_sd": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
            "boot_ci_low": float(lo),
            "boot_ci_high": float(hi),
        })
    boot_df = pd.DataFrame(rows)
    log.info(f"  hexes surviving min_occ={min_occ}: {len(boot_df)}")

    # Merge with 43's original fe_log_rr
    orig_path = DATA / "hex_risk_fe_flags.parquet"
    if orig_path.exists():
        orig_df = pd.read_parquet(orig_path)
        orig_df = orig_df[orig_df["quality"] == "well_sampled"][
            ["hex_id", "fe_log_rr"]].copy()
    else:
        orig_df = pd.DataFrame(columns=["hex_id", "fe_log_rr"])

    merged = boot_df.merge(orig_df, on="hex_id", how="inner")
    log.info(f"  overlap with original: {len(merged)} hexes")

    from scipy.stats import pearsonr, spearmanr
    if len(merged) >= 10:
        rho_p, _ = pearsonr(merged["fe_log_rr"], merged["boot_mean"])
        rho_s, _ = spearmanr(merged["fe_log_rr"], merged["boot_mean"])
        in_ci = ((merged["fe_log_rr"] >= merged["boot_ci_low"]) &
                 (merged["fe_log_rr"] <= merged["boot_ci_high"]))
        coverage = float(in_ci.mean())
        sd_mean = float(merged["boot_sd"].mean())
        sd_median = float(merged["boot_sd"].median())
        sd_p90 = float(merged["boot_sd"].quantile(0.9))
        most_stable = merged.nsmallest(5, "boot_sd")[
            ["hex_id", "fe_log_rr", "boot_mean", "boot_sd"]
        ].to_dict(orient="records")
        least_stable = merged.nlargest(5, "boot_sd")[
            ["hex_id", "fe_log_rr", "boot_mean", "boot_sd"]
        ].to_dict(orient="records")
    else:
        rho_p = rho_s = coverage = sd_mean = sd_median = sd_p90 = None
        most_stable = least_stable = []

    # Exposure coefficient stability
    coefs_arr = np.array(coef_log_exp, dtype=float)
    if len(coefs_arr) > 1:
        cl, ch = np.percentile(coefs_arr, [2.5, 97.5])
        exp_stats = {
            "n": int(len(coefs_arr)),
            "mean": float(coefs_arr.mean()),
            "sd": float(coefs_arr.std(ddof=1)),
            "ci_low": float(cl),
            "ci_high": float(ch),
            "cv": float(coefs_arr.std(ddof=1) / coefs_arr.mean())
                  if coefs_arr.mean() != 0 else None,
            "raw": coefs_arr.tolist(),
        }
    else:
        exp_stats = {"n": len(coefs_arr)}

    merged.to_parquet(OUT_BOOT, index=False)
    log.info(f"saved {OUT_BOOT}")

    return {
        "n_boot": n_boot,
        "n_boot_successful": n_success,
        "n_hex_sample": n_hex_sample,
        "min_occ": min_occ,
        "n_hexes_bootstrap": int(len(boot_df)),
        "n_hexes_overlap_with_orig": int(len(merged)),
        "exposure_coef": exp_stats,
        "per_hex_stability": {
            "pearson_rho": float(rho_p) if rho_p is not None else None,
            "spearman_rho": float(rho_s) if rho_s is not None else None,
            "ci_coverage": coverage,
            "sd_mean": sd_mean,
            "sd_median": sd_median,
            "sd_p90": sd_p90,
        },
        "most_stable_hexes": most_stable,
        "least_stable_hexes": least_stable,
    }


def render_md(res: dict) -> str:
    md = ["# Phase 10 · Iter 4b — Bootstrap limpio\n\n"]
    md.append(
        f"**Diseño:** {res['n_boot']} iteraciones, "
        f"{res['n_hex_sample']} hexes well-sampled por iteración, "
        f"filtro mínimo {res['min_occ']} apariciones.\n\n"
    )
    md.append(f"- Bootstrap exitosos: "
              f"{res['n_boot_successful']}/{res['n_boot']}\n")
    md.append(f"- Hexes bootstrapeados: {res['n_hexes_bootstrap']} "
              f"(vs 23 en iter 4)\n")
    md.append(f"- Hexes con overlap al original: "
              f"{res['n_hexes_overlap_with_orig']}\n\n")

    exp = res["exposure_coef"]
    md.append("## Coeficiente de `log_exposure`\n\n")
    if exp.get("n", 0) > 1:
        md.append(f"- Mean = **{exp['mean']:.4f}**, "
                  f"sd = {exp['sd']:.4f}\n")
        md.append(f"- CI95% = [{exp['ci_low']:.4f}, "
                  f"{exp['ci_high']:.4f}]\n")
        md.append(f"- CV (sd/mean) = **{exp['cv']:.3f}** "
                  f"({'PASS' if exp['cv'] < 0.15 else 'FAIL'} "
                  f"con umbral 0.15)\n\n")

    ph = res["per_hex_stability"]
    md.append("## Estabilidad de `fe_log_rr` por hex\n\n")
    if ph["pearson_rho"] is not None:
        md.append(f"- **Pearson rho (orig vs bootstrap): "
                  f"{ph['pearson_rho']:.4f}** "
                  f"({'PASS' if ph['pearson_rho'] > 0.90 else 'FAIL'} "
                  f"con umbral 0.90)\n")
        md.append(f"- Spearman rho: {ph['spearman_rho']:.4f}\n")
        md.append(f"- Cobertura CI95%: {ph['ci_coverage']:.1%}\n")
        md.append(f"- sd media por hex: {ph['sd_mean']:.4f}\n")
        md.append(f"- sd mediana: {ph['sd_median']:.4f}\n")
        md.append(f"- sd p90: {ph['sd_p90']:.4f}\n\n")

    md.append("### Hexes más estables\n\n")
    for h in res["most_stable_hexes"]:
        md.append(f"- hex {h['hex_id']}: orig={h['fe_log_rr']:.3f}, "
                  f"boot_mean={h['boot_mean']:.3f}, "
                  f"sd={h['boot_sd']:.4f}\n")
    md.append("\n### Hexes menos estables\n\n")
    for h in res["least_stable_hexes"]:
        md.append(f"- hex {h['hex_id']}: orig={h['fe_log_rr']:.3f}, "
                  f"boot_mean={h['boot_mean']:.3f}, "
                  f"sd={h['boot_sd']:.4f}\n")
    md.append("\n")

    md.append("## Interpretación\n\n")
    exp_ok = (exp.get("cv") or 1) < 0.15
    rho_ok = (ph.get("pearson_rho") or 0) > 0.90
    sd_ok = (ph.get("sd_median") or 1) < 0.30
    if exp_ok and rho_ok and sd_ok:
        md.append("**Los tres gates pasan con este diseño.** El mapa "
                  "FE es estable y defendible para Fase 11.\n")
    elif exp_ok and rho_ok:
        md.append("**Coeficientes ordenados pero ruidosos.** El ranking "
                  "se preserva (rho > 0.90) pero los sd por hex son "
                  "grandes. El mapa es apto como herramienta comparativa "
                  "(ranking) pero los intervalos son anchos.\n")
    elif exp_ok:
        md.append("**El coef de exposición es estable pero el ranking "
                  "por hex no se preserva.** Los `fe_log_rr` individuales "
                  "tienen demasiada varianza para usarse como mapa de "
                  "riesgo. **No usar para política.** Usar N4 residuals "
                  "como mapa descriptivo alternativo (menos informativo "
                  "pero honesto).\n")
    else:
        md.append("**Inestabilidad real confirmada.** Incluso con 400 "
                  "hexes por iteración, el FE no converge a un ranking "
                  "estable. Diagnóstico: colinealidad tipo_via↔hex y "
                  "year↔exposure hacen el problema mal identificado. "
                  "**Descartar mapa FE.**\n")

    return "".join(md)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    panel = load_panel()
    res = bootstrap_fe(panel, y_col="n_injured")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "44b_bootstrap_clean.json").write_text(
        dumps_jsonable(res), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(res), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '44b_bootstrap_clean.json'}")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    exp = res["exposure_coef"]
    ph = res["per_hex_stability"]
    if exp.get("n", 0) > 1:
        log.info(f"coef mean={exp['mean']:.4f}, sd={exp['sd']:.4f}, "
                 f"CV={exp['cv']:.3f}")
    log.info(f"hex rho={ph.get('pearson_rho')}, "
             f"sd_median={ph.get('sd_median')}, "
             f"CI coverage={ph.get('ci_coverage')}")


if __name__ == "__main__":
    main()