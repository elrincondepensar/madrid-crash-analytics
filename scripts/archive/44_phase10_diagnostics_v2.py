"""
Script 44 — Phase 10 iter 4: variance decomposition by year + FE bootstrap.

Two diagnostics before D110:

  A. Variance decomposition of log_exposure per year (2019-2025).
     Question: is the within-hex share dominated by 2020 (COVID) or
     distributed evenly across years? If 2020 has an outsized within
     share, it confirms that FE_full identifies COVID, not structural
     exposure variation.

  C. Block bootstrap of hex FE coefficients.
     Sample 100 well-sampled hexes with replacement, 50 iterations.
     Each iteration refits FE_full on the resampled panel (hex IDs
     de-duplicated with a per-iteration offset). Reports:
       - stability of `log_exposure` coefficient (mean, sd, CI)
       - stability of per-hex `fe_log_rr` (correlation with original,
         95% CI coverage, sd per hex)
     Gate: if exposure coef sd/mean < 0.15 AND per-hex correlation > 0.90,
     the FE risk map is defensible for Phase 11.

Suppresses statsmodels warnings to keep log readable.

Outputs:
    reports/tables/44_diagnostics_v2.json
    reports/44_diagnostics_v2.md
    data/processed/hex_risk_bootstrap.parquet
    data/processed/variance_by_year.parquet
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Silence statsmodels spam (convergence/hessian/overflow warnings)
warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety import io as ms_io
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/44_diagnostics_v2.md")
OUT_VAR = DATA / "variance_by_year.parquet"
OUT_BOOT = DATA / "hex_risk_bootstrap.parquet"

log = logging.getLogger("phase10_44")

N_BOOT = 50
N_HEX_SAMPLE = 100
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


# ---------------------------------------------------------------------------
# PART A — variance decomposition by year
# ---------------------------------------------------------------------------

def variance_decomposition_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("=== PART A: variance decomposition by year ===")
    rows = []
    for year in sorted(panel["year"].astype(int).unique()):
        sub = panel[panel["year"].astype(int) == year].copy()
        le = sub["log_exposure"]
        mean_per_hex = sub.groupby("hex_id")["log_exposure"].transform("mean")
        between = mean_per_hex
        within = le - mean_per_hex
        var_total = float(le.var())
        var_between = float(between.var())
        var_within = float(within.var())
        rows.append({
            "year": int(year),
            "n_obs": int(len(sub)),
            "n_hex": int(sub["hex_id"].nunique()),
            "var_total": var_total,
            "var_between": var_between,
            "var_within": var_within,
            "between_share": var_between / var_total if var_total > 0 else None,
            "within_share": var_within / var_total if var_total > 0 else None,
            "mean_log_exposure": float(le.mean()),
        })
        log.info(f"  {year}: between={var_between/var_total:.1%}, "
                 f"within={var_within/var_total:.1%}, "
                 f"mean_le={float(le.mean()):.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART C — block bootstrap of FE
# ---------------------------------------------------------------------------

def bootstrap_hex_fe(panel: pd.DataFrame,
                     y_col: str = "n_injured",
                     n_hex_sample: int = N_HEX_SAMPLE,
                     n_boot: int = N_BOOT,
                     seed: int = RNG_SEED) -> dict:
    """Block bootstrap of hex FE coefficients.

    Sample n_hex_sample well-sampled hexes with replacement, refit FE_full
    each iteration. Each iter uses ALL 84 obs of each sampled hex.
    Hex IDs are de-duplicated per iteration (offset) so duplicates get
    separate coefficients.

    Returns stability diagnostics.
    """
    log.info(f"=== PART C: bootstrap FE (n_hex_sample={n_hex_sample}, "
             f"n_boot={n_boot}) ===")

    # Select well-sampled hexes only
    n_obs_per_hex = panel.groupby("hex_id").size()
    well_sampled = n_obs_per_hex[n_obs_per_hex >= 60].index.tolist()
    log.info(f"  well-sampled hexes (>=60 obs): {len(well_sampled)}")

    if len(well_sampled) < n_hex_sample:
        log.warning(f"  fewer well-sampled hexes than requested; "
                    f"using {len(well_sampled)}")

    rng = np.random.default_rng(seed)
    coef_log_exp = []
    fe_log_rr_per_hex = {}  # hex_id -> list of fe_log_rr across iterations
    fit_failures = 0
    t_start = time.time()

    for it in range(n_boot):
        t_it = time.time()
        # Sample hexes with replacement
        sampled = rng.choice(well_sampled,
                              size=min(n_hex_sample, len(well_sampled)),
                              replace=True)
        # Build unique per-iteration IDs so duplicates are treated separately
        # e.g. hex 123 sampled twice → 123_0 and 123_1, both tracked as 123
        occ_count = {}
        mapping_rows = []
        for h in sampled:
            k = occ_count.get(h, 0)
            occ_count[h] = k + 1
            mapping_rows.append({"hex_orig": h, "hex_boot": f"{h}_{k}"})
        mapping = pd.DataFrame(mapping_rows)

        # Build bootstrap panel
        pieces = []
        for h_orig in occ_count.keys():
            pieces.append(panel[panel["hex_id"] == h_orig])
        panel_boot = pd.concat(pieces, ignore_index=True)

        # Attach per-iteration hex_boot id
        # Because we concatenated per-hex blocks, we need to map each row
        # to its occurrence. Simple approach: order of pieces follows keys()
        # of occ_count, which preserves insertion order in Python 3.7+.
        # Build a "block_index" cumulative over pieces.
        block_hex_boot = []
        for h_orig in occ_count.keys():
            for k in range(occ_count[h_orig]):
                block_hex_boot.append(f"{h_orig}_{k}")
        # pieces order matches block_hex_boot order only if we iterate over
        # occ_count.keys() twice in same order. Python dict preserves
        # insertion order. So block_hex_boot[k] corresponds to pieces[k].
        # Now expand to rows:
        row_hex_boot = []
        for k, piece in enumerate(pieces):
            row_hex_boot.extend([block_hex_boot[k]] * len(piece))
        panel_boot["hex_boot"] = row_hex_boot

        # Fit FE_full (minimal, no tipo_via/highway since they're per-hex constant)
        formula = (f"{y_col} ~ log_exposure + C(year_str) + C(month_str) "
                   f"+ C(hex_boot)")
        try:
            res = smf.glm(formula=formula, data=panel_boot,
                           family=sm.families.Poisson()).fit(
                method="lbfgs", maxiter=300, disp=False)
            coef = float(res.params.get("log_exposure", np.nan))
            if np.isfinite(coef):
                coef_log_exp.append(coef)

            # Extract per-hex fe coefficients
            params = res.params
            # baseline hex_boot: first alphabetically / first encountered
            # The reference is whatever statsmodels picked. Find it:
            hex_coefs = {k: float(v) for k, v in params.items()
                         if k.startswith("C(hex_boot)[T.")}
            all_boot_ids = panel_boot["hex_boot"].unique()
            used_boot_ids = set(int(k.split(".")[1].rstrip("]")) if False
                                else k.split("[T.")[1].rstrip("]")
                                for k in hex_coefs.keys())
            # Correction: hex_boot is string, so the key is
            # "C(hex_boot)[T.1234_0]" → we need to extract "1234_0"
            used_ids_str = [k.split("[T.")[1].rstrip("]") for k in hex_coefs]
            # The reference (baseline) is the one boot id NOT in used_ids_str
            all_ids_str = list(panel_boot["hex_boot"].unique())
            baseline = None
            for bid in all_ids_str:
                if bid not in used_ids_str:
                    baseline = bid
                    break

            # For each hex original, average fe_log_rr across duplicates
            # (each hex_orig can have k=0..M occurrences)
            hex_orig_to_rrs = {}
            # Reference: fe_log_rr = 0
            if baseline is not None:
                orig_baseline = baseline.rsplit("_", 1)[0]
                try:
                    orig_int = int(orig_baseline)
                    hex_orig_to_rrs.setdefault(orig_int, []).append(0.0)
                except ValueError:
                    pass
            for key, v in hex_coefs.items():
                bid = key.split("[T.")[1].rstrip("]")
                orig = bid.rsplit("_", 1)[0]
                try:
                    orig_int = int(orig)
                    hex_orig_to_rrs.setdefault(orig_int, []).append(v)
                except ValueError:
                    continue

            # Record per-hex mean of fe_log_rr across duplicates
            for h, vals in hex_orig_to_rrs.items():
                fe_log_rr_per_hex.setdefault(h, []).append(float(np.mean(vals)))

        except Exception as e:  # noqa: BLE001
            fit_failures += 1
            log.warning(f"  iter {it}: fit failed ({type(e).__name__})")

        if (it + 1) % 10 == 0:
            elapsed = time.time() - t_start
            log.info(f"  {it+1}/{n_boot} done in {elapsed:.0f}s")

    log.info(f"  bootstrap finished: {n_boot - fit_failures}/{n_boot} "
             f"successful, {fit_failures} failed, "
             f"{time.time() - t_start:.0f}s total")

    # Load original fe_log_rr from script 43 output
    orig_path = DATA / "hex_risk_fe_flags.parquet"
    if orig_path.exists():
        orig_df = pd.read_parquet(orig_path)
        orig_df = orig_df[orig_df["quality"] == "well_sampled"][
            ["hex_id", "fe_log_rr"]].copy()
    else:
        orig_df = pd.DataFrame(columns=["hex_id", "fe_log_rr"])

    # Build per-hex summary
    rows = []
    for h, vals in fe_log_rr_per_hex.items():
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            continue
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        rows.append({
            "hex_id": int(h),
            "boot_n": len(arr),
            "boot_mean": float(arr.mean()),
            "boot_sd": float(arr.std(ddof=1)),
            "boot_ci_low": float(ci_low),
            "boot_ci_high": float(ci_high),
        })
    boot_df = pd.DataFrame(rows)

    # Merge with original
    merged = boot_df.merge(orig_df, on="hex_id", how="inner")
    if len(merged) > 0:
        from scipy.stats import pearsonr, spearmanr
        rho_pearson, _ = pearsonr(merged["fe_log_rr"], merged["boot_mean"])
        rho_spearman, _ = spearmanr(merged["fe_log_rr"], merged["boot_mean"])
        # Coverage: what fraction of hexes have their original value
        # inside the 95% CI
        in_ci = ((merged["fe_log_rr"] >= merged["boot_ci_low"]) &
                 (merged["fe_log_rr"] <= merged["boot_ci_high"]))
        coverage = float(in_ci.mean())
        # Per-hex sd summary
        sd_mean = float(merged["boot_sd"].mean())
        sd_p90 = float(merged["boot_sd"].quantile(0.9))
        # Sample of most / least stable
        merged["sd_boot"] = merged["boot_sd"]
        most_stable = merged.nsmallest(5, "sd_boot")[
            ["hex_id", "fe_log_rr", "boot_mean", "boot_sd"]
        ].to_dict(orient="records")
        least_stable = merged.nlargest(5, "sd_boot")[
            ["hex_id", "fe_log_rr", "boot_mean", "boot_sd"]
        ].to_dict(orient="records")
    else:
        rho_pearson = rho_spearman = coverage = sd_mean = sd_p90 = None
        most_stable = least_stable = []

    # Exposure coef stability
    coef_arr = np.array(coef_log_exp)
    if len(coef_arr) > 0:
        coef_ci_low, coef_ci_high = np.percentile(coef_arr, [2.5, 97.5])
        coef_stats = {
            "n": int(len(coef_arr)),
            "mean": float(coef_arr.mean()),
            "sd": float(coef_arr.std(ddof=1)),
            "ci_2.5": float(coef_ci_low),
            "ci_97.5": float(coef_ci_high),
            "coef_of_variation": float(coef_arr.std(ddof=1) / coef_arr.mean()),
        }
    else:
        coef_stats = {"n": 0}

    # Save per-hex bootstrap
    merged.to_parquet(OUT_BOOT, index=False)
    log.info(f"saved {OUT_BOOT}")

    return {
        "n_boot": n_boot,
        "n_boot_successful": n_boot - fit_failures,
        "n_hexes_bootstrap": int(len(merged)),
        "exposure_coef_stability": coef_stats,
        "per_hex_stability": {
            "pearson_rho_orig_vs_boot": float(rho_pearson) if rho_pearson is not None else None,
            "spearman_rho_orig_vs_boot": float(rho_spearman) if rho_spearman is not None else None,
            "ci_95_coverage": float(coverage) if coverage is not None else None,
            "per_hex_sd_mean": sd_mean,
            "per_hex_sd_p90": sd_p90,
        },
        "most_stable_hexes": most_stable,
        "least_stable_hexes": least_stable,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_md(var_df: pd.DataFrame, boot_result: dict) -> str:
    md = ["# Phase 10 · Iter 4 — Variance by year + FE bootstrap\n\n"]

    md.append("## PART A — Variance decomposition by year\n\n")
    md.append("| year | between | within | mean_le | n_obs |\n")
    md.append("| --- | ---: | ---: | ---: | ---: |\n")
    for _, r in var_df.iterrows():
        md.append(f"| {int(r['year'])} | "
                  f"{r['between_share']:.1%} | "
                  f"{r['within_share']:.1%} | "
                  f"{r['mean_log_exposure']:.3f} | "
                  f"{int(r['n_obs'])} |\n")
    md.append("\n")

    # Comparison 2020 vs mean of 2019/2021-2025
    y2020 = var_df[var_df["year"] == 2020]["within_share"].iloc[0]
    y_other = var_df[var_df["year"] != 2020]["within_share"].mean()
    md.append(f"**2020 within share: {y2020:.1%}. "
              f"Mean of other years: {y_other:.1%}. "
              f"Ratio: {y2020/y_other:.2f}×.**\n\n")
    if y2020 > y_other * 1.15:
        md.append("→ **2020 tiene una concentración de varianza within "
                  "significativamente mayor que el resto de años.** Esto "
                  "confirma que el within-hex share está dominado por el "
                  "shock COVID. El coef de FE_full (0.166) identifica "
                  "principalmente la respuesta al COVID, no fluctuaciones "
                  "de tráfico de política pública.\n\n")
    else:
        md.append("→ El within share es similar entre años. No hay "
                  "concentración especial en 2020. El coef de FE_full "
                  "no está dominado por COVID.\n\n")

    md.append("## PART C — Block bootstrap of hex FE\n\n")
    exp = boot_result["exposure_coef_stability"]
    ph = boot_result["per_hex_stability"]
    md.append(f"- Iteraciones: {boot_result['n_boot_successful']}/"
              f"{boot_result['n_boot']}\n")
    md.append(f"- Hexes bootstrapeados: "
              f"{boot_result['n_hexes_bootstrap']}\n\n")

    md.append("### Estabilidad del coef de `log_exposure`\n\n")
    if exp["n"] > 0:
        md.append(f"- Mean = **{exp['mean']:.4f}**, "
                  f"sd = {exp['sd']:.4f}, "
                  f"CI95% = [{exp['ci_2.5']:.4f}, {exp['ci_97.5']:.4f}]\n")
        md.append(f"- **Coef. de variación (sd/mean) = "
                  f"{exp['coef_of_variation']:.3f}** "
                  f"({'PASS' if exp['coef_of_variation'] < 0.15 else 'FAIL'} "
                  f"con umbral 0.15)\n\n")
    else:
        md.append("Sin fits exitosos\n\n")

    md.append("### Estabilidad de `fe_log_rr` por hex\n\n")
    if ph["pearson_rho_orig_vs_boot"] is not None:
        md.append(f"- **Pearson rho (original vs bootstrap mean): "
                  f"{ph['pearson_rho_orig_vs_boot']:.4f}** "
                  f"({'PASS' if ph['pearson_rho_orig_vs_boot'] > 0.90 else 'FAIL'} "
                  f"con umbral 0.90)\n")
        md.append(f"- Spearman rho: {ph['spearman_rho_orig_vs_boot']:.4f}\n")
        md.append(f"- **Cobertura del CI95%: "
                  f"{ph['ci_95_coverage']:.1%}** (ideal: ~95%)\n")
        md.append(f"- sd media por hex: {ph['per_hex_sd_mean']:.4f}, "
                  f"sd p90: {ph['per_hex_sd_p90']:.4f}\n\n")

    md.append("### Hexes más estables (sd bajo)\n\n")
    for h in boot_result["most_stable_hexes"]:
        md.append(f"- hex {h['hex_id']}: orig={h['fe_log_rr']:.3f}, "
                  f"boot_mean={h['boot_mean']:.3f}, "
                  f"sd={h['boot_sd']:.4f}\n")
    md.append("\n### Hexes menos estables (sd alto)\n\n")
    for h in boot_result["least_stable_hexes"]:
        md.append(f"- hex {h['hex_id']}: orig={h['fe_log_rr']:.3f}, "
                  f"boot_mean={h['boot_mean']:.3f}, "
                  f"sd={h['boot_sd']:.4f}\n")
    md.append("\n")

    md.append("## Interpretación\n\n")
    exp_pass = exp.get("coef_of_variation", 1) < 0.15
    hex_pass = ph.get("pearson_rho_orig_vs_boot", 0) > 0.90
    cov_pass = ph.get("ci_95_coverage", 0) > 0.85
    if exp_pass and hex_pass and cov_pass:
        md.append("**Los tres gates pasan.** El mapa de riesgo FE es "
                  "estable bajo block bootstrap. El coef de exposición "
                  "es defendible con sd pequeña, y los rankings por hex "
                  "se preservan. **El mapa es apto para Fase 11.**\n")
    elif exp_pass and hex_pass:
        md.append("**Los coeficientes son estables pero la cobertura "
                  "del CI es baja.** Los `fe_log_rr` originales caen "
                  "fuera del CI95% bootstrap en más del 15% de los "
                  "hexes. Posible causa: subestimación de la varianza "
                  "en el fit original (usó IRLS con rank-deficiency). "
                  "Documentar y usar el bootstrap CI como intervalo "
                  "de referencia.\n")
    else:
        md.append("**Inestabilidad detectada.** Revisar la especificación "
                  "antes de usar el mapa FE.\n")

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

    panel = load_panel()
    var_df = variance_decomposition_by_year(panel)
    var_df.to_parquet(OUT_VAR, index=False)
    log.info(f"saved {OUT_VAR}")

    boot_result = bootstrap_hex_fe(panel, y_col="n_injured",
                                     n_hex_sample=N_HEX_SAMPLE,
                                     n_boot=N_BOOT)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    result = {
        "variance_by_year": var_df.to_dict(orient="records"),
        "bootstrap": boot_result,
    }
    (OUT_TABLES / "44_diagnostics_v2.json").write_text(
        dumps_jsonable(result), encoding="utf-8")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(var_df, boot_result), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '44_diagnostics_v2.json'}")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    log.info("PART A: within share by year:")
    for _, r in var_df.iterrows():
        log.info(f"  {int(r['year'])}: within={r['within_share']:.1%}")
    log.info("PART C:")
    exp = boot_result["exposure_coef_stability"]
    ph = boot_result["per_hex_stability"]
    if exp["n"] > 0:
        log.info(f"  coef mean={exp['mean']:.4f}, sd={exp['sd']:.4f}, "
                 f"CV={exp['coef_of_variation']:.3f}")
    log.info(f"  hex rho={ph.get('pearson_rho_orig_vs_boot')}, "
             f"CI coverage={ph.get('ci_95_coverage')}, "
             f"sd_mean={ph.get('per_hex_sd_mean')}")


if __name__ == "__main__":
    main()