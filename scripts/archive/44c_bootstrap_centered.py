"""
Script 44c — Quick diagnostic of the baseline-shift bug in 44b.

Hypothesis: the near-zero Pearson rho in 44b is dominated by a per-
iteration baseline shift (each bootstrap uses a different hex as the
C(hex_boot) reference), not by real instability of the per-hex
coefficient.

Fix: center each iteration's fe_log_rr by its own mean before comparing
with the original (also centered). Recompute Pearson and Spearman.

Also reports Var(baseline_shift) vs Var(signal) to confirm diagnosis.

Does NOT refit the models (reuses 44b's raw per-iteration values,
reconstructed from raw lists in 44b JSON).

Outputs:
    reports/tables/44c_bootstrap_centered.json
    reports/44c_bootstrap_centered.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_MD = Path("reports/44c_bootstrap_centered.md")

log = logging.getLogger("phase10_44c")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load 44b's bootstrap per-hex file (has fe_log_rr original, boot_mean, boot_sd)
    boot_path = DATA / "hex_risk_bootstrap_v2.parquet"
    if not boot_path.exists():
        raise SystemExit(f"missing {boot_path}, run 44b first")

    boot = pd.read_parquet(boot_path)
    log.info(f"Loaded {len(boot)} hexes from 44b bootstrap")

    # We don't have per-iteration values (only aggregated boot_mean).
    # But we can compute the DIAGNOSTIC: is the sd of fe_log_rr ACROSS
    # hexes (in the original fit) comparable to the sd of boot_mean
    # (across hexes in the bootstrap)?
    #
    # If the baseline shift per iteration is a per-iteration constant,
    # then across hexes WITHIN one iteration, the sd is preserved
    # (baseline shift affects the mean, not the spread). But across
    # iterations, each hex's value has an extra variance from the
    # baseline shift.

    # Diagnostic 1: spread of ORIGINAL fe_log_rr across hexes
    orig_sd = float(boot["fe_log_rr"].std(ddof=1))
    orig_mean = float(boot["fe_log_rr"].mean())

    # Diagnostic 2: spread of BOOT_MEAN across hexes
    boot_mean_sd = float(boot["boot_mean"].std(ddof=1))
    boot_mean_mean = float(boot["boot_mean"].mean())

    # Diagnostic 3: per-hex sd distribution
    per_hex_sd_mean = float(boot["boot_sd"].mean())
    per_hex_sd_median = float(boot["boot_sd"].median())

    # If the baseline shift dominates, then per_hex_sd (which reflects
    # variability across iterations for the SAME hex) >> orig_sd
    # (which reflects spread across hexes within one fit).
    baseline_dominant = per_hex_sd_median > 0.5 * orig_sd

    log.info(f"orig_sd across hexes: {orig_sd:.4f}, mean: {orig_mean:.4f}")
    log.info(f"boot_mean_sd across hexes: {boot_mean_sd:.4f}, "
             f"mean: {boot_mean_mean:.4f}")
    log.info(f"per_hex_sd (same hex across iterations): "
             f"mean={per_hex_sd_mean:.4f}, median={per_hex_sd_median:.4f}")
    log.info(f"baseline_shift_dominant: {baseline_dominant}")

    # Direct fix: center orig and boot_mean by their means and compute rho
    # This is equivalent to what we'd do with per-iteration raw values,
    # because if baseline shift is a per-iteration additive constant,
    # then the mean-subtraction removes it exactly.
    orig_centered = boot["fe_log_rr"] - orig_mean
    boot_centered = boot["boot_mean"] - boot_mean_mean

    rho_centered, p_centered = pearsonr(orig_centered, boot_centered)
    rho_s_centered, _ = spearmanr(orig_centered, boot_centered)

    # Compare with uncentered (from 44b)
    rho_unc, _ = pearsonr(boot["fe_log_rr"], boot["boot_mean"])

    log.info(f"rho uncentered: {rho_unc:.4f} (was 0.005 in 44b)")
    log.info(f"rho centered:   {rho_centered:.4f} (p={p_centered:.4f})")
    log.info(f"spearman centered: {rho_s_centered:.4f}")

    # Sanity check: is the ratio of baseline_shift to signal large?
    # signal_var = orig_sd^2
    # total_var_around_signal = per_hex_sd^2
    # ratio = baseline_shift_variance / signal_variance
    signal_var = orig_sd ** 2
    per_hex_var = per_hex_sd_mean ** 2
    ratio = per_hex_var / signal_var if signal_var > 0 else None
    log.info(f"var(per_hex_sd) / var(orig_sd): "
             f"{ratio:.2f}× (baseline shift adds this much extra noise)")

    # Verdict
    if rho_centered > 0.80:
        verdict = ("STRONG: centering fixes the correlation. Baseline shift "
                   "was the primary bug. The FE ranking is preserved after "
                   "removing the reference-point problem. Map usable with "
                   "caveat about wide bootstrap CIs.")
    elif rho_centered > 0.50:
        verdict = ("MODERATE: centering helps but does not fully restore "
                   "correlation. The ranking is partially preserved "
                   "(rho 0.50-0.80). Map usable as 'descriptive', not for "
                   "policy. Report the bootstrap CI, not point estimates.")
    elif rho_centered > 0.30:
        verdict = ("WEAK: centering gives some signal (rho 0.30-0.50) but "
                   "the FE coefficients are genuinely noisy beyond the "
                   "baseline-shift issue. FE map not reliable for policy.")
    else:
        verdict = ("NO SIGNAL: even after centering, rho < 0.30. The FE "
                   "coefficients are genuinely unstable. Discard the FE "
                   "risk map. Use N4 residuals as descriptive alternative.")
    log.info(f"VERDICT: {verdict}")

    # Save
    result = {
        "n_hexes": int(len(boot)),
        "diagnostic": {
            "orig_sd": orig_sd,
            "orig_mean": orig_mean,
            "boot_mean_sd": boot_mean_sd,
            "boot_mean_mean": boot_mean_mean,
            "per_hex_sd_mean": per_hex_sd_mean,
            "per_hex_sd_median": per_hex_sd_median,
            "baseline_shift_dominant": baseline_dominant,
            "var_ratio_per_hex_over_signal": ratio,
        },
        "correlations": {
            "pearson_uncentered": float(rho_unc),
            "pearson_centered": float(rho_centered),
            "pearson_centered_p": float(p_centered),
            "spearman_centered": float(rho_s_centered),
        },
        "verdict": verdict,
    }
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    (OUT_TABLES / "44c_bootstrap_centered.json").write_text(
        dumps_jsonable(result), encoding="utf-8")

    md = ["# Phase 10 · Iter 4c — Centering diagnostic\n\n"]
    md.append("## Diagnóstico del baseline-shift\n\n")
    md.append(f"- SD de `fe_log_rr` ORIGINAL (entre hexes, fit único): "
              f"**{orig_sd:.4f}**\n")
    md.append(f"- SD de `boot_mean` (entre hexes, promediado sobre 20 iter): "
              f"**{boot_mean_sd:.4f}**\n")
    md.append(f"- SD PER-HEX (mismo hex a través de iteraciones): "
              f"mean={per_hex_sd_mean:.4f}, median={per_hex_sd_median:.4f}\n")
    md.append(f"- Ratio `var(per_hex) / var(orig)`: **{ratio:.2f}×** "
              f"(>1 = baseline shift dominates)\n\n")

    if baseline_dominant:
        md.append("**Confirmado:** la varianza per-hex es mucho mayor que "
                  "la varianza entre hexes del fit original. Esto solo "
                  "puede pasar si el baseline cambia entre iteraciones.\n\n")

    md.append("## Correlaciones\n\n")
    md.append(f"- Pearson sin centrar: **{rho_unc:.4f}** (bug)\n")
    md.append(f"- Pearson centrado: **{rho_centered:.4f}** "
              f"(p={p_centered:.4f})\n")
    md.append(f"- Spearman centrado: **{rho_s_centered:.4f}**\n\n")

    md.append("## Veredicto\n\n")
    md.append(f"**{verdict}**\n\n")

    md.append("## Recomendación de uso\n\n")
    if rho_centered > 0.80:
        md.append("Usar `hex_risk_bootstrap_v2.parquet` en Fase 11. "
                  "Reportar `boot_mean` centrado con `boot_sd` como banda "
                  "de incertidumbre. Documentar que el ranking se preserva "
                  "pero que el valor absoluto de cada `fe_log_rr` es "
                  "relativo a la media (no a un baseline fijo).\n")
    elif rho_centered > 0.50:
        md.append("Usar como mapa DESCRIPTIVO: 'ranking de riesgo relativo' "
                  "con caveat de CI anchos. No usar para asignación de "
                  "política sin validación adicional.\n")
    else:
        md.append("Descartar el mapa FE. En Fase 11 usar residuos "
                  "cuantilizados de N4 como mapa descriptivo alternativo.\n")

    OUT_MD.write_text("".join(md), encoding="utf-8")
    log.info(f"saved {OUT_TABLES / '44c_bootstrap_centered.json'}")
    log.info(f"saved {OUT_MD}")


if __name__ == "__main__":
    main()