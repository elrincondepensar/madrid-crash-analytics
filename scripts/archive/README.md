# Archived scripts

These scripts are kept for reference only. They either have known bugs or have been superseded by later iterations.

| script | reason |
| --- | --- |
| `41b_poisson_diagnostics.py` | Bug de fittedvalues (devuelve X·β, no exp(X·β)). Reemplazado por 41c/41d. |
| `41c_phase10_refit.py` | Refit NB2. Superado por 41d (residuales corregidos). |
| `44_phase10_diagnostics_v2.py` | Bootstrap FE con solo 23 hexes (diseño roto). Superado por 44b. |
| `44b_phase10_bootstrap_clean.py` | Bootstrap limpio (400 hexes × 20 iter) pero sin corregir baseline-shift. |
| `44c_bootstrap_centered.py` | Diagnóstico final: mapa FE inestable, descartado en D110. |
