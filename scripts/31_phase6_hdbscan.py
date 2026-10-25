"""
Phase 6 · Iteration 1 — HDBSCAN sobre accidentes de Madrid (2019-2022 para fit).

Objetivo:
  - Grid search de (min_cluster_size, min_samples) sobre train 2019-2022.
  - Selección de parámetros con criterio documentado (no forzado a 63 clusters).
  - Caracterización de clusters: tamaño, radio, distribución de severidad,
    distribución geográfica por distrito.
  - Doble ejecución: todos los accidentes y solo injured.

Entrada:
  data/processed/accidents_clean.parquet
  data/processed/splits.parquet

Salidas:
  data/processed/hdbscan_clusters_all.parquet    (una fila por accidente con
                                                   cluster label del modelo all)
  data/processed/hdbscan_clusters_injured.parquet
  reports/tables/31_hdbscan_grid.json
  reports/tables/31_hdbscan_clusters.json
  reports/31_hdbscan.md

Run:
    pip install hdbscan (o usar sklearn >= 1.3)
    python scripts/31_phase6_hdbscan.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

warnings.filterwarnings("ignore")

OUT_DATA = Path("data/processed")
OUT_TABLES = Path("reports/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Filtro espacial de Madrid (UTM 30N, conservador)
UTM_X_MIN, UTM_X_MAX = 425_000, 460_000
UTM_Y_MIN, UTM_Y_MAX = 4_460_000, 4_490_000

# Grid de hiperparámetros
MCS_GRID = [15, 30, 50, 100, 150, 200]
MS_GRID = [5, 10, 15, 20, 30]


# ===========================================================================
# Data loading
# ===========================================================================
def load_train(only_injured: bool = False) -> pd.DataFrame:
    """Carga train con coords válidas y dentro de Madrid UTM."""
    acc = pd.read_parquet(OUT_DATA / "accidents_clean.parquet",
                           columns=["num_expediente", "year", "fecha",
                                    "severity", "severity_code",
                                    "coordenada_x_utm", "coordenada_y_utm",
                                    "distrito", "tipo_accidente"])
    splits = pd.read_parquet(OUT_DATA / "splits.parquet",
                              columns=["num_expediente", "split"])
    df = acc.merge(splits, on="num_expediente", how="left")
    df = df.loc[df["split"] == "train"].copy()
    df = df.loc[df["coordenada_x_utm"].notna()
                & df["coordenada_y_utm"].notna()].copy()
    df = df.loc[
        df["coordenada_x_utm"].between(UTM_X_MIN, UTM_X_MAX)
        & df["coordenada_y_utm"].between(UTM_Y_MIN, UTM_Y_MAX)
    ].copy()
    if only_injured:
        df = df.loc[df["severity_code"] >= 1].copy()
    return df.reset_index(drop=True)


# ===========================================================================
# HDBSCAN utilities
# ===========================================================================
def fit_hdbscan(X: np.ndarray, mcs: int, ms: int) -> np.ndarray:
    m = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        allow_single_cluster=False,
    )
    return m.fit_predict(X)


def cluster_diagnostics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Métricas básicas de la clusterización."""
    n = len(labels)
    n_noise = int((labels == -1).sum())
    unique = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(unique)
    if n_clusters == 0:
        return {
            "n_clusters": 0, "noise_share": n_noise / n if n else 0,
            "cluster_sizes": [], "median_cluster_size": None,
            "median_radius_m": None, "largest_cluster_size": None,
            "smallest_cluster_size": None,
        }
    sizes = [int((labels == c).sum()) for c in unique]
    # Radio: distancia máx al centroide de cada cluster
    radii = []
    for c in unique:
        pts = X[labels == c]
        if len(pts) < 2:
            continue
        centroid = pts.mean(axis=0, keepdims=True)
        d = pairwise_distances(pts, centroid).ravel()
        radii.append(float(d.max()))
    return {
        "n_clusters": n_clusters,
        "noise_share": round(n_noise / n, 4),
        "cluster_sizes": sizes,
        "median_cluster_size": float(np.median(sizes)),
        "smallest_cluster_size": int(np.min(sizes)),
        "largest_cluster_size": int(np.max(sizes)),
        "median_radius_m": (round(float(np.median(radii)), 1)
                            if radii else None),
        "p90_radius_m": (round(float(np.percentile(radii, 90)), 1)
                         if radii else None),
    }


# ===========================================================================
# Grid search
# ===========================================================================
def run_grid(X: np.ndarray, label: str) -> dict:
    """Grid search sobre (mcs, ms). Devuelve resultados por combo."""
    print(f"\n=== Grid search HDBSCAN — {label} ({len(X)} points) ===")
    results = {}
    for mcs in MCS_GRID:
        for ms in MS_GRID:
            key = f"mcs={mcs}_ms={ms}"
            print(f"  fitting {key}...", end="", flush=True)
            t0 = time.time()
            labels = fit_hdbscan(X, mcs=mcs, ms=ms)
            diag = cluster_diagnostics(X, labels)
            diag["fit_seconds"] = round(time.time() - t0, 1)
            results[key] = diag
            print(f" {diag['n_clusters']} clusters, "
                  f"noise={diag['noise_share']:.3f}, "
                  f"median_size={diag['median_cluster_size']:.0f}, "
                  f"in {diag['fit_seconds']:.0f}s", flush=True)
    return results


def select_reference(grid_results: dict) -> str:
    """
    Selecciona la combinación de referencia.
    Criterios (documentados):
      1. n_clusters ∈ [40, 150]
      2. noise_share ∈ [0.20, 0.50]
      3. median_radius_m ∈ [50, 300]
      4. Estabilidad: preferir el combo con parámetros "centrales" del grid
         (ms=10, mcs=50 o mcs=100)

    Filtros de sanidad (nuevos, excluyen soluciones degeneradas):
      A. n_clusters >= 5  → descarta colapso a 2-3 blobs.
      B. largest_cluster <= 50% del total de puntos → descarta el caso
         "un único cluster gigante absorbe toda la ciudad".
    """
    candidates = []
    excluded = []
    for key, d in grid_results.items():
        # --- Filtros de sanidad ---
        n_clusters = d.get("n_clusters", 0)
        if n_clusters < 5:
            excluded.append({"key": key, "reason": f"n_clusters={n_clusters}<5"})
            continue
        sizes = d.get("cluster_sizes") or []
        largest = d.get("largest_cluster_size", 0) or 0
        noise = d.get("noise_share", 0.0) or 0.0
        # total estimado = sum(sizes) / (1 - noise)
        s = int(sum(sizes))
        total_est = s / max(1e-9, 1.0 - noise) if s else 0
        if total_est > 0 and largest > 0.5 * total_est:
            excluded.append({
                "key": key,
                "reason": f"largest={largest} > 50% total≈{int(total_est)}",
            })
            continue

        # --- Criterios de calidad ---
        ok_n = 40 <= n_clusters <= 150
        ok_noise = 0.20 <= noise <= 0.50
        radius = d.get("median_radius_m")
        ok_radius = radius is not None and 50 <= radius <= 300
        mcs, ms = [int(p.split("=")[1]) for p in key.split("_")]
        centrality = -(abs(np.log(mcs) - np.log(100))
                        + abs(np.log(ms) - np.log(10)))
        candidates.append({
            "key": key,
            "ok_n": ok_n, "ok_noise": ok_noise, "ok_radius": ok_radius,
            "score": int(ok_n) + int(ok_noise) + int(ok_radius),
            "centrality": centrality,
        })

    if excluded:
        print(f"  Excluded {len(excluded)} degenerate combos "
              f"(e.g. {excluded[0]['key']}: {excluded[0]['reason']})")

    if not candidates:
        print("  WARNING: no candidate survived sanity filters")
        return list(grid_results)[0]

    # Prioridad: cumplir los 3 criterios, luego centralidad del grid
    candidates.sort(key=lambda c: (c["score"], c["centrality"]),
                     reverse=True)
    selected = candidates[0]
    print(f"\n  Reference selected: {selected['key']} "
          f"(score={selected['score']}/3)")
    if selected["score"] < 3:
        print(f"  WARNING: only {selected['score']}/3 criteria met. "
              f"Consider relajando el criterio que falla:")
        if not selected["ok_n"]:
            print("    - n_clusters fuera de [40, 150]")
        if not selected["ok_noise"]:
            print("    - noise fuera de [0.20, 0.50]")
        if not selected["ok_radius"]:
            print("    - median_radius fuera de [50, 300] m")
    return selected["key"]


# ===========================================================================
# Caracterización del clustering
# ===========================================================================
def characterize_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Distribución de severidad, tipo_accidente y distrito por cluster."""
    df = df.copy()
    df["cluster"] = labels
    # Descriptivos por cluster (excluye noise)
    cl = df.loc[df["cluster"] != -1]
    if len(cl) == 0:
        return {"n_clusters": 0, "top_clusters": []}
    summary = (cl.groupby("cluster")
                 .agg(
                     n_accidents=("num_expediente", "size"),
                     n_injured=("severity_code", lambda s: int((s >= 1).sum())),
                     n_grave=("severity_code", lambda s: int((s >= 2).sum())),
                     n_fatal=("severity_code", lambda s: int((s == 3).sum())),
                     mean_x=("coordenada_x_utm", "mean"),
                     mean_y=("coordenada_y_utm", "mean"),
                     dominant_district=("distrito",
                                          lambda s: s.mode().iloc[0]
                                          if not s.mode().empty else None),
                     dominant_accident=("tipo_accidente",
                                         lambda s: s.mode().iloc[0]
                                         if not s.mode().empty else None),
                 )
                 .reset_index())
    summary["injury_rate"] = (summary["n_injured"] /
                               summary["n_accidents"]).round(4)
    summary["grave_rate"] = (summary["n_grave"] /
                              summary["n_accidents"]).round(4)
    # Top 10 por tamaño
    top = summary.sort_values("n_accidents", ascending=False).head(10)
    # Distribución de tamaño
    return {
        "n_clusters": int(summary["cluster"].nunique()),
        "total_in_clusters": int(summary["n_accidents"].sum()),
        "top10_by_size": top.to_dict(orient="records"),
        "global_injury_rate_in_clusters": round(float(
            summary["n_injured"].sum() / summary["n_accidents"].sum()), 4),
        "global_grave_rate_in_clusters": round(float(
            summary["n_grave"].sum() / summary["n_accidents"].sum()), 4),
        "cluster_size_quantiles": {
            "min": int(summary["n_accidents"].min()),
            "q25": float(summary["n_accidents"].quantile(0.25)),
            "median": float(summary["n_accidents"].median()),
            "q75": float(summary["n_accidents"].quantile(0.75)),
            "max": int(summary["n_accidents"].max()),
        },
        "injury_rate_stats": {
            "min": float(summary["injury_rate"].min()),
            "median": float(summary["injury_rate"].median()),
            "max": float(summary["injury_rate"].max()),
        },
    }


# ===========================================================================
# Markdown
# ===========================================================================
def _md(df_):
    h = "| " + " | ".join(str(c) for c in df_.columns) + " |"
    s = "| " + " | ".join("---" for _ in df_.columns) + " |"
    rs = ["| " + " | ".join(str(v) for v in r) + " |"
          for r in df_.itertuples(index=False)]
    return "\n".join([h, s] + rs)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading train accidents (all)...")
    df_all = load_train(only_injured=False)
    print(f"  all: {len(df_all):,} accidents in train (2019-2022), "
          f"coords valid in Madrid.")
    X_all = df_all[["coordenada_x_utm", "coordenada_y_utm"]].to_numpy()

    print("\nLoading train accidents (injured only)...")
    df_inj = load_train(only_injured=True)
    print(f"  injured: {len(df_inj):,} accidents.")
    X_inj = df_inj[["coordenada_x_utm", "coordenada_y_utm"]].to_numpy()

    # ---------------- Grid + selección ----------------
    grid_all = run_grid(X_all, "all accidents")
    grid_inj = run_grid(X_inj, "injured only")

    ref_all = select_reference(grid_all)
    ref_inj = select_reference(grid_inj)
    print(f"\n  Reference for 'all': {ref_all}")
    print(f"  Reference for 'injured': {ref_inj}")

    # ---------------- Fit final con referencia ----------------
    def _parse(key):
        d = {}
        for p in key.split("_"):
            k, v = p.split("=")
            d[k] = int(v)
        return d

    p_all = _parse(ref_all)
    labels_all = fit_hdbscan(X_all, mcs=p_all["mcs"], ms=p_all["ms"])
    labels_inj = fit_hdbscan(X_inj, mcs=_parse(ref_inj)["mcs"],
                              ms=_parse(ref_inj)["ms"])

    # ---------------- Caracterización ----------------
    char_all = characterize_clusters(df_all, labels_all)
    char_inj = characterize_clusters(df_inj, labels_inj)

    # ---------------- Guardar ----------------
    df_all_out = df_all.copy()
    df_all_out["cluster"] = labels_all
    df_all_out[["num_expediente", "year", "severity", "severity_code",
                "coordenada_x_utm", "coordenada_y_utm",
                "distrito", "tipo_accidente", "cluster"]] \
        .to_parquet(OUT_DATA / "hdbscan_clusters_all.parquet", index=False)

    df_inj_out = df_inj.copy()
    df_inj_out["cluster"] = labels_inj
    df_inj_out[["num_expediente", "year", "severity", "severity_code",
                "coordenada_x_utm", "coordenada_y_utm",
                "distrito", "tipo_accidente", "cluster"]] \
        .to_parquet(OUT_DATA / "hdbscan_clusters_injured.parquet", index=False)

    # ---------------- Reportes ----------------
    grid_summary_all = pd.DataFrame([
        {"combo": k,
         "n_clusters": v["n_clusters"],
         "noise_share": v["noise_share"],
         "median_cluster_size": v["median_cluster_size"],
         "median_radius_m": v["median_radius_m"],
         "largest_cluster": v["largest_cluster_size"],
         "fit_seconds": v["fit_seconds"]}
        for k, v in grid_all.items()
    ])
    grid_summary_inj = pd.DataFrame([
        {"combo": k,
         "n_clusters": v["n_clusters"],
         "noise_share": v["noise_share"],
         "median_cluster_size": v["median_cluster_size"],
         "median_radius_m": v["median_radius_m"],
         "largest_cluster": v["largest_cluster_size"],
         "fit_seconds": v["fit_seconds"]}
        for k, v in grid_inj.items()
    ])

    top_all_df = pd.DataFrame(char_all["top10_by_size"])
    top_inj_df = pd.DataFrame(char_inj["top10_by_size"])

    results = {
        "grid_all": grid_all,
        "grid_injured": grid_inj,
        "reference_all": ref_all,
        "reference_injured": ref_inj,
        "characterization_all": char_all,
        "characterization_injured": char_inj,
    }
    save_json(results, OUT_TABLES / "31_hdbscan_clusters.json")
    save_json({k: v for k, v in grid_all.items()},
               OUT_TABLES / "31_hdbscan_grid_all.json")
    save_json({k: v for k, v in grid_inj.items()},
               OUT_TABLES / "31_hdbscan_grid_injured.json")

    sections = {
        "Grid search — ALL accidents": _md(grid_summary_all),
        "Grid search — INJURED accidents": _md(grid_summary_inj),
        "Reference all": (
            f"**{ref_all}**\n\n"
            f"- n_clusters: {grid_all[ref_all]['n_clusters']}\n"
            f"- noise_share: {grid_all[ref_all]['noise_share']:.3f}\n"
            f"- median_cluster_size: {grid_all[ref_all]['median_cluster_size']:.0f}\n"
            f"- median_radius_m: {grid_all[ref_all]['median_radius_m']} m\n"
            f"- total in clusters: {char_all['total_in_clusters']:,}\n"
            f"- injury_rate in clusters: "
            f"{char_all['global_injury_rate_in_clusters']:.4f}\n"
            f"- grave_rate in clusters: "
            f"{char_all['global_grave_rate_in_clusters']:.4f}"
        ),
        "Reference injured": (
            f"**{ref_inj}**\n\n"
            f"- n_clusters: {grid_inj[ref_inj]['n_clusters']}\n"
            f"- noise_share: {grid_inj[ref_inj]['noise_share']:.3f}\n"
            f"- median_cluster_size: "
            f"{grid_inj[ref_inj]['median_cluster_size']:.0f}\n"
            f"- median_radius_m: {grid_inj[ref_inj]['median_radius_m']} m\n"
            f"- total in clusters: {char_inj['total_in_clusters']:,}"
        ),
        "Top 10 clusters — all": _md(top_all_df),
        "Top 10 clusters — injured": _md(top_inj_df),
        "Cluster size distribution — all": dumps_jsonable(
            char_all["cluster_size_quantiles"], indent=2),
        "Injury rate per cluster (all)": dumps_jsonable(
            char_all["injury_rate_stats"], indent=2),
        "Notes": (
            "HDBSCAN sobre coordenadas UTM (EPSG:25830), sin normalizar.\n"
            "Parámetros seleccionados SOLO con train 2019-2022 (sin fuga).\n"
            "Criterio de selección: n_clusters ∈ [40,150], "
            "noise ∈ [0.2,0.5], median_radius ∈ [50,300] m, "
            "centralidad del grid.\n"
            "cluster_selection_epsilon = 0 (default, sin límite).\n"
            "Referencia histórica (63 clusters, 38 persistentes) NO se usa "
            "para tunear — solo comprobación de orden de magnitud."
        ),
    }
    save_markdown_report(sections, Path("reports/31_hdbscan.md"))
    print("\nDone. See reports/31_hdbscan.md.")


if __name__ == "__main__":
    main()