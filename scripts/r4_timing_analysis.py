"""r4 Job 1 analysis: pair cached CJ timing with new fusion timing.

CJ per-protein wall_clock_seconds + peak_gpu_gb are cached in
results/cl7_phase15_cj_<m>_<dataset>.json (across L=30..2180). Fusion per-protein
timing comes from results/r4_fusion_timing_<V>_<dataset>.json (this PR). We pair
by protein_id (same proteins, same A100-80GB) and emit:

  - corrected Table A18 numbers: CJ s/prot, REAL fusion s/test, speedup, peak mem
  - per-(variant,dataset) medians + per-protein speedup distribution
  - data for the regenerated F2 panel (a): fusion wall vs L (real, not flat)

Pure I/O; no GPU. Run after r4_fusion_timing.py lands its JSONs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

# fusion-timing variant tag -> CJ-cache lowercase tag
CJ_TAG = {"8M": "8m", "35M": "35m", "150M": "150m", "650M": "650m", "3B": "3b"}


def _cj_timing(variant: str, dataset: str) -> dict[str, tuple[float, float, int]]:
    f = RESULTS / f"cl7_phase15_cj_{CJ_TAG[variant]}_{dataset}.json"
    if not f.is_file():
        return {}
    d = json.load(f.open())
    out = {}
    for r in d["per_protein"]:
        if r.get("skipped"):
            continue
        out[r["protein_id"]] = (
            float(r["wall_clock_seconds"]),
            float(r.get("peak_gpu_gb", float("nan"))),
            int(r["sequence_length"]),
        )
    return out


def _fusion_timing(variant: str, dataset: str) -> dict[str, tuple[float, float, int]]:
    f = RESULTS / f"r4_fusion_timing_{variant}_{dataset}.json"
    if not f.is_file():
        return {}
    d = json.load(f.open())
    out = {}
    for r in d["per_protein"]:
        if r.get("skipped"):
            continue
        out[r["protein_id"]] = (
            float(r["fusion_wall_clock_seconds"]),
            float(r.get("fusion_peak_gpu_gb", float("nan"))),
            int(r["sequence_length"]),
        )
    return out


def main(argv: list[str] | None = None) -> int:
    variants = ["35M", "150M", "650M", "3B"]
    datasets = ["zhang_eval200", "casp14_fm"]
    rows = []
    fig_points = {}  # (variant, dataset) -> list of (L, cj_wall, fus_wall)
    for v in variants:
        for ds in datasets:
            cj = _cj_timing(v, ds)
            fus = _fusion_timing(v, ds)
            shared = sorted(set(cj) & set(fus))
            if not shared:
                continue
            pts = [(cj[p][2], cj[p][0], fus[p][0]) for p in shared]
            fig_points[(v, ds)] = pts
            cj_w = np.array([cj[p][0] for p in shared])
            fus_w = np.array([fus[p][0] for p in shared])
            cj_m = np.array([cj[p][1] for p in shared])
            fus_m = np.array([fus[p][1] for p in shared])
            speed = cj_w / np.clip(fus_w, 1e-9, None)
            rows.append(
                {
                    "variant": v,
                    "dataset": ds,
                    "n": len(shared),
                    "cj_s_med": float(np.median(cj_w)),
                    "fus_s_med": float(np.median(fus_w)),
                    "cj_s_mean": float(np.mean(cj_w)),
                    "fus_s_mean": float(np.mean(fus_w)),
                    "speedup_med": float(np.median(speed)),
                    "speedup_min": float(np.min(speed)),
                    "speedup_max": float(np.max(speed)),
                    "cj_peak_gb_max": float(np.nanmax(cj_m)),
                    "fus_peak_gb_max": float(np.nanmax(fus_m)),
                }
            )

    print("\n=== r4 corrected timing (paired CJ vs fusion, A100-80GB) ===")
    hdr = f"{'variant':>7} {'dataset':>14} {'N':>4} {'CJ s med':>9} {'Fus s med':>10} {'speedup med':>11} {'speedup rng':>14} {'CJ peakGB':>9} {'Fus peakGB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['variant']:>7} {r['dataset']:>14} {r['n']:>4} {r['cj_s_med']:>9.2f} "
            f"{r['fus_s_med']:>10.3f} {r['speedup_med']:>11.1f} "
            f"{r['speedup_min']:>5.1f}-{r['speedup_max']:<7.1f} "
            f"{r['cj_peak_gb_max']:>9.2f} {r['fus_peak_gb_max']:>10.2f}"
        )

    out = RESULTS / "r4_timing_paired.json"
    out.write_text(
        json.dumps(
            {"rows": rows, "fig_points": {f"{k[0]}|{k[1]}": v for k, v in fig_points.items()}},
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
