"""Generate appendix tables and figures for the paper.

Outputs:
  paper/appendix_data.json — machine-readable tables A1, A4, A5, B
  paper/figures/FA2_progen2_head_profile.pdf/png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_JSON = ROOT / "paper" / "appendix_data.json"
OUT_FIG_DIR = ROOT / "paper" / "figures"


def _load_json(path):
    """Read JSON from a path via a context manager (closes the handle promptly)."""
    with open(path) as fh:
        return json.load(fh)


def bootstrap_ci(xs, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    means = np.sort([rng.choice(xs, size=len(xs), replace=True).mean() for _ in range(n_boot)])
    return (
        float(np.mean(xs)),
        float(means[int(n_boot * alpha / 2)]),
        float(means[int(n_boot * (1 - alpha / 2))]),
        len(xs),
    )


def per_protein_fusion(variant, dataset, op="naive_mean"):
    f = RES / f"cl7_phase15_fusion_{variant}_{dataset}.json"
    if not f.exists():
        return None, None
    d = _load_json(f)
    per = d.get("per_protein", [])
    top1, fus = [], []
    for p in per:
        if p.get("skipped"):
            continue
        r = p.get("results", {}).get("supervised_topk_b2", {})
        if "top1" in r:
            v = r["top1"]
            top1.append(v if isinstance(v, (int, float)) else v.get("P_at_L_over_2_long", np.nan))
        if op in r:
            v = r[op]
            fus.append(v if isinstance(v, (int, float)) else v.get("P_at_L_over_2_long", np.nan))
    return top1, fus


def per_protein_cj_650M_zhang():
    d = _load_json(RES / "cl7_phase15_overlap.json")
    return [r["cj_P_at_L_over_2"] for r in d["rows"]]


def per_protein_cj(variant, dataset):
    if dataset == "zhang_eval200":
        # eval-200 rerun: we ran logit-CJ at every scale; read the per-cell JSON directly.
        suffix = {
            "8M": "8m",
            "35M": "35m",
            "150M": "150m",
            "650M": "650m",
            "3B": "3b",
            "esm1b_1b": "1b",
        }.get(variant)
        if suffix is None:
            return None
        f = RES / f"cl7_phase15_cj_{suffix}_zhang_eval200.json"
        if not f.exists():
            return None
        d = _load_json(f)
        return [p["p_long"] for p in d["per_protein"] if not p.get("skipped")]
    fmap = {
        ("650M", "zhang_random_50"): None,
        ("3B", "zhang_random_50"): "cl7_phase15_cj_3b_zhang_random_50.json",
        ("8M", "casp14_fm"): "cl7_phase15_cj_8m_casp14_fm.json",
        ("35M", "casp14_fm"): "cl7_phase15_cj_35m_casp14_fm.json",
        ("150M", "casp14_fm"): "cl7_phase15_cj_150m_casp14_fm.json",
        ("3B", "casp14_fm"): "cl7_phase15_cj_3b_casp14_fm.json",
        ("esm1b_1b", "zhang_random_50"): "cl7_phase15_cj_1b_zhang_random_50.json",
        ("esm1b_1b", "casp14_fm"): "cl7_phase15_cj_1b_casp14_fm.json",
    }
    key = (variant, dataset)
    # 650M CASP14-FM logit-CJ lives in the scaling summary (standard_cj_batched),
    # not a dedicated per-cell file.
    if key == ("650M", "casp14_fm"):
        f = RES / "casp14_fm__scaling_summary.json"
        if not f.exists():
            return None
        d = _load_json(f)
        per = d.get("standard_cj_batched", {}).get("per_protein", {})
        return [v["P_long"] for v in per.values() if v.get("P_long") is not None]
    if key not in fmap:
        return None
    if key == ("650M", "zhang_random_50"):
        return per_protein_cj_650M_zhang()
    if fmap[key] is None:
        return None
    f = RES / fmap[key]
    if not f.exists():
        return None
    d = _load_json(f)
    return [p["p_long"] for p in d["per_protein"] if not p.get("skipped")]


def per_protein_repr_cj(variant, dataset):
    candidates = [
        f"cl7_phase15_repr_cj_esm2_{variant}_{dataset}.json",
        f"cl7_phase15_repr_cj_{variant}_{dataset}.json",
    ]
    for c in candidates:
        f = RES / c
        if f.exists():
            d = _load_json(f)
            return [p["p_long"] for p in d["per_protein"] if not p.get("skipped")]
    return None


# ----- A1: per-(variant, dataset) bootstrap CI -----
def build_a1():
    rows = []
    # In-distribution eval migrated Zhang-50 -> disjoint eval-200 (select on select-10).
    # CASP14-FM retained as the cross-dataset generalization check.
    cells = [
        ("8M", "zhang_eval200"),
        ("8M", "casp14_fm"),
        ("35M", "zhang_eval200"),
        ("35M", "casp14_fm"),
        ("150M", "zhang_eval200"),
        ("150M", "casp14_fm"),
        ("650M", "zhang_eval200"),
        ("650M", "casp14_fm"),
        ("3B", "zhang_eval200"),
        ("3B", "casp14_fm"),
        ("esm1b_1b", "zhang_eval200"),
        ("esm1b_1b", "casp14_fm"),
        ("amplify_350M", "zhang_eval200"),
        ("amplify_350M", "casp14_fm"),
        ("prott5_XL", "zhang_eval200"),
        ("prott5_XL", "casp14_fm"),
    ]
    for var, ds in cells:
        top1, fus = per_protein_fusion(var, ds)
        if top1 is None or len(top1) == 0:
            # synthesis fallback (no CI)
            continue
        t1_m, t1_lo, t1_hi, t1_n = bootstrap_ci(top1)
        f_m, f_lo, f_hi, f_n = bootstrap_ci(fus)
        cj = per_protein_cj(var, ds)
        rcj = per_protein_repr_cj(var, ds)
        cj_stats = bootstrap_ci(cj) if cj else (None,) * 4
        rcj_stats = bootstrap_ci(rcj) if rcj else (None,) * 4
        rows.append(
            {
                "variant": var,
                "dataset": ds,
                "top1": {"mean": t1_m, "ci_lo": t1_lo, "ci_hi": t1_hi, "n": t1_n},
                "fusion_naive_mean": {"mean": f_m, "ci_lo": f_lo, "ci_hi": f_hi, "n": f_n},
                "cj": (
                    {
                        "mean": cj_stats[0],
                        "ci_lo": cj_stats[1],
                        "ci_hi": cj_stats[2],
                        "n": cj_stats[3],
                    }
                    if cj
                    else None
                ),
                "repr_cj": (
                    {
                        "mean": rcj_stats[0],
                        "ci_lo": rcj_stats[1],
                        "ci_hi": rcj_stats[2],
                        "n": rcj_stats[3],
                    }
                    if rcj
                    else None
                ),
            }
        )
    return rows


# ----- A4: AMPLIFY repr-CJ layer sweep -----
def build_a4():
    rows = []
    for layer in [0, 8, 16, 24, 28, 31]:
        f = RES / f"cl7_phase15_repr_cj_amplify_350M_layer{layer}_zhang_eval200.json"
        if not f.exists():
            continue
        d = _load_json(f)
        vals = [p["p_long"] for p in d["per_protein"] if not p.get("skipped")]
        m, lo, hi, n = bootstrap_ci(vals)
        rows.append({"layer": layer, "mean": m, "ci_lo": lo, "ci_hi": hi, "n": n})
    return rows


# ----- A5: per-test wall-clock + holdout transfer -----
def build_a5():
    cost = _load_json(RES / "cl7_phase15_cost_accounting.json")["rows"]
    wall = []
    for r in cost:
        wall.append(
            {
                "variant": r["variant"],
                "dataset": r["dataset"],
                "cj_seconds_per_protein": r["cj_mean_seconds"],
                "cj_forwards_per_protein": r["cj_mean_forwards"],
                "fusion_seconds_per_test": r["fusion_per_test_seconds"],
                "identification_total_seconds": r["identification_seconds_total"],
                "cj_total_USD_at_N1000": r["cj_total_USD_at_N1000"],
                "fusion_total_USD_at_N1000": r["fusion_total_USD_at_N1000"],
                "speedup_at_N1000": r["speedup_at_N1000"],
            }
        )
    # Holdout transfer (seed-43)
    transfer = []
    for variant, dataset in [("650M", "zhang_random_50"), ("3B", "zhang_random_50")]:
        seed42_f = RES / f"cl7_phase15_fusion_sprint3_{variant}_{dataset}.json"
        seed43_f = RES / f"cl7_phase15_fusion_sprint3_{variant}_{dataset}_seed43.json"
        for label, path in [("seed42", seed42_f), ("seed43", seed43_f)]:
            if path.exists():
                d = _load_json(path)
                # Find naive_mean mean
                s = d.get("summary", {}).get("supervised_topk_b2", {}).get("naive_mean", {})
                if s:
                    transfer.append(
                        {"variant": variant, "split": label, "mean": s.get("mean"), "n": s.get("n")}
                    )
    return {"per_test_wall": wall, "holdout_transfer": transfer}


# ----- B: head-precision rank distributions -----
def build_b():
    archs = {
        "esm2_8M": ("b2_head_probe_8M_zhang_eval200.json", "L5 H4"),
        "esm2_35M": ("b2_head_probe_35M_zhang_eval200.json", "L11 H13"),
        "esm2_150M": ("b2_head_probe_150M_zhang_eval200.json", "L28 H5"),
        "esm2_650M": ("b2_head_probe_zhang_eval200.json", "L32 H13"),
        "esm2_3B": ("b2_head_probe_3B_zhang_eval200.json", "L32 H39"),
        "esm1b": ("b2_head_probe_esm1b_1b_zhang_eval200.json", "L29 H7"),
        "amplify_350M": ("b2_head_probe_amplify_350M_zhang_eval200.json", "L31 H11"),
        "prott5_XL": ("b2_head_probe_prott5_XL_zhang_eval200.json", "L22 H11"),
        "progen2_xl": ("b2_head_probe_progen2_xlarge_zhang_eval200.json", "L26 H2"),
    }
    rows = []
    for label, (fname, top1) in archs.items():
        f = RES / fname
        if not f.exists():
            continue
        layer, head = [int(x[1:]) for x in top1.split()]
        d = _load_json(f)
        per = d["per_protein"]
        nL, nH = per[0]["n_layers"], per[0]["n_heads"]
        ranks = []
        for entry in per:
            hp = np.array(entry["head_precisions"]).reshape(nL, nH)
            flat = hp.flatten()
            idx = layer * nH + head
            ranks.append(int((flat > flat[idx]).sum()) + 1)
        ranks = np.asarray(ranks)
        rows.append(
            {
                "arch": label,
                "global_top1": top1,
                "n_heads": int(nL * nH),
                "n_proteins": len(ranks),
                "win_rate_global_is_perprotein_top1": float((ranks == 1).mean()),
                "top1_within_top3": float((ranks <= 3).mean()),
                "top1_within_top10": float((ranks <= 10).mean()),
                "median_rank": int(np.median(ranks)),
            }
        )
    return rows


# ----- Figure A2: ProGen2 vs ESM-2-650M head-precision profile -----
def render_a2_figure():
    pairs = [
        # Both on eval-200 (ProGen2-xlarge eval-200 head-probe re-run with clean-container isolation).
        ("ESM-2-650M (MLM)", "b2_head_probe_zhang_eval200.json", "tab:blue"),
        (
            "ProGen2-xlarge (causal LM)",
            "b2_head_probe_progen2_xlarge_zhang_eval200.json",
            "tab:red",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (label, fname, color) in zip(axes, pairs, strict=False):
        d = _load_json(RES / fname)
        per = d["per_protein"]
        nL, nH = per[0]["n_layers"], per[0]["n_heads"]
        # Per-(layer, head) MEAN precision across proteins
        all_hp = np.array([np.array(p["head_precisions"]).reshape(nL, nH) for p in per])
        mean_hp = all_hp.mean(axis=0)
        flat_sorted = np.sort(mean_hp.flatten())[::-1]
        ax.plot(np.arange(1, len(flat_sorted) + 1), flat_sorted, color=color, lw=1.4)
        ax.set_xscale("log")
        ax.set_xlabel("Head rank (mean P@L/2 long across selection set)")
        ax.set_ylabel("Mean P@L/2 long")
        ax.set_title(label)
        ax.set_ylim(0, max(0.75, flat_sorted[0] * 1.05))
        ax.grid(True, which="both", alpha=0.3)
        # Annotate top-1
        ax.scatter([1], [flat_sorted[0]], color=color, s=30, zorder=5)
        ax.annotate(f" top-1 = {flat_sorted[0]:.3f}", (1, flat_sorted[0]), fontsize=9)
    fig.suptitle("Figure A2 — Head-precision rank profile: MLM (sharp spike) vs causal (flat)")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "FA2_progen2_head_profile.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / "FA2_progen2_head_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----- repr-CJ vs logit-CJ Pearson r for F6 caption -----
def pearson_repr_logit_650M():
    d = _load_json(RES / "cl7_phase15_cj_650m_zhang_eval200.json")
    cj_rows = [r for r in d["per_protein"] if not r.get("skipped")]
    # repr-CJ per-protein
    rcj_d = _load_json(RES / "cl7_phase15_repr_cj_esm2_650M_zhang_eval200.json")
    rcj_map = {p["protein_id"]: p["p_long"] for p in rcj_d["per_protein"] if not p.get("skipped")}
    paired = []
    for r in cj_rows:
        pid = r["protein_id"]
        if pid in rcj_map:
            paired.append((r["p_long"], rcj_map[pid]))
    if not paired:
        return None
    xs, ys = zip(*paired, strict=False)
    r = float(np.corrcoef(xs, ys)[0, 1])
    return {
        "n": len(paired),
        "pearson_r": r,
        "mean_logit": float(np.mean(xs)),
        "mean_repr": float(np.mean(ys)),
    }


def main():
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "A1_per_protein_bootstrap": build_a1(),
        "A4_amplify_layer_sweep": build_a4(),
        "A5_wall_clock_and_transfer": build_a5(),
        "B_head_rank_distributions": build_b(),
        "F6_repr_vs_logit_pearson": pearson_repr_logit_650M(),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    render_a2_figure()
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_FIG_DIR / 'FA2_progen2_head_profile.pdf'}")


if __name__ == "__main__":
    main()
