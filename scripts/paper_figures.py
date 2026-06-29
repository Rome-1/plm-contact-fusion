"""All paper figures from cached results — single reproducible build.

Usage:
    PYTHONPATH=src python3 scripts/paper_figures.py
    # writes paper/figures/F{1..6}.{pdf,png}

Figures:
  F1 Headline grouped bars: fusion vs CJ vs top-1 across architectures (Zhang-50)
  F2 Cost economy log-log: GPU $ vs N test proteins
  F3 K-sweep curves per architecture
  F4 Cluster geometry: per-model head precision distribution (histograms)
  F5 ProGen2 negative result: matched-scale bars vs ESM-2-650M (every method)
  F6 repr-CJ vs logit-CJ scatter on ESM-2-650M (per-protein)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        out = FIGURES / f"{name}.{ext}"
        fig.savefig(out)
    plt.close(fig)


def _load_synthesis() -> list[dict]:
    p = RESULTS / "cl7_phase15_synthesis.json"
    with p.open() as fh:
        return json.load(fh)["rows"]


def _pretty_label(family: str, variant: str) -> str:
    """Capitalize architecture/variant identifiers for figure axes."""
    family_map = {
        "esm2": "ESM-2",
        "esm1b": "ESM-1b",
        "amplify": "AMPLIFY",
        "prott5": "ProtT5",
        "progen2": "ProGen2",
    }
    f = family_map.get(family, family.upper())
    return f"{f}-{variant}"


def figure_1_headline() -> None:
    """Grouped bars: top-1, fusion, best-CJ for each (family, variant) on
    Zhang-50. Backfills CJ entries for the ESM-2 small variants from
    per-cell JSONs (the synthesis file has them as NaN)."""
    rows = _load_synthesis()
    cells = [r for r in rows if r["dataset"] == "zhang_eval200" and not np.isnan(r["top1_p"])]

    # Backfill CJ from the per-cell JSONs where the synthesis is NaN.
    backfill_map = {
        ("esm2", "8M"): "cl7_phase15_cj_8m_zhang_eval200.json",
        ("esm2", "35M"): "cl7_phase15_cj_35m_zhang_eval200.json",
        ("esm2", "150M"): "cl7_phase15_cj_150m_zhang_eval200.json",
    }
    for r in cells:
        key = (r["family"], r["variant"])
        if key in backfill_map and (
            r.get("cj_baseline") is None or np.isnan(r.get("cj_baseline", float("nan")))
        ):
            p = RESULTS / backfill_map[key]
            if p.exists():
                d = json.load(p.open())
                vals = [pp["p_long"] for pp in d["per_protein"] if not pp.get("skipped")]
                if vals:
                    r["cj_baseline"] = float(np.mean(vals))

    def _label(fam, var):
        m = {
            "esm2": "ESM-2",
            "esm1b": "ESM-1b",
            "amplify": "AMPLIFY",
            "prott5": "ProtT5",
            "progen2": "ProGen2",
        }
        if fam == "esm1b":
            return "ESM-1b"
        if fam == "progen2":
            return "ProGen2"
        if fam == "prott5":
            return "ProtT5-XL"
        return f"{m.get(fam, fam.upper())}-{var}"

    labels = [_label(r["family"], r["variant"]) for r in cells]
    top1 = np.array([r["top1_p"] for r in cells])
    fusion = np.array([r["naive_mean"] for r in cells])
    best_cj = np.array(
        [
            max(
                v
                for v in [r.get("cj_baseline"), r.get("repr_cj")]
                if v is not None and not np.isnan(v)
            )
            if any(
                v is not None and not np.isnan(v) for v in [r.get("cj_baseline"), r.get("repr_cj")]
            )
            else np.nan
            for r in cells
        ]
    )

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(cells))
    w = 0.27
    bar_kwargs = dict(linewidth=0.4, edgecolor="white")
    ax.bar(x - w, top1, w, label="best single head", color="#bfbfbf", **bar_kwargs)
    ax.bar(x, fusion, w, label=r"fusion (top-$K$ mean, $K{=}10$)", color="#2171b5", **bar_kwargs)
    cj_mask = ~np.isnan(best_cj)
    ax.bar(
        x[cj_mask] + w,
        best_cj[cj_mask],
        w,
        label="Categorical Jacobian",
        color="#d94801",
        **bar_kwargs,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    # Dataset is stated in the caption; the parenthetical here clipped the
    # closing ')' at the top of the rotated y-label under bbox='tight'.
    ax.set_ylabel(r"top-$L/2$ long-range precision")
    ax.set_ylim(0, 0.95)
    ax.set_yticks(np.arange(0.0, 1.0, 0.2))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2, columnspacing=0.8)
    _save(fig, "F1_headline")


def figure_2_cost_economy() -> None:
    """Two-panel cost figure:
      (a) per-protein wall-clock vs protein length L (showing CJ's O(L) scaling
          and fusion's L-independence)
      (b) amortized total wall-clock vs N test proteins (linear axes)
    Rebuilt per r3 feedback: no log-log, clean legend (one row per arch),
    L on x-axis added.
    """
    # ── Panel (a): paired CJ-vs-fusion wall vs L (real per-protein timings) ──
    # CJ: cached per-protein wall_clock (cl7_phase15_cj_*). Fusion: measured
    # per-protein wall_clock from the r4 paired-timing run (same proteins, same
    # A100-80GB). Falls back to the prior dotted-horizontal estimate only if the
    # r4 fusion-timing JSON is absent.
    arch_files = [
        ("ESM-2-150M", "150m", "150M", "#9ecae1", 2.0),
        ("ESM-2-650M", "650m", "650M", "#4292c6", 8.0),
        ("ESM-2-3B", "3b", "3B", "#08519c", 18.0),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1, 1.05]}, constrained_layout=True
    )
    ax_left, ax_right = axes

    cj_handles = []
    ymax = 1.0
    for arch, cjtag, fustag, c, fusion_s_fallback in arch_files:
        d = json.load((RESULTS / f"cl7_phase15_cj_{cjtag}_zhang_eval200.json").open())
        rows = [r for r in d["per_protein"] if not r.get("skipped")]
        Ls = np.array([r["sequence_length"] for r in rows])
        walls = np.array([r["wall_clock_seconds"] for r in rows])
        order = np.argsort(Ls)
        ymax = max(ymax, float(walls.max()))
        h = ax_left.scatter(
            Ls[order], walls[order], s=14, color=c, alpha=0.85,
            edgecolors="white", linewidths=0.4, label=f"{arch} CJ",
        )
        cj_handles.append(h)
        # fusion: real per-protein scatter (open markers), same color.
        fp = RESULTS / f"r4_fusion_timing_{fustag}_zhang_eval200.json"
        if fp.exists():
            fd = json.load(fp.open())
            fr = [r for r in fd["per_protein"] if not r.get("skipped")]
            fL = np.array([r["sequence_length"] for r in fr])
            fw = np.array([r["fusion_wall_clock_seconds"] for r in fr])
            fo = np.argsort(fL)
            ax_left.scatter(
                fL[fo], fw[fo], s=15, facecolors="none", edgecolors=c,
                linewidths=0.9, marker="o", alpha=0.9,
            )
        else:
            ax_left.axhline(fusion_s_fallback, color=c, linestyle=":", linewidth=1.4, alpha=0.85)

    ax_left.set_xlabel(r"Protein length $L$ (residues)")
    ax_left.set_ylabel("Wall-clock per protein (s)")
    ax_left.set_xlim(150, 620)
    # Log y: CJ (~10-300s) and fusion (~0.02-0.1s) span ~3 orders of magnitude;
    # a linear axis squashes fusion to zero and hides its own O(L^2) growth.
    ax_left.set_yscale("log")
    ax_left.set_ylim(0.02, ymax * 2.0)
    ax_left.set_title("(a) Per-protein wall-clock vs $L$ (log scale)", fontsize=10, pad=4)
    # Legend: filled markers = CJ per arch; one proxy = the open-marker fusion.
    from matplotlib.lines import Line2D

    fusion_proxy = Line2D(
        [0], [0], color="#555555", linestyle="none", marker="o",
        markerfacecolor="none", markeredgecolor="#555555", markersize=5,
        label="fusion (1 forward, measured)",
    )
    ax_left.legend(
        handles=[*cj_handles, fusion_proxy],
        loc="center left",
        bbox_to_anchor=(0.0, 0.42),
        fontsize=7.5,
        frameon=False,
        handletextpad=0.4,
    )
    for spine in ("top", "right"):
        ax_left.spines[spine].set_visible(False)
    ax_left.grid(axis="y", linewidth=0.3, alpha=0.35)
    ax_left.set_axisbelow(True)

    # ── Panel (b): MEASURED speedup vs model size (Zhang eval-200) ─────────
    # Per-protein median wall-clock speedup (CJ / fusion) with min-max whiskers,
    # from the r4 paired-timing run (same proteins, same A100-80GB). Fully
    # measured -- replaces the prior linear cost-amortization extrapolation.
    td = json.load((RESULTS / "r4_timing_paired.json").open())
    by_variant = {r["variant"]: r for r in td["rows"] if r["dataset"] == "zhang_eval200"}
    spd_arch = [
        ("35M", 35e6),
        ("150M", 150e6),
        ("650M", 650e6),
        ("3B", 3.0e9),
    ]
    xs, meds, los, his = [], [], [], []
    for var, params in spd_arch:
        r = by_variant[var]
        xs.append(params)
        meds.append(r["speedup_med"])
        los.append(r["speedup_med"] - r["speedup_min"])
        his.append(r["speedup_max"] - r["speedup_med"])
    xs = np.array(xs)
    meds = np.array(meds)
    ax_right.errorbar(
        xs, meds, yerr=[los, his], fmt="o-", color="#08519c",
        ecolor="#9ecae1", elinewidth=1.5, capsize=3, markersize=6,
        linewidth=1.6, label="median speedup (whiskers: per-protein min–max)",
    )
    # Annotate the endpoints so the 150x -> 1600x growth reads at a glance.
    ax_right.annotate(f"{meds[0]:.0f}$\\times$", xy=(xs[0], meds[0]), xytext=(6, -12),
                      textcoords="offset points", ha="left", fontsize=8.5, color="#08519c")
    ax_right.annotate(f"{meds[-1]:.0f}$\\times$", xy=(xs[-1], meds[-1]), xytext=(-2, 9),
                      textcoords="offset points", ha="right", fontsize=8.5, color="#08519c")
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.set_xlabel("Model size (parameters)")
    ax_right.set_ylabel(r"Median speedup (CJ / fusion, $\times$)")
    ax_right.set_xticks([p for _, p in spd_arch])
    ax_right.set_xticklabels([v for v, _ in spd_arch])
    ax_right.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_right.tick_params(axis="x", which="minor", length=0)
    ax_right.set_yticks([100, 300, 1000, 3000])
    ax_right.set_yticklabels(["100", "300", "1000", "3000"])
    ax_right.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_right.set_xlim(2.5e7, 4.0e9)
    ax_right.set_ylim(50, 4000)
    ax_right.set_title("(b) Measured speedup vs model size", fontsize=10, pad=4)
    for spine in ("top", "right"):
        ax_right.spines[spine].set_visible(False)
    ax_right.grid(axis="y", linewidth=0.3, alpha=0.35)
    ax_right.set_axisbelow(True)
    ax_right.legend(loc="upper left", fontsize=7, frameon=False, handletextpad=0.4)

    _save(fig, "F2_cost_economy")


def figure_3_ksweep() -> None:
    """K-sweep curves per architecture. Tasteful palette, sweet-K stars."""
    cells = [
        ("esm2", "8M", "8M_"),
        ("esm2", "35M", "35M_"),
        ("esm1b", "1b", "esm1b_1b_"),
        ("amplify", "350M", "amplify_350M_"),
        ("prott5", "XL", "prott5_XL_"),
        ("progen2", "xlarge", "progen2_xlarge_"),
    ]
    Ks = [1, 2, 3, 5, 7, 10, 20]

    # Concentrated-cluster arches in warm earth-tones; diffuse-cluster in cool blues.
    palette = {
        ("esm2", "8M"): "#bcbddc",  # light purple (concentrated, small)
        ("esm2", "35M"): "#807dba",  # purple (concentrated, larger)
        ("amplify", "350M"): "#c66e3f",  # rust (concentrated)
        ("progen2", "xlarge"): "#9c9c9c",  # neutral grey (causal LM scope boundary)
        ("esm1b", "1b"): "#41b6c4",  # teal (diffuse)
        ("prott5", "XL"): "#225ea8",  # deep blue (diffuse)
    }

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for family, variant, _suffix in cells:
        ys = []
        for k in Ks:
            family_tag = "" if family == "esm2" else f"_{family}"
            fpath = (
                RESULTS / f"cl7_phase15_fusion_ksweep_k{k}{family_tag}_{variant}_zhang_eval200.json"
            )
            if not fpath.exists():
                fpath = RESULTS / f"cl7_phase15_fusion_ksweep_k{k}_{variant}_zhang_eval200.json"
            if not fpath.exists():
                ys.append(np.nan)
                continue
            with fpath.open() as fh:
                d = json.load(fh)
            sup = d["summary"].get("supervised_topk_b2", {})
            ys.append(sup.get("naive_mean", {}).get("mean", np.nan))
        ys_arr = np.array(ys)
        if np.any(~np.isnan(ys_arr)):
            sweet_idx = int(np.nanargmax(ys_arr))
            sweet_K = Ks[sweet_idx]
            sweet_y = ys_arr[sweet_idx]
        else:
            sweet_K, sweet_y = None, None
        c = palette.get((family, variant), "#444")
        label_str = _pretty_label(family, variant).replace("ESM-1b-1b", "ESM-1b")
        label_str = label_str.replace("ProGen2-xlarge", "ProGen2")
        if sweet_K is not None:
            label_str += rf"  ($K^\star{{=}}{sweet_K}$)"
        ax.plot(
            Ks, ys_arr, "-o", color=c, label=label_str, linewidth=1.5, markersize=4.5, alpha=0.95
        )
        if sweet_K is not None:
            ax.scatter(
                [sweet_K],
                [sweet_y],
                s=110,
                marker="*",
                facecolors=c,
                edgecolors="white",
                linewidths=1.0,
                zorder=6,
            )

    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel(r"$K$ (number of top heads fused)")
    ax.set_ylabel(r"fusion top-$L/2$ long precision (Zhang eval-200)")
    ax.set_ylim(0.1, 0.78)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    # Legend outside on the right to keep the plotting area clean.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5,
        frameon=False,
        handletextpad=0.5,
    )
    plt.tight_layout()
    _save(fig, "F3_ksweep")


def figure_4_cluster_geometry() -> None:
    """Per-architecture: histogram of mean per-head precision (descending),
    with the sweet-K cluster highlighted."""
    cells = [
        ("ESM-2-8M", "8M_", 3),
        ("ESM-2-35M", "35M_", 10),
        ("ESM-2-650M", "", 10),
        ("ESM-2-3B", "3B_", 10),
        ("ESM-1b", "esm1b_1b_", 7),
        ("AMPLIFY-350M", "amplify_350M_", 3),
        ("ProtT5-XL", "prott5_XL_", 7),
        ("ProGen2", "progen2_xlarge_", 2),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(10.5, 4.6), sharex=True, sharey=False, constrained_layout=True
    )
    for ax, (label, suffix, sweet_K) in zip(axes.flat, cells, strict=False):
        path = RESULTS / f"b2_head_probe_{suffix}zhang_eval200.json"
        if not path.exists():
            ax.set_title(f"{label} (no data)")
            continue
        with path.open() as fh:
            d = json.load(fh)
        s = d["summary"]
        grid = np.array(s["mean_per_head"])
        flat = sorted(grid.flatten(), reverse=True)
        x = np.arange(20)
        ax.bar(x, flat[:20], color="#d9d9d9", linewidth=0.3, edgecolor="white", zorder=2)
        ax.bar(
            x[:sweet_K],
            flat[:sweet_K],
            color="#2171b5",
            linewidth=0.3,
            edgecolor="white",
            zorder=3,
            label=rf"top $K^\star{{=}}{sweet_K}$",
        )
        ax.set_title(label, fontsize=9, pad=2.5)
        ax.set_xlim(-0.7, 19.7)
        ax.tick_params(axis="both", labelsize=7, length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=7, frameon=False, handletextpad=0.4, handlelength=1.0)
    for ax in axes[-1, :]:
        ax.set_xlabel("head rank", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mean P@$L/2$ long", fontsize=8)
    _save(fig, "F4_cluster_geometry")


def figure_5_progen2_negative() -> None:
    """ProGen2 vs ESM-2-650M side-by-side: every method, matched scale."""
    rows = _load_synthesis()
    progen2 = next(r for r in rows if r["family"] == "progen2" and r["dataset"] == "zhang_eval200")
    esm650 = next(
        r
        for r in rows
        if r["family"] == "esm2" and r["variant"] == "650M" and r["dataset"] == "zhang_eval200"
    )

    methods = ["top1_p", "naive_mean", "cj_baseline", "repr_cj"]
    method_labels = ["top-1 head", "fusion (K=10)", "logit-CJ", "repr-CJ"]
    progen2_vals = [progen2.get(m, np.nan) for m in methods]
    esm_vals = [esm650.get(m, np.nan) for m in methods]

    # Replace NaN with 0 for plotting and annotate "n/a"
    def _safe(v):
        return 0.0 if v is None or np.isnan(v) else v

    progen2_plot = [_safe(v) for v in progen2_vals]
    esm_plot = [_safe(v) for v in esm_vals]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(methods))
    w = 0.34
    bar_kw = dict(linewidth=0.4, edgecolor="white")
    b1 = ax.bar(x - w / 2, esm_plot, w, label="ESM-2-650M (MLM)", color="#2171b5", **bar_kw)
    b2 = ax.bar(
        x + w / 2, progen2_plot, w, label="ProGen2-xlarge (causal LM)", color="#7a7a7a", **bar_kw
    )
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel(r"top-$L/2$ long-range precision (Zhang eval-200)")
    ax.set_ylim(0, 0.95)
    ax.set_yticks(np.arange(0.0, 1.0, 0.2))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.5)
    for bars, vals in [(b1, esm_vals), (b2, progen2_vals)]:
        for bar, v in zip(bars, vals, strict=False):
            label = f"{v:.2f}" if v is not None and not np.isnan(v) else "n/a"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.012,
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222",
            )
    ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    _save(fig, "F5_progen2_negative")


def figure_6_repr_cj_validation() -> None:
    """Per-protein scatter: repr-CJ p_long vs logit-CJ p_long on ESM-2-650M Zhang-50."""
    cj_path = RESULTS / "cl7_phase15_cj_650m_zhang_eval200.json"
    repr_path = RESULTS / "cl7_phase15_repr_cj_esm2_650M_zhang_eval200.json"
    with repr_path.open() as fh:
        repr_d = json.load(fh)
    repr_per = {r["protein_id"]: r["p_long"] for r in repr_d["per_protein"] if not r.get("skipped")}

    # Logit-CJ per-protein p_long on eval-200.
    with cj_path.open() as fh:
        cj_d = json.load(fh)
    cj_per = {r["protein_id"]: r["p_long"] for r in cj_d["per_protein"] if not r.get("skipped")}

    common = sorted(set(repr_per) & set(cj_per))
    repr_v = np.array([repr_per[k] for k in common])
    cj_v = np.array([cj_per[k] for k in common])

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lo = min(repr_v.min(), cj_v.min()) - 0.02
    hi = max(repr_v.max(), cj_v.max()) + 0.02
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#aaaaaa",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
        label=r"$y = x$",
    )
    ax.scatter(
        cj_v,
        repr_v,
        s=22,
        alpha=0.85,
        color="#2171b5",
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
        label=f"ESM-2-650M Zhang eval-200  ($N{{=}}${len(common)})",
    )
    r = float(np.corrcoef(cj_v, repr_v)[0, 1])
    ax.set_xlabel(r"logit-CJ top-$L/2$ long precision")
    ax.set_ylabel(r"repr-CJ top-$L/2$ long precision")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.05,
        0.93,
        f"Pearson $r = {r:.2f}$",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color="#222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", lw=0.5),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    _save(fig, "F6_repr_cj_validation")


def main() -> None:
    print(f"Writing figures to {FIGURES}/")
    figure_1_headline()
    print("  F1 headline")
    figure_2_cost_economy()
    print("  F2 cost economy")
    figure_3_ksweep()
    print("  F3 K-sweep")
    figure_4_cluster_geometry()
    print("  F4 cluster geometry")
    figure_5_progen2_negative()
    print("  F5 ProGen2 negative")
    figure_6_repr_cj_validation()
    print("  F6 repr-CJ validation")
    print(
        f"\nDone. {len(list(FIGURES.glob('*.pdf')))} PDFs + {len(list(FIGURES.glob('*.png')))} PNGs in {FIGURES}"
    )


if __name__ == "__main__":
    main()
