"""r4 confidence analysis plots (local, no GPU). Consumes
results/r4_confidence_650M_zhang_eval200.json + results/r4_confmaps/*.npz.

Produces:
  paper/figures/F7_confidence_examples.png  -- per-protein method maps + true contacts
  paper/figures/F7_confidence_aggregate.png -- sharpness-vs-precision + calibration
and prints an aggregate confidence table per method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "paper" / "figures"
MAPS = RESULTS / "r4_confmaps"

METHODS = ["fusion", "top1", "logit", "repr", "wtw"]
LABEL = {"fusion": "Fusion (top-K)", "top1": "Top-1 head", "logit": "logit-CJ",
         "repr": "repr-CJ", "wtw": "WtW-CJ"}


def _aggregate(rows):
    scored = [r for r in rows if not r.get("skipped") and r.get("metrics")]
    agg = {}
    for m in METHODS:
        vals = {k: [] for k in ("precision", "cutoff_z", "gini", "dprime", "top_decile_precision")}
        deciles = []
        for r in scored:
            mm = r["metrics"].get(m)
            if not mm:
                continue
            for k in vals:
                if mm.get(k) is not None and not np.isnan(mm[k]):
                    vals[k].append(mm[k])
            deciles.append(mm["decile_precisions"])
        agg[m] = {
            k: (float(np.mean(v)), float(np.std(v) / np.sqrt(max(1, len(v))))) for k, v in vals.items()
        }
        agg[m]["decile_curve"] = np.nanmean(np.array(deciles, dtype=float), axis=0).tolist() if deciles else []
        agg[m]["n"] = len(deciles)
    return agg, scored


def _print_table(agg):
    print("\n=== Confidence by method (ESM-2-650M, Zhang eval-200; mean ± sem) ===")
    hdr = f"{'method':>14} {'P@L/2':>12} {'sharpness(z)':>14} {'gini':>11} {'sep(d-prime)':>14} {'top-decile P':>13}"
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        a = agg[m]
        def f(k):
            mu, se = a[k]; return f"{mu:.3f}±{se:.3f}"
        print(f"{LABEL[m]:>14} {f('precision'):>12} {f('cutoff_z'):>14} {f('gini'):>11} "
              f"{f('dprime'):>14} {f('top_decile_precision'):>13}  (n={a['n']})")


def _fig_aggregate(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    palette = {"fusion": "#2166ac", "top1": "#92c5de", "logit": "#b2182b",
               "repr": "#ef8a62", "wtw": "#fddbc7"}
    # (a) sharpness vs precision
    for m in METHODS:
        x, xe = agg[m]["cutoff_z"]; y, ye = agg[m]["precision"]
        axL.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=8, color=palette[m],
                     ecolor="#999", elinewidth=0.8, capsize=2, label=LABEL[m])
        axL.annotate(LABEL[m], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    axL.set_xlabel("Sharpness  (cutoff z-score: how far the L/2-th\nscore sits above the map mean)")
    axL.set_ylabel("P@$L/2$ long (correctness)")
    axL.set_title("(a) Confident vs correct", fontsize=10)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.grid(lw=0.3, alpha=0.35); axL.set_axisbelow(True)
    # (b) calibration curves
    for m in METHODS:
        c = agg[m]["decile_curve"]
        if c:
            axR.plot(range(1, len(c) + 1), c, "-o", ms=3.5, color=palette[m], label=LABEL[m], lw=1.3)
    axR.set_xlabel("Score decile (1 = highest-scoring pairs)")
    axR.set_ylabel("Precision (fraction true contacts)")
    axR.set_title("(b) Calibration: does a high score mean a contact?", fontsize=10)
    axR.set_xticks(range(1, 11))
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.grid(lw=0.3, alpha=0.35); axR.set_axisbelow(True)
    axR.legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(FIGS / "F7_confidence_aggregate.png", dpi=170)
    fig.savefig(FIGS / "F7_confidence_aggregate.pdf")
    plt.close(fig)
    print(f"wrote {FIGS/'F7_confidence_aggregate.png'}")


def _fig_examples(save_ids, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = {r["protein_id"]: r for r in rows}
    save_ids = [p for p in save_ids if (MAPS / f"{p}_fusion.npz").exists()]
    nrow, ncol = len(save_ids), len(METHODS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.3 * nrow), squeeze=False)
    for ri, pid in enumerate(save_ids):
        cd = np.load(MAPS / f"{pid}_contacts.npz")
        C = cd["c"]
        L = C.shape[0]
        ii, jj = np.indices((L, L))
        long = (jj - ii) >= 24
        ys, xs = np.where(C & long)  # true long-range contacts (upper tri)
        for ci, m in enumerate(METHODS):
            ax = axes[ri][ci]
            M = np.load(MAPS / f"{pid}_{m}.npz")["m"].astype(np.float64)
            vis = M.copy()
            vmax = np.percentile(vis, 99.5)
            vmin = np.percentile(vis, 50)
            ax.imshow(vis, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
            # overlay true long-range contacts (mirror to both triangles) as faint cyan dots
            ax.scatter(xs, ys, s=1.2, c="#22d3ee", alpha=0.6, linewidths=0)
            ax.scatter(ys, xs, s=1.2, c="#22d3ee", alpha=0.6, linewidths=0)
            prec = meta[pid]["metrics"].get(m, {}).get("precision", float("nan"))
            z = meta[pid]["metrics"].get(m, {}).get("cutoff_z", float("nan"))
            if ri == 0:
                ax.set_title(LABEL[m], fontsize=9)
            ax.set_xlabel(f"P={prec:.2f}  z={z:.1f}", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"{pid}\nL={L}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-method score maps (magma) with true long-range contacts (cyan); "
                 "P = P@L/2 long, z = sharpness", fontsize=9, y=1.005)
    fig.tight_layout()
    fig.savefig(FIGS / "F7_confidence_examples.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIGS / "F7_confidence_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS/'F7_confidence_examples.png'}  ({nrow} proteins x {ncol} methods)")


def _fig_paper(agg, scored, save_ids):
    """Combined early-paper figure: 3 example proteins (shortest/middle/longest)
    x 5 methods on top (short-range band masked), confident-vs-correct scatter
    below. Per Rome: keep examples + scatter, drop the decile-calibration panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = {r["protein_id"]: r for r in scored}
    have = [p for p in save_ids if (MAPS / f"{p}_fusion.npz").exists()]
    bylen = sorted(have, key=lambda p: meta[p]["sequence_length"])
    examples = [bylen[0], bylen[len(bylen) // 2], bylen[-1]]  # shortest, middle, longest

    import matplotlib as mpl
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(9.8, 10.6))
    gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 0.34, 1.7],
                          hspace=0.38, wspace=0.08)
    for ri, pid in enumerate(examples):
        C = np.load(MAPS / f"{pid}_contacts.npz")["c"]
        L = C.shape[0]
        ii, jj = np.indices((L, L))
        long = (jj - ii) >= 24
        band = np.abs(jj - ii) < 24
        ys, xs = np.where(C & long)
        for ci, m in enumerate(METHODS):
            ax = fig.add_subplot(gs[ri, ci])
            M = np.load(MAPS / f"{pid}_{m}.npz")["m"].astype(np.float64)
            disp = M.copy()
            disp[band] = np.nan  # hide the short-range diagonal; paper is long-range
            cmap = plt.cm.magma.copy()
            cmap.set_bad("#101010")
            # FULL dynamic range (no confidence threshold) so the map's own
            # diffuseness/concentration is visible; sharpness is quantified in
            # the panel below. Tiny tail clips only keep outliers from washing out.
            vmin = np.nanpercentile(disp, 2)
            vmax = np.nanpercentile(disp, 99.5)
            ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.scatter(xs, ys, s=3.0, c="#27e0ef", alpha=0.95, linewidths=0)
            ax.scatter(ys, xs, s=3.0, c="#27e0ef", alpha=0.95, linewidths=0)
            prec = meta[pid]["metrics"].get(m, {}).get("precision", float("nan"))
            if ri == 0:
                ax.set_title(LABEL[m], fontsize=10, pad=5)
            ax.set_xlabel(f"P@$L/2$ = {prec:.2f}", fontsize=8)
            if ci == 0:
                tag = ["shortest", "middle", "longest"][ri]
                ax.set_ylabel(f"{tag}\n{pid}\n$L$ = {L}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    # ── strip: colorbar (left) + cyan-contact legend (right) ──────────────
    axst = fig.add_subplot(gs[3, :]); axst.axis("off")
    cax = axst.inset_axes([0.18, 0.45, 0.22, 0.40])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="magma")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=8.5)
    cb.set_label("predicted score (each map scaled independently)", fontsize=9)
    axst.scatter([0.58], [0.62], s=55, c="#27e0ef", transform=axst.transAxes, clip_on=False)
    axst.text(0.605, 0.62, "true long-range contact  ($|i{-}j|\\geq 24$)",
              transform=axst.transAxes, fontsize=10, va="center")

    # ── scatter: confident vs accurate (full width, legend not labels) ────
    palette = {"fusion": "#1a5fb4", "top1": "#99c1f1", "logit": "#c01c28",
               "repr": "#e66100", "wtw": "#813d9c"}
    axs = fig.add_subplot(gs[4, :])
    handles = []
    for m in METHODS:
        x, xe = agg[m]["cutoff_z"]; y, ye = agg[m]["precision"]
        axs.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=12, color=palette[m],
                     ecolor="#c7c7c7", elinewidth=1.1, capsize=3, zorder=3)
        handles.append(Line2D([], [], marker="o", color=palette[m], linestyle="none",
                              ms=10, label=LABEL[m]))
    axs.legend(handles=handles, loc="lower right", frameon=False, fontsize=10,
               title="readout", title_fontsize=10)
    axs.set_xlabel("Confidence  ($\\rightarrow$ sharper: how far the $L/2$-th score sits above the map mean)",
                   fontsize=11)
    axs.set_ylabel("Accuracy\n(P@$L/2$ long)", fontsize=11)
    axs.set_title("Confident vs. accurate across $N{=}40$ proteins (mean $\\pm$ sem)",
                  fontsize=11.5, pad=6)
    axs.margins(0.16)
    for s in ("top", "right"):
        axs.spines[s].set_visible(False)
    axs.grid(lw=0.3, alpha=0.35); axs.set_axisbelow(True)
    axs.tick_params(labelsize=9.5)

    fig.savefig(FIGS / "F_confidence.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGS / "F_confidence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS/'F_confidence.png'} (examples: {examples})")


def main(argv=None):
    f = RESULTS / "r4_confidence_650M_zhang_eval200.json"
    if not f.is_file():
        print(f"missing {f}", file=sys.stderr); return 1
    d = json.loads(f.read_text())
    LABEL["fusion"] = f"Fusion (top-{d.get('k', 10)})"  # show the actual K, not "K"
    rows = d["per_protein"]
    agg, scored = _aggregate(rows)
    _print_table(agg)
    _fig_paper(agg, scored, d["save_ids"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
