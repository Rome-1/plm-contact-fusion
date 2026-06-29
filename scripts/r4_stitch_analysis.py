"""r4 stitching analysis + figure. Consumes results/r4_stitch_e1_zhang.json
(chunking-penalty ablation) + results/r4_stitch_e2_long.json (real long demo).

Emits paper/figures/F_stitch.png:
  (a) E1: full-forward vs stitched P@L/2 long vs window W, with the
      fraction-of-true-contacts-beyond-W ceiling.
  (b) E2: stitched P@L/2 long on the two real L>1024 CASP14 targets.
And prints an aggregate table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "paper" / "figures"


def _e1_agg(rows):
    rows = [r for r in rows if not r.get("skipped")]
    Ws = sorted({int(w) for r in rows for w in r.get("stitched", {})})
    full = np.array([r["full_p"] for r in rows if "full_p" in r])
    out = {"n": len(rows), "Ws": Ws,
           "full_mean": float(full.mean()), "full_sem": float(full.std() / np.sqrt(len(full)))}
    per_w = {}
    for W in Ws:
        ps = np.array([r["stitched"][str(W)]["p"] for r in rows if str(W) in r["stitched"]])
        fb = np.array([r["stitched"][str(W)]["frac_true_beyond_W"] for r in rows
                       if str(W) in r["stitched"] and not np.isnan(r["stitched"][str(W)]["frac_true_beyond_W"])])
        nw = np.array([r["stitched"][str(W)]["n_windows"] for r in rows if str(W) in r["stitched"]])
        per_w[W] = {"p_mean": float(ps.mean()), "p_sem": float(ps.std() / np.sqrt(len(ps))),
                    "frac_beyond_mean": float(fb.mean()) if len(fb) else float("nan"),
                    "n_windows_mean": float(nw.mean())}
    out["per_w"] = per_w
    return out


def _fig(e1, e2_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.7),
                                   gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    Ws = e1["Ws"]
    pm = [e1["per_w"][W]["p_mean"] for W in Ws]
    ps = [e1["per_w"][W]["p_sem"] for W in Ws]
    fb = [e1["per_w"][W]["frac_beyond_mean"] for W in Ws]
    axa.axhline(e1["full_mean"], color="#1a5fb4", ls="--", lw=1.4,
                label=f"full forward (no tiling): {e1['full_mean']:.2f}")
    axa.errorbar(Ws, pm, yerr=ps, fmt="-o", color="#e66100", ms=7, capsize=3,
                 label="stitched (tiled fusion)")
    axa.set_xlabel("window size $W$ (residues)")
    axa.set_ylabel("P@$L/2$ long")
    axa.set_title("(a) Chunking penalty vs window\n(Zhang eval-200, full forward feasible)", fontsize=9.5)
    axa.set_xticks(Ws)
    ax2 = axa.twinx()
    ax2.plot(Ws, fb, ":s", color="#888", ms=5, label="true contacts with $|i{-}j|>W$ (unreachable)")
    ax2.set_ylabel("frac. true contacts $|i{-}j|>W$", color="#666", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#666")
    ax2.set_ylim(0, max(0.02, max(fb) * 1.4))
    h1, l1 = axa.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axa.legend(h1 + h2, l1 + l2, fontsize=7.6, frameon=False, loc="lower right")
    for s in ("top",):
        axa.spines[s].set_visible(False)

    # (b) E2 real long proteins: ESM-2 stitched vs ProtT5 native (1 forward)
    rows = [r for r in e2_rows if not r.get("skipped")]
    native = {}
    lf = RESULTS / "r4_longctx_prott5.json"
    if lf.is_file():
        native = {r["protein_id"]: r for r in json.loads(lf.read_text())["per_protein"]
                  if not r.get("skipped")}
    labels, st_vals, st_nw, nat_vals = [], [], [], []
    for r in rows:
        wkeys = sorted(int(w) for w in r.get("stitched", {}))
        if not wkeys:
            continue
        bw = wkeys[-1]
        labels.append(f"{r['protein_id']}\n$L$={r['sequence_length']}")
        st_vals.append(r["stitched"][str(bw)]["p"])
        st_nw.append(r["stitched"][str(bw)]["n_windows"])
        nat_vals.append(native.get(r["protein_id"], {}).get("p_long", float("nan")))
    x = np.arange(len(labels)); w = 0.38
    b1 = axb.bar(x - w / 2, st_vals, w, color="#813d9c", label="ESM-2 stitched")
    b2 = axb.bar(x + w / 2, nat_vals, w, color="#1a9641", label="ProtT5 native (1 fwd)")
    for xi, v, nw in zip(x, st_vals, st_nw):
        axb.text(xi - w / 2, v + 0.006, f"{v:.2f}\n{nw}win", ha="center", fontsize=7)
    for xi, v in zip(x, nat_vals):
        if not np.isnan(v):
            axb.text(xi + w / 2, v + 0.006, f"{v:.2f}\n1 fwd", ha="center", fontsize=7)
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=8)
    axb.set_ylabel("P@$L/2$ long")
    allv = [v for v in st_vals + nat_vals if not np.isnan(v)]
    axb.set_ylim(0, max(0.3, max(allv) * 1.45) if allv else 0.3)
    axb.set_title("(b) Beyond the context window: stitch vs native\n(CASP14-FM $L>1024$; no ESM-2 full forward possible)", fontsize=9.5)
    axb.legend(fontsize=7.5, frameon=False, loc="upper right")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    fig.savefig(FIGS / "F_stitch.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGS / "F_stitch.pdf", bbox_inches="tight")
    print(f"wrote {FIGS/'F_stitch.png'}")


def main(argv=None):
    f1 = RESULTS / "r4_stitch_e1_zhang.json"
    f2 = RESULTS / "r4_stitch_e2_long.json"
    if not (f1.is_file() and f2.is_file()):
        print("missing stitch result JSONs", file=sys.stderr); return 1
    e1 = _e1_agg(json.loads(f1.read_text())["per_protein"])
    e2_rows = json.loads(f2.read_text())["per_protein"]

    print(f"\n=== E1 chunking-penalty ablation (Zhang eval-200, N={e1['n']}) ===")
    print(f"  full forward (no tiling): P@L/2 = {e1['full_mean']:.3f} ± {e1['full_sem']:.3f}")
    print(f"  {'W':>5} {'stitched P':>12} {'penalty':>9} {'frac |i-j|>W':>13} {'n_win':>6}")
    for W in e1["Ws"]:
        w = e1["per_w"][W]
        print(f"  {W:>5} {w['p_mean']:>7.3f}±{w['p_sem']:.3f} {e1['full_mean']-w['p_mean']:>9.3f} "
              f"{w['frac_beyond_mean']:>13.3f} {w['n_windows_mean']:>6.1f}")
    print("\n=== E2 real long proteins (L>1024, infeasible in one forward) ===")
    for r in e2_rows:
        if r.get("skipped"):
            print(f"  {r['protein_id']}: SKIPPED ({r.get('reason')})"); continue
        s = ", ".join(f"W={w}: P={d['p']:.3f} ({d['n_windows']}win)"
                      for w, d in sorted(r["stitched"].items(), key=lambda kv: int(kv[0])))
        print(f"  {r['protein_id']} L={r['sequence_length']}: {s}")
    _fig(e1, e2_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
