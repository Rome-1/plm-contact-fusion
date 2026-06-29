"""r4 Job 2 analysis: summarize repr/WtW/logit-CJ pair-level agreement.

Reads results/r4_cj_maps_<variant>_<dataset>.json (per-protein scalars from
cj_maps_batch) and emits paper-ready aggregates:

  - mean precision per readout (sanity: logit ~ cached Standard CJ)
  - mean/median top-L/2 long Jaccard and Pearson(long pairs) for each pair
    {logit~repr, logit~wtw, repr~wtw}
  - failure cases: proteins with lowest logit~repr agreement

Theory read (see paper repr-CJ theory paragraph):
  repr~wtw high  => W^T W near-isotropic on contact-carrying directions
  wtw~logit high => dense+gelu+layernorm nonlinearity contributes little
Pure I/O; run after r4_repr_cj_maps.py lands its JSONs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _summarize(path: Path) -> dict:
    d = json.load(path.open())
    scored = [r for r in d["per_protein"] if not r.get("skipped")]
    if not scored:
        return {}

    def col(group, sub):
        return np.array([r[group][sub] for r in scored if not np.isnan(r[group][sub])])

    pairs = ["logit_repr", "logit_wtw", "repr_wtw"]
    out = {
        "variant": d["variant"],
        "dataset": d["dataset"],
        "n": len(scored),
        "precision": {k: float(np.mean([r["precision"][k] for r in scored]))
                      for k in ("logit", "repr", "wtw")},
        "jaccard_mean": {p: float(col("jaccard_topl2_long", p).mean()) for p in pairs},
        "jaccard_median": {p: float(np.median(col("jaccard_topl2_long", p))) for p in pairs},
        "pearson_mean": {p: float(col("pearson_long_pairs", p).mean()) for p in pairs},
    }
    # failure cases: lowest logit~repr Jaccard
    ranked = sorted(scored, key=lambda r: r["jaccard_topl2_long"]["logit_repr"])
    out["worst_logit_repr"] = [
        {"protein_id": r["protein_id"], "L": r["sequence_length"],
         "jac_logit_repr": round(r["jaccard_topl2_long"]["logit_repr"], 3),
         "jac_repr_wtw": round(r["jaccard_topl2_long"]["repr_wtw"], 3)}
        for r in ranked[:5]
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    files = sorted(
        f for f in RESULTS.glob("r4_cj_maps_*.json") if f.name != "r4_cj_maps_summary.json"
    )
    if not files:
        print("no r4_cj_maps_*.json yet", file=sys.stderr)
        return 1
    summaries = []
    for f in files:
        s = _summarize(f)
        if s:
            summaries.append(s)
    for s in summaries:
        print(f"\n=== {s['variant']} / {s['dataset']}  (N={s['n']}) ===")
        print(f"  precision:  logit={s['precision']['logit']:.3f}  "
              f"repr={s['precision']['repr']:.3f}  wtw={s['precision']['wtw']:.3f}")
        print(f"  top-L/2 long Jaccard (mean):  "
              f"logit~repr={s['jaccard_mean']['logit_repr']:.3f}  "
              f"logit~wtw={s['jaccard_mean']['logit_wtw']:.3f}  "
              f"repr~wtw={s['jaccard_mean']['repr_wtw']:.3f}")
        print(f"  long-pair Pearson (mean):     "
              f"logit~repr={s['pearson_mean']['logit_repr']:.3f}  "
              f"logit~wtw={s['pearson_mean']['logit_wtw']:.3f}  "
              f"repr~wtw={s['pearson_mean']['repr_wtw']:.3f}")
        print(f"  worst logit~repr proteins: "
              + ", ".join(f"{w['protein_id']}(L{w['L']},J{w['jac_logit_repr']})"
                          for w in s['worst_logit_repr']))
    out = RESULTS / "r4_cj_maps_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
