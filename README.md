# plm-contact-fusion

**Protein contacts are already in the attention: a single-forward-pass alternative to the Categorical Jacobian.**

Code and result tables for the paper. The method: the long-range contact signal a
protein language model encodes is concentrated in a small, identifiable cluster of
attention heads. Averaging the top-*K* contact-relevant heads — selected once from a
ranking on as few as 10 labeled proteins, with no per-pair weight fitting — recovers
residue–residue contacts in **one forward pass** and beats the Categorical Jacobian
of [Zhang et al. 2024](https://www.pnas.org/doi/10.1073/pnas.2406285121) (which needs
~19·L forward passes) on every bidirectional PLM tested, on data filtered to remove
sequence-level pretraining overlap.

The paper is in [`paper/paper.pdf`](paper/paper.pdf).

## Key results

- **Leakage-clean primary test** (CAMEO-PTA25, Hermann-2024 filter, *N*=29):
  fusion beats the Categorical Jacobian by +9 pp on ESM-2-650M (*p*<0.001), with the
  within-model margin reproducing across ESM-2-3B, ProtT5-XL, ESM-1b, and the recent
  ESM Cambrian backbone.
- **Sharper, not just better** (Fig 2): fusion concentrates its score mass on a few
  pairs (Gini 0.26 vs 0.12–0.17 for the perturbation readouts) *and* is the most
  accurate — it commits to a handful of contacts and is right.
- **~150–1600× cheaper, measured** (Fig 3): on identical A100-80GB hardware the single
  fusion forward runs from ~150× (ESM-2-35M) to ~1600× (ESM-2-3B) faster than the
  Categorical Jacobian's ~19·L perturbations, the gap widening with model size — the
  irreducible forward-count ratio, not a batching artifact.
- **Selection, not averaging, is the operative step**: at a matched 50-protein label
  budget, the unweighted mean ties a supervised L1 logistic regression on the same heads.
- **representation-CJ**, an output-head-independent generalization of the Jacobian,
  extends it to architectures without a masked-LM head (ProtT5, AMPLIFY); where both are
  defined it agrees with the Categorical Jacobian (per-protein Pearson *r*≈0.95), and an
  explicit *W*ᵀ*W*-metric readout localizes that agreement to the readout metric (App K).
- **Scales past the context window** (App Q): longer chains are read by tiling and
  stitching overlapping windows (near-lossless) or by a longer-context backbone (ProtT5,
  ESM-C), recovering contacts on two real *L*>1024 targets in ~1 s of forwards.
- Both methods **collapse on causal LMs** (ProGen2), a scope boundary consistent with
  bidirectional pretraining mattering for attention-encoded pair structure.

## Repository layout

```
src/sj/            method + evaluation library (head fusion, metrics, data loaders, model loader)
src/baselines/     Categorical Jacobian (Zhang 2024), representation-CJ, Rao 2021 attention readout
scripts/           experiment runners (timing, confidence, repr/WtW agreement, stitching) + figure/appendix-table generation
paper/             paper.tex, references.bib, figures/, paper.pdf, appendix_data.json
data/              dataset build inputs + provenance (Zhang-1431, CASP14-FM, CAMEO-PTA25)
checkpoints/       SHA256 manifests for the model weights (integrity gate)
```

The method itself is small and self-contained:

- `src/sj/probes/head_fusion.py` — the top-*K* head-fusion operators (naive mean + alternatives).
- `src/sj/eval/contacts.py` — top-*L/k* short/medium/long-range precision and APC.
- `src/baselines/zhang_2024_cj.py`, `representation_cj.py` — the Categorical Jacobian and its
  hidden-state generalization.
- `src/sj/probes/model_adapters.py`, `src/sj/model.py` — per-architecture attention/logit
  extraction with a SHA256 weight-integrity gate.

## Reproducing

The paper is self-contained: `paper/paper.pdf` builds from `paper/paper.tex` with the
committed figures (`paper/figures/`) and summary tables (`paper/appendix_data.json`).

The experiments ran on A100-80GB GPUs. The r4 experiment scripts (`scripts/r4_*.py`)
define the per-protein timing, confidence, repr/WtW/logit metric-agreement, and
stitching runs; the matching `*_analysis.py` / `*_plot.py` scripts, together with
`scripts/paper_figures.py` and `scripts/build_appendix.py`, rebuild the figures and
`appendix_data.json` from per-cell result JSONs. The scoring, fusion, and evaluation
code above runs on any GPU (or CPU for small proteins) to score a single protein. The
bulky per-protein result JSONs and `.npz` score maps are not bundled — the committed
`appendix_data.json` carries the per-cell means and bootstrap intervals.

Model weights load through a SHA256-manifest gate (`checkpoints/*.sha256`) so a
reproducer can verify byte-identical checkpoints before any forward pass.

## Citation

```bibtex
@misc{thorstenson2026plmcontactfusion,
  title  = {Protein contacts are already in the attention: a single-forward-pass alternative to the Categorical Jacobian},
  author = {Thorstenson, Rome},
  year   = {2026},
  note   = {Preprint},
}
```

## License

MIT — see [LICENSE](LICENSE).
