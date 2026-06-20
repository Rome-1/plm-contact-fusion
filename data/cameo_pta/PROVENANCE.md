# Provenance: `data/cameo_pta/` (CAMEO-PTA25, leakage-clean split)

CAMEO-PTA25 is the paper's primary, leakage-clean evaluation set: single-chain
CAMEO targets deposited on or after 2022-09-01 (post-dating ESM-2's UniRef50
2021_04 pretraining release), filtered against current UniRef50 at 50% MMseqs2
identity / 80% query coverage following the leakage protocol of Hermann et al.
2024. Of 553 post-cutoff targets, 56 pass the filter; 39 retain L ≥ 75 (the
long-range eligibility cutoff), of which 29 form the leakage-clean evaluation
half (the other 10 are the disjoint head-selection set).

Files:

- `sequences.fasta` — the surviving target sequences.
- `contacts/<target_id>.npz` — `contacts` (L, L) bool (Cα–Cα < 8 Å long-range)
  + `valid_residues` (L,) bool.
- `hermann_filter.json` — the audit log of the leakage filter: UniRef version,
  identity threshold, deposition cutoff, per-target keep/drop decisions, and the
  surviving target IDs.

`sj.data.cameo_pretraining_aware` loads these at evaluation time and verifies the
filter audit log is present. See Appendix A of the paper for the conservative
current-UniRef50 substitution and a 2024_03-release sanity check.
