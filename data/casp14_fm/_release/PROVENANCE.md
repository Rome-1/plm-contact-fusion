# CASP14 free-modeling target list — vendored release

## v0 vendored 2026-05-09

14 CASP14 FM domain targets, picked for length spread (L=95..2180) over
the long-protein scaling experiment. Source data:

  - Domain assessment table from predictioncenter.org/casp14/domains_summary.cgi
  - Target sequences from predictioncenter.org/download_area/CASP14/sequences/casp14.seq.txt
  - Chain layout for 6VR4 verified via the RCSB REST API
    (data.rcsb.org/rest/v1/core/entry/6VR4): homo-dimer of the same
    2194-residue protein, so all 6vr4-group targets map to chain A.

The 11 6vr4-group targets (T1031, T1033, T1037, T1039, T1040, T1041,
T1042, T1043, T1044) are NON-OVERLAPPING DOMAINS of the same parent
protein (crAss-like phage phi14:2 virion-packaged DNA-dependent RNA
polymerase, "S0A2C3"). The CASP14 organizers extracted each as a
separate prediction problem of the corresponding sub-region. We give
the predictor the CASP14 sub-region sequence verbatim; the build
runs BLOSUM62 alignment against the full 6VR4 chain A residue-by-residue
to find the matching region and extract Cβ contacts for that span. The
valid_residues mask flags any positions in our sequence that don't have
a resolved PDB residue.

T1044 is the full 2180-residue parent — included here as the long-protein
test point. ESM-2-650M was trained at L≤1024, so this is OOD for the
model; the sequence_length_cap in CJConfig (default 2000) needs bumping
to dispatch CJ on T1044.

Targets NOT in v0 (deferred to v0.1):

- FM-classified targets without a deposited PDB at assessment time:
  T1047s1, T1061-D2, T1090, T1093-D1, T1093-D3, T1094-D2, T1096-D1,
  T1096-D2. Some of these may now have post-CASP14 PDB deposits;
  worth a one-shot RCSB lookup pass before v0.1.
- FM/TBM-borderline targets (T1035, T1038-D2, T1046s1, T1052-D3,
  T1053-D1, etc.) — excluded for honesty about pure-FM classification.

## What goes here

This directory holds the canonical CASP14 FM (and optionally TBM) target
list the build consumes.

Required file: `targets.json` — a JSON array of objects, one per target:

```json
[
  {"target_id": "T1024-D1", "pdb_chain": "6T1ZA", "sequence": "MGSSH..."},
  {"target_id": "T1026-D1", "pdb_chain": "6S44A", "sequence": "MASTV..."},
  ...
]
```

Field semantics:

- `target_id` — CASP14 target ID, e.g. `T1024-D1` (target T1024, domain 1).
  Used as the unique key in our build outputs (`contacts/<target_id>.npz`,
  FASTA records).
- `pdb_chain` — the PDB chain ID for the deposited native structure
  (`<4-char PDB ID><chain letter>`). Used to fetch the deposited native
  structure from `files.rcsb.org` and compute the contact map.
- `sequence` — the official CASP14 target sequence (after stripping
  multi-domain breaks if `target_id` includes a domain suffix).
  Length must match the residue range that the named PDB chain covers.

## How to populate (one-time manual step)

Until this is automated, populate `targets.json` from:

  https://predictioncenter.org/casp14/results.cgi  →  "Free modeling"

For each FM target listed there:

1. Note the target ID (e.g. `T1024-D1`).
2. Look up the deposited native PDB ID + chain (usually a single
   4-letter PDB ID + chain letter; check the "PDB code" column).
3. Pull the official target sequence from the per-target page (the
   "Target sequence" FASTA).
4. Add an entry to `targets.json` with the three fields above.

CASP14 has roughly 36 FM targets across 71 evaluation domains. Some FM
targets do not have a deposited PDB at all (T1xxx where the native is
unreleased) — skip those; they cannot serve as ground truth.

From the populated `targets.json`, the build is a deterministic, network-only
transform: for each target it fetches the named PDB chain from RCSB, extracts
the sequence, and writes `contacts/<target_id>.npz` (Cβ–Cβ < 8 Å long-range
contact map + valid-residue mask). `sj.data.casp14_fm` loads these at
evaluation time.

## Why the vendored target list

CASP14's per-target HTML pages aren't stable URLs and the official
`targets.summary` text file changes formatting between CASP rounds.
Vendoring the (target_id, pdb_chain, sequence) triple here pins the
benchmark exactly and lets the build be a deterministic
network-only operation against RCSB.
