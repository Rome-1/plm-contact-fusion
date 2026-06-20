# Provenance: `data/zhang_1431/`

The committed inputs in this directory — `selected_protein.json`,
`full_seq_dict.json`, `ss_dict.json`, and `manifest.sha256` — are the vendored
release files from Zhang et al. 2024's 1,431-chain benchmark, and are
sufficient to regenerate every downstream artifact.

The downstream artifacts (`sequences.fasta`, `contacts/<chain_id>.npz`, a
top-level `manifest.sha256`, `failures.jsonl`) are gitignored to avoid blob
churn. They are a deterministic transform of the inputs: for each selected
chain, fetch the PDB structure, extract the sequence, and compute the
long-range contact map (Cβ–Cβ < 8 Å, Cα for glycine), writing `contacts`
(L, L) bool + `valid_residues` (L,) bool per chain. `sj.data.zhang_1431`
loads these artifacts at evaluation time.

The loader validates each NPZ against the committed `manifest.sha256` for
integrity. `failures.jsonl` records any chain ID that could not be built
(obsolete PDB, renamed chain, etc.).
