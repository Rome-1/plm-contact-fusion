# Zhang 2024 release files (vendored)

Source: <https://github.com/zzhangzzhang/pLMs-interpretability/tree/main/data>

Vendored 2026-05-05 from main branch. SHA256s in `manifest.sha256` are the
authoritative integrity check; the dataset build refuses to proceed if the
local files don't match the manifest. See `REBUILD.md` for how the downstream
artifacts are derived from these inputs.

| File | Size | Purpose |
|---|---|---|
| `selected_protein.json` | 12.9 KB | Canonical 1431-protein chain ID list (Zhang's curated subset of Gremlin) |
| `full_seq_dict.json`    | 633 KB  | Gremlin-derived sequences keyed by chain ID (2170 total; 1431 used) |
| `ss_dict.json`          | 485 KB  | Secondary-structure annotations per chain (informational; not used in v0 build) |

These files carry Zhang et al.'s release license; see the upstream repo.
