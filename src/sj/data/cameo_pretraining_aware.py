"""Pretraining-aware CAMEO split — Hermann et al. 2024 leakage filter.

Ground-truth + filter protocol by design:

  1. Pull CAMEO targets with deposition date >= 2022-09-01 (post ESM-2
     UniRef50 March-2018 cutoff + a generous gap).
  2. For each candidate target, query MMseqs2 50% similarity against the
     UniRef50-2018-03 cluster representatives. If any cluster representative
     matches, drop the target (Hermann 2024 quantified ~11.1% leakage
     distortion when this filter is omitted).
  3. Build the standard ``sequences.fasta + contacts/*.npz`` shape so the
     loader matches Zhang/CASP.

Two paths to populate the data root, in order of preference:

  - **Hermann GitLab releases** (preferred): if Hermann et al. published
    ready-to-use splits and they're compatible with our shape, fetch +
    convert. the leakage filter described in data/cameo_pta provenance does this.
  - **Reproduce filter**: download CAMEO targets directly + run MMseqs2
    locally against UniRef50-2018-03. Operator-side; needs MMseqs2 installed.

Until ``data/cameo_pta/`` exists, the loader raises FileNotFoundError —
the same halt-loud-on-missing pattern Zhang and CASP loaders use.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sj.data.datasets import (
    DatasetLoader,
    ProteinExample,
    default_data_root,
)

logger = logging.getLogger(__name__)

DATASET_DIR_NAME = "cameo_pta"
SEQUENCES_FILENAME = "sequences.fasta"
CONTACTS_DIRNAME = "contacts"
MANIFEST_FILENAME = "manifest.sha256"
FILTER_LOG_FILENAME = "hermann_filter.json"

# Hermann et al. 2024: UniRef50 cluster representatives at MMseqs2
# 50% similarity. Constants pinned here so the build script and the
# loader's audit-log assertion agree.
#
# UNIREF50_VERSION lists acceptable releases (in priority order). 2018-03 was
# the original target (ESM-1b's pretraining release); 2021_04 is what ESM-2
# actually used; "current" is what the dataset build downloads in
# practice (live archive, strict superset → more conservative filter). The
# loader accepts any of these.
UNIREF50_VERSION = ("2018-03", "2021_04", "current")
MMSEQS2_SIMILARITY_THRESHOLD = 0.50
DEPOSITION_DATE_CUTOFF = "2022-09-01"


@dataclass
class CAMEOPretrainingAwareLoader(DatasetLoader):
    """Iterates the Hermann-filtered CAMEO split.

    Verifies the audit-log JSON at load time so a misconfigured build
    cannot silently disable the leakage filter — the loader fails loud
    if FILTER_LOG_FILENAME is missing or claims a different similarity
    threshold than ``MMSEQS2_SIMILARITY_THRESHOLD``.
    """

    data_root: Path
    name: str = "cameo_pta"

    def __post_init__(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"CAMEO pretraining-aware data root not found: {self.data_root}. "
                "Populate data/cameo_pta/ with the built dataset."
            )
        seqs = self.data_root / SEQUENCES_FILENAME
        if not seqs.is_file():
            raise FileNotFoundError(f"Missing FASTA: {seqs}")
        self._sequences = _parse_fasta(seqs)
        self._contacts_dir = self.data_root / CONTACTS_DIRNAME
        if not self._contacts_dir.is_dir():
            raise FileNotFoundError(f"Missing contacts dir: {self._contacts_dir}")
        # Audit-log presence + threshold check (leakage-filter audit).
        log_path = self.data_root / FILTER_LOG_FILENAME
        if not log_path.is_file():
            raise FileNotFoundError(
                f"Missing Hermann filter audit log: {log_path}. "
                "Build script must record the filter's threshold + UniRef50 version."
            )
        import json

        log = json.loads(log_path.read_text())
        threshold = log.get("mmseqs2_similarity_threshold")
        if threshold is None:
            threshold = log.get("similarity_threshold")
        if threshold != MMSEQS2_SIMILARITY_THRESHOLD:
            raise ValueError(
                f"Hermann filter audit log records similarity threshold {threshold}, "
                f"expected {MMSEQS2_SIMILARITY_THRESHOLD}. Refusing to proceed."
            )
        version = log.get("uniref50_version")
        if version is None:
            version = log.get("uniref_version")
        if version not in UNIREF50_VERSION:
            raise ValueError(
                f"Hermann filter audit log records UniRef50 version {version!r}, "
                f"expected one of {UNIREF50_VERSION!r}. Refusing to proceed."
            )

    def __iter__(self) -> Iterator[ProteinExample]:
        for protein_id in sorted(self._sequences):
            seq = self._sequences[protein_id]
            npz_path = self._contacts_dir / f"{protein_id}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing contacts for {protein_id}: expected {npz_path}")
            with np.load(npz_path) as data:
                contacts = data["contacts"].astype(bool)
                valid_residues = (
                    data["valid_residues"].astype(bool) if "valid_residues" in data.files else None
                )
            yield ProteinExample(
                protein_id=protein_id,
                sequence=seq,
                contact_map=contacts,
                valid_residues=valid_residues,
                metadata={"split": self.name},
            )

    def __len__(self) -> int:
        return len(self._sequences)


def _parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records[current_id] = "".join(chunks)
            current_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if current_id is not None:
        records[current_id] = "".join(chunks)
    if not records:
        raise ValueError(f"FASTA at {path} contained no records")
    return records


def default_cameo_pta_root() -> Path:
    return default_data_root() / DATASET_DIR_NAME


def cameo_pta_split(which: str, *, data_root: Path | None = None) -> list:
    """Disjoint select/eval split of the leakage-filtered CAMEO-PTA set for the
    end-to-end leakage-clean comparison (r5 review: replace the old 50/50
    half_a/half_b split, which left only N=19 for evaluation).

    Head selection needs only ~10 labeled proteins, so we select on a
    length-stratified 10 of the L>=75-eligible targets and evaluate on the
    remaining L>=75 targets (disjoint), which roughly doubles the clean eval N.
    ``which`` is ``"select10"`` or ``"eval"``. L<75 targets are excluded from
    both (no long-range pairs; selection ranks by long-range precision).
    """
    root = data_root if data_root is not None else default_cameo_pta_root()
    full = list(CAMEOPretrainingAwareLoader(data_root=root))
    eligible = sorted((p for p in full if p.length >= 75), key=lambda p: (p.length, p.protein_id))
    n = len(eligible)
    sel_idx = set(range(n)) if n <= 10 else {round(i * (n - 1) / 9) for i in range(10)}
    select_ids = {eligible[i].protein_id for i in sorted(sel_idx)}
    if which == "select10":
        return [p for p in eligible if p.protein_id in select_ids]
    if which == "eval":
        return [p for p in eligible if p.protein_id not in select_ids]
    raise ValueError(f"unknown cameo_pta split {which!r} (expected 'select10' or 'eval')")


__all__ = [
    "CONTACTS_DIRNAME",
    "DATASET_DIR_NAME",
    "DEPOSITION_DATE_CUTOFF",
    "FILTER_LOG_FILENAME",
    "MANIFEST_FILENAME",
    "MMSEQS2_SIMILARITY_THRESHOLD",
    "SEQUENCES_FILENAME",
    "UNIREF50_VERSION",
    "CAMEOPretrainingAwareLoader",
    "default_cameo_pta_root",
]
