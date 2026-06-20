"""CASP14 free-modeling + TBM split (54 TBM + 36 FM, 90 total).

Loader for the trRosetta CASP14 release. Same on-disk layout convention as
zhang_1431.py for cross-loader uniformity:

    data/casp14_fm/sequences.fasta
    data/casp14_fm/contacts/<id>.npz ``contacts`` (L,L) bool, 8 Å Cβ-Cβ
    data/casp14_fm/manifest.sha256 per-protein NPZ hashes

Dedup vs Zhang-1431 happens at build time (to be added under — same shape as the dataset build script): targets
that share a sequence with any Zhang 1431 entry are dropped before the
loader sees them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sj.data.datasets import (
    DatasetLoader,
    ProteinExample,
    default_data_root,
)

DATASET_DIR_NAME = "casp14_fm"
SEQUENCES_FILENAME = "sequences.fasta"
CONTACTS_DIRNAME = "contacts"
MANIFEST_FILENAME = "manifest.sha256"


@dataclass
class CASP14FMLoader(DatasetLoader):
    """Iterates CASP14 FM + TBM in deterministic ID order."""

    data_root: Path
    name: str = "casp14_fm"

    def __post_init__(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"CASP14 FM data root not found: {self.data_root}. "
                "Populate data/casp14_fm/ with the built dataset."
            )
        seqs = self.data_root / SEQUENCES_FILENAME
        if not seqs.is_file():
            raise FileNotFoundError(f"Missing FASTA: {seqs}")
        self._sequences = _parse_fasta(seqs)
        self._contacts_dir = self.data_root / CONTACTS_DIRNAME
        if not self._contacts_dir.is_dir():
            raise FileNotFoundError(f"Missing contacts dir: {self._contacts_dir}")

    def __iter__(self) -> Iterator[ProteinExample]:
        for protein_id in sorted(self._sequences):
            seq = self._sequences[protein_id]
            npz_path = self._contacts_dir / f"{protein_id}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing contacts for {protein_id}: expected {npz_path}")
            with np.load(npz_path) as data:
                contacts = data["contacts"].astype(bool)
                # valid_residues is optional in the on-disk npz for back-compat
                # with older builds that pre-date the resolved-subset mask.
                valid_residues = (
                    data["valid_residues"].astype(bool) if "valid_residues" in data.files else None
                )
            yield ProteinExample(
                protein_id=protein_id,
                sequence=seq,
                contact_map=contacts,
                valid_residues=valid_residues,
                metadata={"split": self.name, "category": _category_from_id(protein_id)},
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


def _category_from_id(protein_id: str) -> str:
    """CASP14 uses target IDs like ``T1024-D1`` (FM) or ``T1024-TBM`` — best-effort tag."""
    if "FM" in protein_id.upper():
        return "FM"
    if "TBM" in protein_id.upper():
        return "TBM"
    return "unknown"


def default_casp14_fm_root() -> Path:
    return default_data_root() / DATASET_DIR_NAME


__all__ = [
    "CONTACTS_DIRNAME",
    "DATASET_DIR_NAME",
    "MANIFEST_FILENAME",
    "SEQUENCES_FILENAME",
    "CASP14FMLoader",
    "default_casp14_fm_root",
]
