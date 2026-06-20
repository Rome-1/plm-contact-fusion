"""Zhang 2024 1,431-protein contact-prediction benchmark loader.

Zhang et al. PNAS 2024 (doi: 10.1073/pnas.2406285121) released a curated
1,431-protein subset for evaluating PLM-derived contact methods. Each
example carries a sequence and a Cβ-Cβ contact mask at 8 Å.

Source files (committed to the repo OR loaded from the results store):

  data/zhang_1431/sequences.fasta one record per protein, id = chain code
  data/zhang_1431/contacts/<id>.npz ``contacts`` (L,L) bool, 8 Å Cβ-Cβ

The sequences-FASTA + per-protein NPZ split is intentional: contact maps
are deterministic functions of PDB structure but PDB structures occasionally
update in place. Splitting them out lets a SHA256-style integrity story
extend to ground truth: a manifest under ``data/zhang_1431/manifest.sha256``
holds the SHA256 of every NPZ so a reviewer can verify that the contacts
this paper trains the eval against are bit-identical to the ones cited.

the dataset build.py``,
to be added under ) downloads + parses the PDB structures.
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

# Subdirectory under ``data/`` for this split.
DATASET_DIR_NAME = "zhang_1431"
SEQUENCES_FILENAME = "sequences.fasta"
CONTACTS_DIRNAME = "contacts"
MANIFEST_FILENAME = "manifest.sha256"


@dataclass
class Zhang1431Loader(DatasetLoader):
    """Iterates the Zhang 2024 1,431-protein set in deterministic ID order."""

    data_root: Path
    name: str = "zhang_1431"

    def __post_init__(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"Zhang 1431 data root not found: {self.data_root}. "
                "Run scripts/the dataset build script to populate."
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
            yield self._load_one(protein_id)

    def __len__(self) -> int:
        return len(self._sequences)

    def _load_one(self, protein_id: str) -> ProteinExample:
        seq = self._sequences[protein_id]
        npz_path = self._contacts_dir / f"{protein_id}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing contacts for {protein_id}: expected {npz_path}")
        with np.load(npz_path) as data:
            contacts = data["contacts"].astype(bool)
            # valid_residues is optional — older NPZs predate the
            # PDB-resolved-only eval protocol and treat every position
            # as valid. New NPZs from scripts/the dataset build script
            # always include it.
            valid = data["valid_residues"].astype(bool) if "valid_residues" in data.files else None
        return ProteinExample(
            protein_id=protein_id,
            sequence=seq,
            contact_map=contacts,
            metadata={"split": self.name},
            valid_residues=valid,
        )


def _parse_fasta(path: Path) -> dict[str, str]:
    """Minimal FASTA reader. Header is up to the first whitespace.

    Sufficient for Zhang's flat single-line release format. We avoid pulling
    Biopython into the loader so this module stays importable in the
    minimal CPU-only test environment.
    """
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


def default_zhang_1431_root() -> Path:
    """Return the canonical local path for Zhang 1431 data."""
    return default_data_root() / DATASET_DIR_NAME


__all__ = [
    "CONTACTS_DIRNAME",
    "DATASET_DIR_NAME",
    "MANIFEST_FILENAME",
    "SEQUENCES_FILENAME",
    "Zhang1431Loader",
    "default_zhang_1431_root",
]
