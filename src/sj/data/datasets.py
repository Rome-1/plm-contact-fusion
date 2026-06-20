"""Shared dataset primitives — Sequence + ContactMap pair, common loader interface.

A dataset loader yields ``ProteinExample`` rows; every variant + baseline
consumes the same shape regardless of whether the data came from Zhang's
1431, CASP14 FM, or pretraining-aware CAMEO. Putting the type here lets
the three loaders share an integration test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# Standard 20 amino acids in single-letter code. ESM-2 tokenizes each as a
# distinct token; sj.eval and Standard CJ both restrict perturbation alts
# to this alphabet (vs the full 33-token ESM vocab which includes special
# tokens like <cls>, <pad>, X, B, Z, ...).
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Cβ-Cβ contact threshold in Å (Cα for glycine).
CONTACT_THRESHOLD_ANGSTROM = 8.0


@dataclass(frozen=True)
class ProteinExample:
    """One protein: (id, sequence, ground-truth contact mask, optional metadata).

    Contact mask is ``(L, L) bool`` with True at positions where Cβ-Cβ
    distance is below CONTACT_THRESHOLD_ANGSTROM. Diagonal and ``|i-j|<6``
    bands are kept True/False as raw — range-split happens at eval time
    in sj.eval.contacts.

    ``valid_residues`` is an optional ``(L,) bool`` array marking sequence
    positions for which ground-truth structural information is available.
    Zhang's 1431 set has resolved-residue coverage that is a strict subset
    of the Gremlin sequence ESM-2 sees as input; the eval pipeline must
    restrict top-L/k ranking to pairs (i, j) where both positions are
    valid (otherwise unresolved positions get ranked against contact-map
    ``False`` and tank precision artificially). When ``None`` the loader
    treats every position as valid (back-compat for splits where every
    residue is resolved).
    """

    protein_id: str
    sequence: str
    contact_map: NDArray[np.bool_]
    metadata: dict[str, str | int | float] | None = None
    valid_residues: NDArray[np.bool_] | None = None

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __post_init__(self) -> None:
        if self.contact_map.shape != (self.length, self.length):
            raise ValueError(
                f"contact_map shape {self.contact_map.shape} != "
                f"(L, L) for L={self.length} ({self.protein_id})"
            )
        if self.valid_residues is not None and self.valid_residues.shape != (self.length,):
            raise ValueError(
                f"valid_residues shape {self.valid_residues.shape} != "
                f"(L,) for L={self.length} ({self.protein_id})"
            )
        for c in self.sequence:
            if c not in STANDARD_AA and c != "X":
                # X (unknown) is tolerated in raw FASTA; alts will skip it.
                # Anything else is a malformed sequence.
                raise ValueError(
                    f"Sequence for {self.protein_id} contains non-standard "
                    f"residue {c!r}; loader must filter or remap."
                )


class DatasetLoader(Protocol):
    """Iterable interface every split implements."""

    name: str # "zhang_1431" | "casp14_fm" | "cameo_pta"

    def __iter__(self): # type: ignore[no-untyped-def]
        """Yield ``ProteinExample`` rows in deterministic order."""
        ...

    def __len__(self) -> int:
        """Number of proteins in the split."""
        ...


def cb_distance_matrix(
    coords: NDArray[np.float64], residue_names: list[str]
) -> NDArray[np.float64]:
    """Compute (L, L) Cβ-Cβ distance matrix from Cβ coords (Cα for glycine).

    ``coords`` is (L, 3); residue index aligns with sequence index. Caller
    is responsible for substituting Cα at glycine positions before calling.
    Output is symmetric and has zero on the diagonal; downstream code masks
    the |i-j|<6 band before range-split top-L/k.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected coords of shape (L, 3), got {coords.shape}")
    if len(residue_names) != coords.shape[0]:
        raise ValueError(f"residue_names length {len(residue_names)} != coords L {coords.shape[0]}")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def contacts_from_distances(distances: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Boolean contact mask from a distance matrix at CONTACT_THRESHOLD_ANGSTROM."""
    return distances < CONTACT_THRESHOLD_ANGSTROM


__all__ = [
    "CONTACT_THRESHOLD_ANGSTROM",
    "STANDARD_AA",
    "DatasetLoader",
    "ProteinExample",
    "cb_distance_matrix",
    "contacts_from_distances",
]


def default_data_root() -> Path:
    """Where dataset files live by default. the results store or local repo."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data"
