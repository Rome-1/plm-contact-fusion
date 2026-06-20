"""Contact-map metrics shared across all variants and baselines.

Contains the math that does NOT depend on the model: APC correction,
range-split top-L/k precision against ground truth, and bootstrap CI on the
delta between two contact-map sets. Variants and baselines feed in their
score maps; this file decides what the headline numbers are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# brief: standard short/medium/long range cutoffs in |i-j|.
SHORT_RANGE = (6, 11)
MEDIUM_RANGE = (12, 23)
LONG_RANGE_MIN = 24

# Finer 5-band breakdown for long-protein analysis. Same start (≥6) but
# splits the long band into long / very_long / ultra_long so contact-prediction
# quality can be tracked against separation. The narrow bands matter especially
# at L > 1000 where "long range ≥24" alone hides the actual separation profile.
VERY_SHORT_RANGE = (6, 11) # alias of SHORT_RANGE for naming consistency
SHORT_FINE_RANGE = (12, 23) # alias of MEDIUM_RANGE
MEDIUM_FINE_RANGE = (24, 47)
LONG_FINE_RANGE = (48, 95)
ULTRA_LONG_RANGE_MIN = 96

RangeName = Literal[
    "short",
    "medium",
    "long", # historical 3-band split
    "very_short",
    "short_fine",
    "medium_fine",
    "long_fine",
    "ultra_long", # 5-band fine split
]


@dataclass(frozen=True)
class TopLkResult:
    """Top-L/k precision broken out by range, plus the count of true positives."""

    range_name: RangeName
    k: int # divisor of L (e.g. 2 for top-L/2)
    precision: float
    true_positives: int
    predicted: int


def apc_correction(score_map: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average product correction (Dunn et al. 2008) on a square score map.

    APC subtracts ``mean(row_i) * mean(col_j) / mean(all)`` from each ``S_ij``.
    Standard contact-prediction post-processing — strips out background
    column/row biases so true direct couplings rise to the top.

    Numerical guard: if the global mean is exactly zero (e.g. an empty input
    or all-zero scores), returns the input unchanged rather than dividing by
    zero, which is the conventional behavior in the contact-prediction
    literature (no signal → no correction).
    """
    if score_map.ndim != 2 or score_map.shape[0] != score_map.shape[1]:
        raise ValueError(f"APC expects a square 2D map, got shape {score_map.shape}")
    row_mean = score_map.mean(axis=1, keepdims=True)
    col_mean = score_map.mean(axis=0, keepdims=True)
    total_mean = score_map.mean()
    if total_mean == 0.0:
        return score_map.copy()
    correction = (row_mean * col_mean) / total_mean
    return score_map - correction


def _range_mask(length: int, range_name: RangeName) -> NDArray[np.bool_]:
    """Bool mask of (L, L) selecting the |i-j| band for ``range_name``."""
    i_idx, j_idx = np.indices((length, length))
    sep = np.abs(i_idx - j_idx)
    if range_name == "short":
        lo, hi = SHORT_RANGE
        return (sep >= lo) & (sep <= hi)
    if range_name == "medium":
        lo, hi = MEDIUM_RANGE
        return (sep >= lo) & (sep <= hi)
    if range_name == "long":
        return sep >= LONG_RANGE_MIN
    # Fine 5-band split:
    if range_name == "very_short":
        lo, hi = VERY_SHORT_RANGE
        return (sep >= lo) & (sep <= hi)
    if range_name == "short_fine":
        lo, hi = SHORT_FINE_RANGE
        return (sep >= lo) & (sep <= hi)
    if range_name == "medium_fine":
        lo, hi = MEDIUM_FINE_RANGE
        return (sep >= lo) & (sep <= hi)
    if range_name == "long_fine":
        lo, hi = LONG_FINE_RANGE
        return (sep >= lo) & (sep <= hi)
    if range_name == "ultra_long":
        return sep >= ULTRA_LONG_RANGE_MIN
    raise ValueError(f"Unknown range_name: {range_name!r}")


def top_lk_precision(
    score_map: NDArray[np.float64],
    ground_truth: NDArray[np.bool_],
    *,
    range_name: RangeName,
    k: int = 2,
    valid_residues: NDArray[np.bool_] | None = None,
) -> TopLkResult:
    """Top-L/k precision in one range band.

    Convention: take the upper triangle only (symmetric maps), restrict to
    the requested |i-j| band, rank by score descending, take top L/k pairs,
    fraction of those that are true contacts.

    ``valid_residues`` is an optional ``(L,) bool`` mask of sequence
    positions for which ground truth is meaningful (e.g., PDB-resolved
    residues for Zhang's 1431 set, where Gremlin sequences include
    residues with no structural coordinates). Pairs ``(i, j)`` with
    either position outside the mask are excluded from both ranking and
    scoring; this matches Zhang's protocol of evaluating only on
    PDB-resolved positions. ``L`` in the top-L/k denominator continues
    to refer to the full input sequence length (not the resolved-only
    count) so the "headline" precision number stays comparable across
    proteins with different resolved-residue coverage.
    """
    if score_map.shape != ground_truth.shape:
        raise ValueError(f"score map shape {score_map.shape} != ground truth {ground_truth.shape}")
    if score_map.ndim != 2 or score_map.shape[0] != score_map.shape[1]:
        raise ValueError(f"Expected square 2D map, got {score_map.shape}")
    length = score_map.shape[0]
    upper = np.triu(np.ones_like(score_map, dtype=bool), k=1)
    mask = upper & _range_mask(length, range_name)
    if valid_residues is not None:
        if valid_residues.shape != (length,):
            raise ValueError(f"valid_residues shape {valid_residues.shape} != ({length},)")
        mask = mask & valid_residues[:, None] & valid_residues[None, :]
    flat_scores = score_map[mask]
    flat_truth = ground_truth[mask]
    if flat_scores.size == 0:
        return TopLkResult(range_name=range_name, k=k, precision=0.0, true_positives=0, predicted=0)
    top_n = max(1, length // k)
    top_n = min(top_n, flat_scores.size)
    # argpartition for top-N, then gather truth at those positions.
    top_idx = np.argpartition(-flat_scores, top_n - 1)[:top_n]
    tp = int(flat_truth[top_idx].sum())
    return TopLkResult(
        range_name=range_name,
        k=k,
        precision=tp / top_n,
        true_positives=tp,
        predicted=top_n,
    )


def bootstrap_delta_ci(
    deltas: NDArray[np.float64],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of a per-protein delta vector.

    the design requires a 95% CI on the Zhang reproduction delta and on every
    variant-vs-baseline delta. This is the shared engine; callers build the
    per-protein delta vector themselves (e.g. variant top-L/2 minus Zhang
    top-L/2 paired by protein_id).
    """
    if deltas.ndim != 1:
        raise ValueError(f"deltas must be 1D, got shape {deltas.shape}")
    if deltas.size == 0:
        raise ValueError("deltas is empty; cannot bootstrap")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    rng = rng if rng is not None else np.random.default_rng()
    n = deltas.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = deltas[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(resampled_means, [alpha, 1 - alpha])
    return float(lo), float(hi)


__all__ = [
    "LONG_FINE_RANGE",
    "LONG_RANGE_MIN",
    "MEDIUM_FINE_RANGE",
    "MEDIUM_RANGE",
    "SHORT_FINE_RANGE",
    "SHORT_RANGE",
    "ULTRA_LONG_RANGE_MIN",
    "VERY_SHORT_RANGE",
    "TopLkResult",
    "apc_correction",
    "bootstrap_delta_ci",
    "top_lk_precision",
]
