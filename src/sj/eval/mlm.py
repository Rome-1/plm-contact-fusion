"""MLM perplexity aggregates.

Detects ESM-2 out-of-distribution behavior at long L: at the edge of the
training context (L > 1024 for ESM-2-650M) the per-position softmax loses
calibration, so MLM loss at the native amino acid at every position j
inflates. Tracking a few aggregates per protein shows which length regime
the model is still functional in.

Aggregates emitted (one number each, all in nats):
  - ``mean_all_positions``: mean over every valid position j
  - ``mean_long_range_non_contacts``: mean restricted to j that has
    NO long-range contact partner i (|i-j| ≥ 24, contact_map[i, j] False)
  - ``mean_long_range_contacts``: mean over j that IS a long-range
    contact partner of some i

Inputs are taken AS-IS (caller is responsible for applying any softmax
+ log over the full vocab, or just the standard-AA subset). Both
inputs and outputs use natural log.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_mlm_loss_aggregates(
    log_probs_per_position: NDArray[np.float32],
    native_token_ids: NDArray[np.int64],
    contact_map: NDArray[np.bool_],
    valid_residues: NDArray[np.bool_] | None = None,
    *,
    seq_sep: int = 24,
) -> dict[str, float]:
    """Return MLM-loss aggregates from one clean forward.

    Parameters
    ----------
    log_probs_per_position
        ``(L, V)`` float array of log-softmax over the model's full
        vocab at every sequence position.
    native_token_ids
        ``(L,)`` int array of the native token ID at each position.
    contact_map
        ``(L, L)`` bool — ground truth contacts (already restricted to
        |i-j| ≥ 6 by the loader).
    valid_residues
        Optional ``(L,)`` bool of positions for which ground truth is
        meaningful. Restricts every aggregate.
    seq_sep
        Long-range cutoff in |i-j|. Default 24 matches LONG_RANGE_MIN.
    """
    L = native_token_ids.shape[0]
    if log_probs_per_position.shape[0] != L:
        raise ValueError(
            f"log_probs ({log_probs_per_position.shape}) vs native_token_ids ({native_token_ids.shape}) length mismatch"
        )
    if contact_map.shape != (L, L):
        raise ValueError(f"contact_map {contact_map.shape} != ({L}, {L})")
    if valid_residues is None:
        valid_residues = np.ones(L, dtype=bool)
    elif valid_residues.shape != (L,):
        raise ValueError(f"valid_residues {valid_residues.shape} != ({L},)")

    # -log p(native at j | clean input)
    nll_per_j = -log_probs_per_position[np.arange(L), native_token_ids]

    # Long-range contact mask per j: any i with |i-j| ≥ seq_sep s.t. contact_map[i, j]
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    long_pairs = (sep >= seq_sep) & contact_map
    j_has_long_contact = long_pairs.any(axis=0)
    j_has_long_non_contact = (sep >= seq_sep).any(axis=0) & ~j_has_long_contact

    valid = valid_residues
    out: dict[str, float] = {}
    for label, mask in [
        ("mean_all_positions", valid),
        ("mean_long_range_contacts", valid & j_has_long_contact),
        ("mean_long_range_non_contacts", valid & j_has_long_non_contact),
    ]:
        if not mask.any():
            out[label] = float("nan")
            continue
        out[label] = float(nll_per_j[mask].mean())
    return out


__all__ = ["compute_mlm_loss_aggregates"]
