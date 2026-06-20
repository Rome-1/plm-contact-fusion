"""Head-fusion operators for Phase 1.5.

Goal: take K candidate per-head attention maps (each (L, L), already
symmetrized + APC-corrected + zero-diagonal) and fuse into a single
(L, L) contact-score map. The naive arithmetic mean (existing
the attention-ensemble baseline) underperforms the best single head — these
operators target three diagnosed failure modes:

  1. Vote dilution: heads attend to disjoint subsets of contacts;
     mean dilutes a confident vote with K-1 near-zeros. Fixes:
     elementwise max, geometric mean, Hadamard product.
  2. Scale heterogeneity: heads have very different score scales;
     mean is dominated by the highest-variance head. Fixes:
     z-score-then-mean, reciprocal rank fusion, top-N union.
  3. Positional contamination: many heads still carry sequence-distance
     bias after per-head APC. Fixes: positional-fraction filter, then
     fuse the residual.

All operators are parameter-free or use only unsupervised hyperparameters
(no contact labels touched). Inputs are NumPy arrays so this module is
test-friendly and runs CPU-side after the GPU forward.

References for the methods:
  - Reciprocal Rank Fusion: Cormack et al. 2009 (CIKM) — IR consensus.
  - Hopfield-like iteration: Ramsauer et al. 2020 (modern Hopfield nets).
  - APC: Dunn et al. 2008 (Bioinformatics) — re-applied here as a closure step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Public method names as a registry — each method is `fuse_<name>(stack, **kwargs) -> (L, L)`.
FUSION_METHOD_NAMES = (
    "top1",
    "naive_mean",
    "elementwise_max",
    "geometric_mean",
    "zscore_mean",
    "reciprocal_rank_fusion",
    "positional_filter_mean",
    "entropy_gated_mean",
    "graph_closure",
    "hopfield_iterate",
    "spectral_consensus",
)


# ---------- helpers ----------


def _validate_stack(stack: NDArray[np.floating]) -> None:
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError(f"expected stack shape (K, L, L), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("empty stack")


def _zero_diag(m: NDArray[np.floating]) -> NDArray[np.floating]:
    out = m.copy()
    np.fill_diagonal(out, 0.0)
    return out


def apc(m: NDArray[np.floating]) -> NDArray[np.floating]:
    """Average-Product Correction (Dunn 2008). Subtract row*col/grand mean."""
    row = m.mean(axis=1, keepdims=True)
    col = m.mean(axis=0, keepdims=True)
    grand = m.mean()
    if grand == 0:
        return m.copy()
    return m - (row @ col) / grand


def _symmetrize(m: NDArray[np.floating]) -> NDArray[np.floating]:
    return 0.5 * (m + m.T)


def _row_entropy(a: NDArray[np.floating]) -> float:
    """Mean over rows of Shannon entropy of each row (treated as a distribution)."""
    p = np.maximum(a, 0.0)
    rs = p.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    p = p / rs
    eps = 1e-12
    H = -np.sum(p * np.log(p + eps), axis=1)
    return float(H.mean())


def _positional_correlation(a: NDArray[np.floating]) -> float:
    """Pearson r between attention map and 1/|i - j| (positional null)."""
    L = a.shape[0]
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    sep = np.abs(i - j).astype(np.float64)
    pos = np.zeros_like(sep)
    np.divide(1.0, sep, out=pos, where=sep > 0)
    iu = np.triu_indices(L, k=1)
    av = a[iu].astype(np.float64)
    pv = pos[iu]
    if av.std() < 1e-12 or pv.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(av, pv)[0, 1])


# ---------- combination operators ----------


def fuse_top1(stack: NDArray[np.floating]) -> NDArray[np.floating]:
    """Baseline: just return the first map (caller is expected to put the
    aggregate-best head first)."""
    _validate_stack(stack)
    return stack[0].copy()


def fuse_naive_mean(stack: NDArray[np.floating]) -> NDArray[np.floating]:
    """Baseline failure mode: arithmetic mean of all K maps."""
    _validate_stack(stack)
    return stack.mean(axis=0)


def fuse_elementwise_max(stack: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per-pair OR: each head specializes; max preserves the most confident
    head's vote at each pair instead of diluting with K-1 near-zeros."""
    _validate_stack(stack)
    return stack.max(axis=0)


def fuse_geometric_mean(stack: NDArray[np.floating], eps: float = 1e-9) -> NDArray[np.floating]:
    """Per-pair AND-ish: requires all heads to put nonzero mass; idiosyncratic
    noise is suppressed because one near-zero head crushes the product."""
    _validate_stack(stack)
    return np.exp(np.log(stack + eps).mean(axis=0)) - eps


def fuse_zscore_mean(stack: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per-head z-score (over off-diagonal upper triangle) then arithmetic
    mean. Removes scale heterogeneity — the largest-variance head no
    longer dominates."""
    _validate_stack(stack)
    K, L, _ = stack.shape
    iu = np.triu_indices(L, k=1)
    out = np.zeros_like(stack)
    for k in range(K):
        v = stack[k, iu[0], iu[1]]
        mu, sd = v.mean(), v.std()
        if sd < 1e-12:
            out[k] = stack[k] - mu
        else:
            out[k] = (stack[k] - mu) / sd
    return out.mean(axis=0)


def fuse_reciprocal_rank_fusion(
    stack: NDArray[np.floating], c: float = 60.0
) -> NDArray[np.floating]:
    """Cormack 2009 RRF. Each head contributes 1/(c + rank) per pair; sum
    across heads. Scale-free; only ranking matters."""
    _validate_stack(stack)
    K, L, _ = stack.shape
    iu = np.triu_indices(L, k=1)
    n_pairs = iu[0].size
    out = np.zeros((L, L), dtype=np.float64)
    for k in range(K):
        flat = stack[k, iu[0], iu[1]]
        order = np.argsort(-flat)
        ranks = np.empty(n_pairs, dtype=np.int64)
        ranks[order] = np.arange(n_pairs)
        scores = 1.0 / (c + ranks)
        out[iu[0], iu[1]] += scores
    out = _symmetrize(out)
    return out


def fuse_positional_filter_mean(
    stack: NDArray[np.floating], threshold: float = 0.3
) -> NDArray[np.floating]:
    """Drop heads whose APC'd map still correlates with 1/|i - j| above
    `threshold`; mean the rest. Targets sequence-distance contamination."""
    _validate_stack(stack)
    keep_mask = np.array(
        [_positional_correlation(stack[k]) <= threshold for k in range(stack.shape[0])]
    )
    if not keep_mask.any():
        # Degenerate case: every head is positional. Fall back to top-1.
        return stack[0].copy()
    return stack[keep_mask].mean(axis=0)


def fuse_entropy_gated_mean(
    stack: NDArray[np.floating], alpha: float = 4.0
) -> NDArray[np.floating]:
    """Weight each head by ``exp(-alpha * row_entropy_mean)`` — sharper
    heads get more weight. Per-protein adaptive (no global rank needed)."""
    _validate_stack(stack)
    weights = np.array([np.exp(-alpha * _row_entropy(stack[k])) for k in range(stack.shape[0])])
    if weights.sum() < 1e-12:
        return stack.mean(axis=0)
    weights = weights / weights.sum()
    return np.einsum("k,kij->ij", weights, stack)


def fuse_graph_closure(
    stack: NDArray[np.floating], alpha: float = 0.5, reapc: bool = True
) -> NDArray[np.floating]:
    """Take the mean, then add a one-step graph closure: S' = S + alpha * S@S.
    If i-j and j-k both contact, boost i-k. Re-APC the result."""
    _validate_stack(stack)
    s = stack.mean(axis=0)
    s2 = s @ s
    if s2.max() > 0:
        s2 = s2 / s2.max() * s.max()
    out = s + alpha * s2
    out = _symmetrize(out)
    out = _zero_diag(out)
    if reapc:
        out = apc(out)
        out = _zero_diag(out)
    return out


def fuse_hopfield_iterate(
    stack: NDArray[np.floating], beta: float = 1.0, n_steps: int = 3
) -> NDArray[np.floating]:
    """Modern-Hopfield-style iterated fixed point. Init at top-1 head;
    update M_{t+1} = softmax_pair(beta * mean_h(A_h * M_t)). Reinforces
    pairs many heads agree on; cancels noise."""
    _validate_stack(stack)
    M = stack[0].copy()
    for _ in range(n_steps):
        # mean_h (A_h * M) — Hadamard, not matmul; preserves locality
        update = (stack * M[None, :, :]).mean(axis=0)
        # pair-wise softmax (over off-diagonal upper triangle)
        update = update - update.max() # numerical stability
        exp_u = np.exp(beta * update)
        np.fill_diagonal(exp_u, 0.0)
        denom = exp_u.sum()
        if denom < 1e-12:
            break
        M = exp_u / denom
        M = _symmetrize(M)
    return M


def fuse_spectral_consensus(stack: NDArray[np.floating], rank: int = 3) -> NDArray[np.floating]:
    """Stack heads as (K, L*L); take rank-r SVD of the K-by-(L*L) matrix;
    reconstruct the projection of the mean onto the top-r right singular
    vectors. Shared low-rank structure across heads = real signal."""
    _validate_stack(stack)
    K, L, _ = stack.shape
    flat = stack.reshape(K, -1).astype(np.float64)
    rank = min(rank, K, flat.shape[1])
    # economy SVD
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    flat_low = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    out = flat_low.mean(axis=0).reshape(L, L)
    out = _symmetrize(out)
    out = _zero_diag(out)
    return out


# ---------- selection helpers (unsupervised alternatives) ----------


def select_self_consistency_topk(
    head_stack_all: NDArray[np.floating],
    k: int,
) -> NDArray[np.intp]:
    """Pick the K heads with highest mean Pearson agreement with all others.

    Args:
        head_stack_all: (n_heads_total, L, L) — APC'd attention maps for
            every candidate head (e.g., all heads in last 3 layers).
        k: number of heads to keep.

    Returns:
        Indices into the first axis of ``head_stack_all`` for the K
        most-consensus-like heads.
    """
    iu = np.triu_indices(head_stack_all.shape[1], k=1)
    flat = head_stack_all[:, iu[0], iu[1]] # (n, n_pairs)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    flat = flat / norms
    sim = flat @ flat.T # (n, n) cosine similarity
    np.fill_diagonal(sim, 0.0)
    mean_agreement = sim.mean(axis=1)
    return np.argsort(-mean_agreement)[:k]


def select_positional_filter(
    head_stack_all: NDArray[np.floating],
    k: int,
    threshold: float = 0.3,
) -> NDArray[np.intp]:
    """Drop heads whose maps correlate with 1/|i - j| above ``threshold``,
    return the top-K indices of the survivors (in original order)."""
    n = head_stack_all.shape[0]
    pos_corrs = np.array([_positional_correlation(head_stack_all[i]) for i in range(n)])
    keep_mask = pos_corrs <= threshold
    survivors = np.where(keep_mask)[0]
    return survivors[:k]


def select_entropy_per_protein(
    head_stack_all: NDArray[np.floating],
    k: int,
) -> NDArray[np.intp]:
    """Pick the K heads with the lowest mean row entropy (sharpest)."""
    ents = np.array([_row_entropy(head_stack_all[i]) for i in range(head_stack_all.shape[0])])
    return np.argsort(ents)[:k]


__all__ = [
    "FUSION_METHOD_NAMES",
    "apc",
    "fuse_elementwise_max",
    "fuse_entropy_gated_mean",
    "fuse_geometric_mean",
    "fuse_graph_closure",
    "fuse_hopfield_iterate",
    "fuse_naive_mean",
    "fuse_positional_filter_mean",
    "fuse_reciprocal_rank_fusion",
    "fuse_spectral_consensus",
    "fuse_top1",
    "fuse_zscore_mean",
    "select_entropy_per_protein",
    "select_positional_filter",
    "select_self_consistency_topk",
]
