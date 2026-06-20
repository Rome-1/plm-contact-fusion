"""Representation-CJ: an architecture-agnostic CJ analog using hidden states.

Standard CJ (Zhang 2024) requires a per-position categorical distribution over
the 20 amino acids — works for MLMs (ESM-1/2, ESM-1b, AMPLIFY) but doesn't
fit encoder-decoder span-corruption (ProtT5) or causal (ProGen2) models.

Representation-CJ replaces the logit-response with a hidden-state response.
For each (position i, alt amino acid a'):
  1. Replace sequence[i] = a'
  2. Forward perturbed input
  3. Extract hidden states at a chosen layer
  4. response[i, a', j] = ||h_perturbed[j] - h_native[j]|| (or cosine, ...)

Then aggregate across alts (mean) → (L, L), symmetrize + APC + zero-diag →
contact score. The aggregation matches Zhang's 4-way mean-center step in
spirit but operates on a 3D tensor (no a'' axis).

Cost: same order as logit-CJ (~19L forwards per protein, batched).

Why this generalizes:
  - ESM family / AMPLIFY: works (sanity check vs logit-CJ)
  - ProtT5 (encoder): use encoder hidden states; no decoder needed
  - ProGen2 (causal): hidden states at j only depend on tokens ≤j, so
    response at j is structurally zero for i > j (upper-triangular signal)

Reference: this is the natural generalization of Bhattacharya et al. 2022's
"single-layer attention suffices" framing — measure structural-Jacobian-like
information content of hidden representations, not just logits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray

from sj.eval.contacts import apc_correction

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY" # canonical 20, alphabetical


class ReprCJMode(StrEnum):
    SERIAL = "serial"
    BATCHED = "batched"


class ReprCJDistance(StrEnum):
    L2 = "l2" # per-position ||h_perturbed[j] - h_native[j]||_2
    COSINE = "cosine" # 1 - cos(h_perturbed[j], h_native[j])
    RMS = "rms" # ||diff||_2 / ||h_native[j]||_2 (scale-normalized)


@dataclass(frozen=True)
class ReprCJConfig:
    layer: int = -1
    """Hidden-state layer index. -1 = last; 0 = embedding output."""

    mode: ReprCJMode = ReprCJMode.BATCHED
    distance: ReprCJDistance = ReprCJDistance.L2
    bf16: bool = True
    max_perturbs_per_batch: int = 64
    apply_apc: bool = True
    symmetrize: bool = True
    causal: bool = False
    """Causal-CJ mode for autoregressive LMs (ProGen2). When True:
    - Skip symmetrize step (response matrix is structurally upper-triangular)
    - Score directly from raw mean (don't fold lower triangle in)
    - Optionally combine with reversed-sequence pass (caller decides via
      a separate run_representation_cj call on the reversed sequence)
    Background: in causal LMs, h[j] only depends on tokens ≤j, so perturbing
    position i > j produces zero response at j. Standard symmetrize+APC washes
    out the actual upper-triangular signal. Causal mode preserves it."""


@dataclass(frozen=True)
class ReprCJResult:
    contact_score: NDArray[np.float32]
    forwards_count: int
    wall_clock_seconds: float
    peak_memory_bytes: int
    sequence_length: int
    n_alts: int
    layer: int


# Adapter callable signature: forward(model, input_ids, bf16, layer) -> NDArray (B, L, D)
# Returns the per-position hidden state at `layer`, sliced to the L sequence
# positions (no special tokens).
HiddenForwardFn = Callable[[object, "torch.Tensor", bool, int, slice], NDArray[np.floating]]


def _esm_hidden_forward(
    model, input_ids: torch.Tensor, bf16: bool, layer: int, seq_slice: slice
) -> NDArray[np.floating]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            output_attentions=False,
        )
    # out.hidden_states is (n_layers+1,) tuple of (B, T, D)
    h = out.hidden_states[layer]
    return h[:, seq_slice, :].to(torch.float32).cpu().numpy()


def _amplify_hidden_forward(
    model, input_ids: torch.Tensor, bf16: bool, layer: int, seq_slice: slice
) -> NDArray[np.floating]:
    """AMPLIFY needs an additive attention_mask (0/-inf) — see model_adapters."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    additive_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=additive_mask,
            output_hidden_states=True,
            output_attentions=False,
        )
    h = out.hidden_states[layer]
    return h[:, seq_slice, :].to(torch.float32).cpu().numpy()


def _prott5_hidden_forward(
    model, input_ids: torch.Tensor, bf16: bool, layer: int, seq_slice: slice
) -> NDArray[np.floating]:
    """ProtT5 encoder. Sequence tokenization expects spaces — caller handles.
    Hidden states at requested layer."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            output_attentions=False,
        )
    h = out.hidden_states[layer]
    return h[:, seq_slice, :].to(torch.float32).cpu().numpy()


def _progen2_hidden_forward(
    model, input_ids: torch.Tensor, bf16: bool, layer: int, seq_slice: slice
) -> NDArray[np.floating]:
    """ProGen2 causal LM. Hidden states at layer; response will be
    structurally upper-triangular (i ≤ j positions affect h[j])."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            output_attentions=False,
        )
    h = out.hidden_states[layer]
    return h[:, seq_slice, :].to(torch.float32).cpu().numpy()


def _esmc_hidden_forward(
    model,
    input_ids,
    bf16,
    layer,
    seq_slice,
): # type: ignore[no-untyped-def]
    """ESMC (Biohub fork) hidden states.

    The Biohub ESMC fork crashes with ``CUDA error: unspecified launch failure``
    in ``modeling_esmc.py`` (the ``elif bool_mask.all() and not output_attentions``
    branch) under torch 2.11 / CUDA 13 when attentions are NOT requested. The fusion
    path avoids it by passing ``output_attentions=True``; we do the same here so
    repr-CJ routes through the working attention branch. For the default last layer
    we read ``out.last_hidden_state`` (one tensor, exposed directly); only fall back
    to the all-layers path for an explicit interior layer.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    autocast_dtype = torch.bfloat16
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        if layer in (-1, None):
            out = model(input_ids=input_ids, output_attentions=True)
            h = out.last_hidden_state
        else:
            out = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)
            h = out.hidden_states[layer]
    return h[:, seq_slice, :].to(torch.float32).cpu().numpy()


_HIDDEN_FORWARDS: dict[str, HiddenForwardFn] = {
    "esm2": _esm_hidden_forward,
    "esm1b": _esm_hidden_forward,
    "esmc": _esmc_hidden_forward, # last_hidden_state path (avoid 31-layer materialisation)
    "amplify": _amplify_hidden_forward,
    "prott5": _prott5_hidden_forward,
    "progen2": _progen2_hidden_forward,
}


def _distance(
    h_native: NDArray[np.floating],
    h_perturbed: NDArray[np.floating],
    metric: ReprCJDistance,
) -> NDArray[np.floating]:
    """h_native: (L, D); h_perturbed: (L, D); returns (L,) per-position distance."""
    diff = h_perturbed - h_native
    if metric == ReprCJDistance.L2:
        return np.linalg.norm(diff, axis=-1)
    if metric == ReprCJDistance.COSINE:
        a = h_native / (np.linalg.norm(h_native, axis=-1, keepdims=True) + 1e-12)
        b = h_perturbed / (np.linalg.norm(h_perturbed, axis=-1, keepdims=True) + 1e-12)
        return 1.0 - np.einsum("ld,ld->l", a, b)
    if metric == ReprCJDistance.RMS:
        norm = np.linalg.norm(h_native, axis=-1) + 1e-12
        return np.linalg.norm(diff, axis=-1) / norm
    raise ValueError(f"unknown distance {metric}")


def run_representation_cj(
    sequence: str,
    *,
    model,
    tokenizer,
    family: str,
    tokenize_fn, # callable(tokenizer, seq) -> (input_ids, seq_slice)
    config: ReprCJConfig | None = None,
) -> ReprCJResult:
    """Compute representation-CJ contact score for one protein.

    Args:
        sequence: protein sequence (no special tokens; tokenize_fn adds them)
        model: HF model (or adapter wrapper) supporting output_hidden_states=True
        tokenizer: matching tokenizer
        family: one of esm2/esm1b/amplify/prott5/progen2 (selects forward fn)
        tokenize_fn: callable from sj.probes.model_adapters returning
                     (input_ids tensor, seq_slice) for one sequence

    Returns:
        ReprCJResult with (L, L) contact_score after aggregate + symmetrize +
        APC + zero-diag.
    """
    if config is None:
        config = ReprCJConfig()
    if family not in _HIDDEN_FORWARDS:
        raise ValueError(f"unknown family {family!r}")
    hidden_forward = _HIDDEN_FORWARDS[family]

    L = len(sequence)
    n_alts = len(STANDARD_AA)

    t0 = time.perf_counter()
    device = next(model.parameters()).device

    # Native hidden states
    native_ids, seq_slice = tokenize_fn(tokenizer, sequence)
    h_native_batch = hidden_forward(model, native_ids, config.bf16, config.layer, seq_slice)
    h_native = h_native_batch[0] # (L, D)
    forwards_count = 1

    # Allocate response tensor (i, alt, j)
    raw = np.zeros((L, n_alts, L), dtype=np.float32)

    if config.mode == ReprCJMode.SERIAL:
        # One perturbation per forward — slow but trivial.
        for i in range(L):
            for alt_idx, alt_aa in enumerate(STANDARD_AA):
                seq_p = sequence[:i] + alt_aa + sequence[i + 1 :]
                ids_p, sl_p = tokenize_fn(tokenizer, seq_p)
                h_p = hidden_forward(model, ids_p, config.bf16, config.layer, sl_p)[0]
                raw[i, alt_idx] = _distance(h_native, h_p, config.distance)
                forwards_count += 1
    elif config.mode == ReprCJMode.BATCHED:
        # Pack up to max_perturbs_per_batch (i, alt) pairs into one forward.
        # Build the (sequence_id, position, alt_idx, alt_aa) work list, then
        # tokenize each and stack.
        work: list[tuple[int, int]] = [(i, alt_idx) for i in range(L) for alt_idx in range(n_alts)]
        # Pre-tokenize each unique perturbed sequence. (i, alt_idx) → ids tensor.
        # All the same length because we replace one token; tokenizer adds the
        # same special tokens every time. So we can stack into one (B, T) batch.
        for chunk_start in range(0, len(work), config.max_perturbs_per_batch):
            chunk = work[chunk_start : chunk_start + config.max_perturbs_per_batch]
            chunk_ids = []
            for i, alt_idx in chunk:
                alt_aa = STANDARD_AA[alt_idx]
                seq_p = sequence[:i] + alt_aa + sequence[i + 1 :]
                ids_p, _ = tokenize_fn(tokenizer, seq_p)
                chunk_ids.append(ids_p[0]) # squeeze batch dim
            batch = torch.stack(chunk_ids, dim=0)
            h_chunk = hidden_forward(model, batch, config.bf16, config.layer, seq_slice)
            forwards_count += 1
            for k, (i, alt_idx) in enumerate(chunk):
                raw[i, alt_idx] = _distance(h_native, h_chunk[k], config.distance)
    else:
        raise ValueError(f"unknown mode {config.mode}")

    # Aggregate across alts: (L, L)
    score = raw.mean(axis=1)
    if config.causal:
        # Upper-triangular only: position j responds to perturbations at i ≤ j.
        # The lower triangle (i > j) is structurally zero; we preserve the
        # upper triangle as-is and mirror it to the lower triangle so the
        # standard top-L/2 long readout (which uses upper-triangle pairs) gets
        # the meaningful signal.
        upper = np.triu(score, k=1)
        score = upper + upper.T # symmetric copy of just the upper triangle
        if config.apply_apc:
            score = apc_correction(score.astype(np.float64))
            score = 0.5 * (score + score.T)
        np.fill_diagonal(score, 0.0)
    else:
        # Symmetrize + APC + zero diag (standard bidirectional flow)
        if config.symmetrize:
            score = 0.5 * (score + score.T)
        if config.apply_apc:
            score = apc_correction(score.astype(np.float64))
            if config.symmetrize:
                score = 0.5 * (score + score.T)
        np.fill_diagonal(score, 0.0)
    score = score.astype(np.float32)

    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return ReprCJResult(
        contact_score=score,
        forwards_count=forwards_count,
        wall_clock_seconds=wall,
        peak_memory_bytes=int(peak),
        sequence_length=L,
        n_alts=n_alts,
        layer=config.layer,
    )


__all__ = [
    "STANDARD_AA",
    "ReprCJConfig",
    "ReprCJDistance",
    "ReprCJMode",
    "ReprCJResult",
    "run_representation_cj",
]
