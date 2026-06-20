"""Zhang 2024 categorical Jacobian baseline (Baseline A serial + Baseline B batched).

Reference: Zhang et al., PNAS 2024 (doi: 10.1073/pnas.2406285121).
ColabBio reference (Ovchinnikov ``utils.py:get_contacts`` +
  Zhang ``jac/02_get_jac_batch.py:get_categorical_jacobian``):
  https://github.com/zzhangzzhang/pLMs-interpretability
  https://github.com/sokrypton/ColabBio/blob/main/categorical_jacobian/utils.py

Algorithm (matches Zhang's reference code):

  1. Build the raw 4D categorical Jacobian. For each position i ∈ [0, L)
     and each standard AA a' ∈ Σ (all 20 — INCLUDING the native, whose
     row is zero before centering and contributes a small residual after
     centering), perturb input at position i to a' and run one forward.
     The raw entry at (i, a', j, a'') is the logit at j, AA a'', minus
     the native logit at j, AA a'':

       J[i, a', j, a''] = logits_perturbed[i→a'][j, a''] - logits_native[j, a'']

     Shape: (L, 20, L, 20). Fp32 memory at L=600: ~576 MB.

  2. Center along all 4 axes: ``for d in range(4): J -= J.mean(d, keepdims=True)``.
     This removes the additive marginal effects (site-i, perturbation-a',
     site-j, AA-baseline-a''), leaving only the pairwise interaction term —
     the direct categorical analog of how DCA computes its coupling matrix
     J_ij after subtracting the field terms h_i, h_j.

  3. Frobenius norm over the 4D's two AA axes (1 and 3):

       score[i, j] = sqrt(Σ_{a', a''} J_centered[i, a', j, a'']²)

     This collapses the (20, 20) per-pair coupling block to its magnitude.

  4. Zero diagonal, APC-correct (Dunn et al. 2008), symmetrize.

Why we use Zhang's algorithm and not a mean-of-norms approximation: the
4-way centering + Frobenius preserves per-(a', a'') interaction structure
(without it the score conflates "j genuinely couples to i via a specific
AA pair" with "j is generally noisy"). The 4D tensor is also the minimal
sufficient statistic for downstream aggregators
variant — they reduce the same tensor with different operators.

Two compute modes by design:
  - SERIAL: 20 single-row forwards per perturbation site, dispatched in a
    Python loop. Closest analog of what Zhang's per-protein script does
    (their loop batches 20 alts per position via tile, but each position
    is dispatched serially); we want this baseline for a fair "engineering
    speedup" attribution.
  - BATCHED: pack as many (sequence with one-position perturbation) rows
    as fit into one batched forward across positions, capped by
    ``max_perturbs_per_batch``. Same arithmetic; different scheduling. Our
    paper's headline "vs CJ" speedup claim is vs BATCHED.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from sj.data.datasets import STANDARD_AA
from sj.eval.contacts import apc_correction

logger = logging.getLogger(__name__)


class CJBaselineMode(StrEnum):
    """Which compute baseline to report."""

    SERIAL = "serial" # Baseline A: Zhang's published per-position serial CJ
    BATCHED = "batched" # Baseline B: maximally-batched on the same hardware


@dataclass(frozen=True)
class CJConfig:
    """Knobs for the standard categorical Jacobian sweep."""

    mode: CJBaselineMode = CJBaselineMode.BATCHED
    # Hard cap on sequence length. Beyond this the per-position perturbation
    # set must be chunked to fit memory. Set to 2200 so the longest CASP14-FM
    # target (T1044, L=2180) is admissible — verify CPU RAM (raw fp32 at
    # L=2200 is ~7.7 GB; fp16 raw_dtype halves it).
    sequence_length_cap: int = 2200
    bf16: bool = True
    # Cap on perturbations packed into one BATCHED forward. Tuned for A100
    # 80GB at L=512 — drops linearly with L^2 so the runner halves the
    # batch when L doubles. Override via config for benchmarking.
    max_perturbs_per_batch: int = 64
    apply_apc: bool = True
    symmetrize: bool = True
    # CPU-side dtype for the (L, 20, L, 20) raw 4D Jacobian. "float32" is
    # the historical default; "float16" halves CPU RAM (matters at L≥1500 —
    # fp32 raw at L=2000 is 6.4 GB, fp16 is 3.2 GB) at the cost of slightly
    # noisier centering. (numpy 1.26 lacks native bf16 support; fp16 is the
    # only practical sub-fp32 numeric type without a torch round-trip.) The
    # 4-way mean-center math always promotes to fp64 internally, so the
    # dtype here only affects the storage between collection and reduction.
    # Per-alt logit deltas are typically in [-30, 30] — fp16 dynamic range
    # is fine. Empirically <1e-4 score-map difference vs fp32 storage.
    raw_dtype: str = "float32"


@dataclass
class CJResult:
    """Output of run_standard_cj: contact-score map + compute log."""

    contact_score: NDArray[np.float32] # (L, L) post-APC, post-symmetrize
    forwards_count: int
    wall_clock_seconds: float
    peak_memory_bytes: int
    sequence_length: int
    mode: CJBaselineMode
    # Raw 4D categorical Jacobian, kept around so downstream
    # aggregator variants can re-collapse without a re-run. Shape
    # (L, 20, L, 20); raw[i, a', j, a''] = logits_perturbed[i→a'][j, a''] -
    # logits_native[j, a'']. The 4-way centering + Frobenius reduction in
    # ``run_standard_cj`` reads from this tensor; downstream variants apply
    # different reductions (e.g. per-pair max).
    raw_jacobian: NDArray[np.float32] | None = field(default=None)


def _standard_aa_token_ids(tokenizer) -> tuple[int, ...]: # type: ignore[no-untyped-def]
    """Return ESM-2 token IDs for the 20 standard amino acids, in STANDARD_AA order."""
    ids = []
    for aa in STANDARD_AA:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer has no token for standard AA {aa!r}")
        ids.append(int(token_id))
    return tuple(ids)


def _native_token_ids(sequence: str, tokenizer) -> Tensor: # type: ignore[no-untyped-def]
    """Tokenize a single sequence to (1, L+2) ids including BOS/EOS."""
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]


def _forward_logits( # type: ignore[no-untyped-def]
    model,
    input_ids: Tensor,
    *,
    bf16: bool,
) -> Tensor:
    """Run ESM-2 lm-head forward, return (B, L+2, vocab) logits.

    Uses ``torch.inference_mode`` + autocast so callers don't have to.
    """
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        out = model(input_ids=input_ids)
    # transformers EsmForMaskedLM returns ``logits``; some EsmModel variants
    # return ``last_hidden_state`` and require an LM-head call. The contract
    # for this baseline is that ``model`` is a masked-LM-capable head.
    if hasattr(out, "logits"):
        return out.logits
    raise AttributeError(
        "Model output has no .logits attribute. Standard CJ requires a "
        "masked-language-model head; load via EsmForMaskedLM, not EsmModel."
    )


def run_standard_cj( # type: ignore[no-untyped-def]
    sequence: str,
    *,
    model,
    tokenizer,
    config: CJConfig | None = None,
) -> CJResult:
    """Compute Standard CJ for one protein.

    Returns a contact-score map (post-APC if configured) plus compute log.
    """
    config = config or CJConfig()
    L = len(sequence)
    if config.sequence_length_cap < L:
        raise ValueError(f"L={L} exceeds sequence_length_cap={config.sequence_length_cap}")

    standard_aa_ids = _standard_aa_token_ids(tokenizer)
    aa_id_to_index = {tid: i for i, tid in enumerate(standard_aa_ids)}

    native_ids = _native_token_ids(sequence, tokenizer) # (1, L+2)
    # Drop BOS at idx 0 and EOS at idx L+1. For ESM-2 tokenizer this is
    # always positions [1, L+1).
    seq_slice = slice(1, L + 1)

    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Native logits over the standard-AA subset at every position.
    t0 = time.perf_counter()
    native_logits = _forward_logits(model, native_ids, bf16=config.bf16) # (1, L+2, V)
    native_aa_logits = native_logits[0, seq_slice][:, list(standard_aa_ids)] # (L, 20)
    native_aa_logits_np = native_aa_logits.to(torch.float32).cpu().numpy()
    forwards = 1

    # Raw 4D categorical Jacobian: J[i, a', j, a''].
    # Index a' ranges over ALL 20 standard AAs (incl native) — see module
    # docstring for why. Native row collapses to zero before centering
    # but contributes a small residual after the 4-way mean subtraction.
    # Dtype configurable: fp32 historic default; bf16 halves CPU RAM for
    # long proteins. _runner-side fills are downcast on assignment.
    raw_np_dtype = getattr(np, config.raw_dtype)
    raw = np.zeros((L, 20, L, 20), dtype=raw_np_dtype)
    if config.mode == CJBaselineMode.SERIAL:
        forwards += _run_serial(
            model=model,
            native_ids=native_ids,
            native_aa_logits=native_aa_logits_np,
            standard_aa_ids=standard_aa_ids,
            aa_id_to_index=aa_id_to_index,
            sequence=sequence,
            seq_slice=seq_slice,
            raw=raw,
            bf16=config.bf16,
        )
    elif config.mode == CJBaselineMode.BATCHED:
        forwards += _run_batched(
            model=model,
            native_ids=native_ids,
            native_aa_logits=native_aa_logits_np,
            standard_aa_ids=standard_aa_ids,
            aa_id_to_index=aa_id_to_index,
            sequence=sequence,
            seq_slice=seq_slice,
            raw=raw,
            bf16=config.bf16,
            max_perturbs_per_batch=config.max_perturbs_per_batch,
        )
    else:
        raise ValueError(f"Unknown CJ mode: {config.mode}")

    # Zhang's reference contact extraction (utils.py:get_contacts):
    # 4-way mean-center → Frobenius norm over (a', a'') → zero diag → APC → symmetrize.
    score = _jacobian_to_contact_score(
        raw, apply_apc=config.apply_apc, symmetrize=config.symmetrize
    )

    wall_clock = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return CJResult(
        contact_score=score,
        forwards_count=forwards,
        wall_clock_seconds=wall_clock,
        peak_memory_bytes=int(peak_mem),
        sequence_length=L,
        mode=config.mode,
        raw_jacobian=raw,
    )


def _jacobian_to_contact_score(
    raw: NDArray[np.float32],
    *,
    apply_apc: bool,
    symmetrize: bool,
) -> NDArray[np.float32]:
    """Reduce raw 4D Jacobian (L, 20, L, 20) → (L, L) contact score.

    Mirrors Ovchinnikov's ``utils.py:get_contacts`` from the ColabBio
    reference. fp64 promotion before centering avoids accumulation error
    on long sequences; downcast at the end.
    """
    j = raw.astype(np.float64)
    for axis in range(4):
        j -= j.mean(axis=axis, keepdims=True)
    score = np.sqrt(np.square(j).sum(axis=(1, 3))) # (L, L)
    np.fill_diagonal(score, 0.0)
    if apply_apc:
        score = apc_correction(score)
    if symmetrize:
        score = 0.5 * (score + score.T)
    return score.astype(np.float32)


def _run_serial( # type: ignore[no-untyped-def]
    *,
    model,
    native_ids: Tensor,
    native_aa_logits: NDArray[np.float32],
    standard_aa_ids: tuple[int, ...],
    aa_id_to_index: dict[int, int],
    sequence: str,
    seq_slice: slice,
    raw: NDArray[np.float32],
    bf16: bool,
) -> int:
    """Baseline A: 20 single-sequence forwards per perturbation site (incl native)."""
    L = len(sequence)
    standard_ids_list = list(standard_aa_ids)
    forwards = 0
    for i in range(L):
        native_id = int(native_ids[0, i + 1].item())
        if native_id not in aa_id_to_index:
            # Non-standard residue (e.g. X). Leave 4D row as zeros — Standard
            # CJ literature treats these as missing data; centering still
            # produces a numerically defined block but the score there is
            # uninformative; the eval-time valid_residues mask drops it.
            continue
        for a_prime, alt_id in enumerate(standard_aa_ids):
            perturbed = native_ids.clone()
            perturbed[0, i + 1] = alt_id
            logits = _forward_logits(model, perturbed, bf16=bf16)
            forwards += 1
            alt_aa_logits = (
                logits[0, seq_slice][:, standard_ids_list].to(torch.float32).cpu().numpy()
            )
            raw[i, a_prime, :, :] = alt_aa_logits - native_aa_logits
    return forwards


def _run_batched( # type: ignore[no-untyped-def]
    *,
    model,
    native_ids: Tensor,
    native_aa_logits: NDArray[np.float32],
    standard_aa_ids: tuple[int, ...],
    aa_id_to_index: dict[int, int],
    sequence: str,
    seq_slice: slice,
    raw: NDArray[np.float32],
    bf16: bool,
    max_perturbs_per_batch: int,
) -> int:
    """Baseline B: pack up to ``max_perturbs_per_batch`` (i, a') perturbations per forward.

    Each row of the batched input differs from native_ids in exactly one
    position. We track (i, a') → batch row so we can write into ``raw``
    once per batch.
    """
    L = len(sequence)
    standard_ids_list = list(standard_aa_ids)

    # Build the full (i, a', alt_id) work list, then chunk. Includes the
    # native-AA case at each position so the 4D tensor row 'a'==native' is
    # populated (will collapse to zero pre-centering by definition).
    work: list[tuple[int, int, int]] = []
    for i in range(L):
        native_id = int(native_ids[0, i + 1].item())
        if native_id not in aa_id_to_index:
            continue
        for a_prime, alt_id in enumerate(standard_aa_ids):
            work.append((i, a_prime, alt_id))

    forwards = 0
    for chunk_start in range(0, len(work), max_perturbs_per_batch):
        chunk = work[chunk_start : chunk_start + max_perturbs_per_batch]
        batch_size = len(chunk)
        batch = native_ids.repeat(batch_size, 1) # (B, L+2)
        for row, (i, _a, alt_id) in enumerate(chunk):
            batch[row, i + 1] = alt_id
        logits = _forward_logits(model, batch, bf16=bf16) # (B, L+2, V)
        forwards += 1
        # (B, L, 20) — restrict to standard AAs at sequence positions.
        alt_aa_logits = (
            logits[:, seq_slice][:, :, standard_ids_list].to(torch.float32).cpu().numpy()
        )
        for row, (i, a_prime, _alt_id) in enumerate(chunk):
            raw[i, a_prime, :, :] = alt_aa_logits[row] - native_aa_logits

    return forwards


__all__ = [
    "CJBaselineMode",
    "CJConfig",
    "CJResult",
    "run_standard_cj",
]
