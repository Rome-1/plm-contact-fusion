"""Rao 2021 attention-head contact prediction, rerun on top-L/2.

Rao et al. ICLR 2021 (OpenReview fylclEqgvgd) trained a logistic regression
on top of ESM-1b attention heads to predict contacts. Their reported
0.527 long-range P@L is on **top-L**, not top-L/2 — making it apples-to-
oranges with Zhang 2024's top-L/2 number. the design requires a rerun on
top-L/2 splits so the comparison lands on the same metric.

In HuggingFace Transformers, the contact-head weights ride along with the
ESM-2 checkpoint as ``model.esm.contact_head`` (a small linear layer
operating on the symmetrized attention map). We don't retrain; we just
evaluate on the same proteins with the existing head.

This module provides ``run_rao_attention_contacts(sequence, model, tokenizer)``
and a note that the contact-head IS supervised (Rao 2021 trained the
logistic regression), so the resulting numbers go in the paper as a
supervised reference, not an unsupervised competitor.

This file relies on ESM-2's loaded model exposing the attention contact
head; if loaded via EsmModel + EsmForMaskedLM, the head lives at
``model.esm.contact_head`` (HF Transformers convention). The function
halts loud if the head is missing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from sj.eval.contacts import apc_correction

logger = logging.getLogger(__name__)

VARIANT_ID = "rao_2021_attention"


@dataclass
class RaoResult:
    contact_score: NDArray[np.float32]
    forwards_count: int
    wall_clock_seconds: float
    sequence_length: int


def _load_contact_head(model): # type: ignore[no-untyped-def]
    """Return the EsmContactHead module attached to a HF EsmForMaskedLM, or raise."""
    head = getattr(model.esm, "contact_head", None)
    if head is None:
        raise AttributeError(
            "model.esm has no contact_head — Rao 2021 baseline requires the "
            "regression head that ships with facebook/esm2_t33_650M_UR50D. "
            "If you instantiated EsmModel without the head, switch to "
            "EsmForMaskedLM and reload."
        )
    return head


def run_rao_attention_contacts( # type: ignore[no-untyped-def]
    sequence: str,
    *,
    model,
    tokenizer,
    bf16: bool = True,
    apply_apc: bool = True,
) -> RaoResult:
    """Compute the Rao 2021 attention-head contact map for one sequence.

    Returns a (L, L) score map suitable for top-L/2 evaluation. The head
    already produces a calibrated probability; we APC-correct it on top
    to match the rest of the pipeline's reporting convention.
    """
    contact_head = _load_contact_head(model)
    L = len(sequence)
    device = next(model.parameters()).device

    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    t0 = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16),
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
    # HF returns attentions as a tuple of (B, num_heads, L+2, L+2) per layer.
    if not hasattr(out, "attentions") or out.attentions is None:
        raise RuntimeError("model did not return attentions; need output_attentions=True")
    attentions = torch.stack(out.attentions, dim=1) # (B, n_layers, n_heads, L+2, L+2)
    score_map = contact_head(input_ids, attentions) # (B, L, L)
    score = score_map[0].to(torch.float32).cpu().numpy() # (L, L)

    # contact_head already symmetrizes + APCs internally in HF impl;
    # we APC again only if the attribute is missing or disabled.
    if apply_apc and not getattr(contact_head, "apc_corrected", True):
        score = apc_correction(score.astype(np.float64)).astype(np.float32)
    wall = time.perf_counter() - t0
    return RaoResult(
        contact_score=score.astype(np.float32),
        forwards_count=1,
        wall_clock_seconds=wall,
        sequence_length=L,
    )


__all__ = ["VARIANT_ID", "RaoResult", "run_rao_attention_contacts"]
