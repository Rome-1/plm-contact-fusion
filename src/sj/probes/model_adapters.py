"""Family-specific glue for the head-probe + fusion scripts.

Each adapter handles three things that differ across model families:
  1. Tokenization (which special tokens get prepended/appended → seq_slice).
  2. Forward call (encoder-only vs causal vs encoder-decoder; what to pass).
  3. Attention extraction (which output attribute, padding/causal masks).

The head-probe + fusion code becomes family-agnostic: it asks the adapter
for ``per_head_attention_maps(seq) -> list[(L, L)]`` and trusts the adapter
to handle BOS/EOS, sentencepiece quirks, causal masking, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class ProteinModelAdapter:
    family: str
    variant: str
    n_layers: int
    n_heads: int
    description: str
    _loader: Callable[..., tuple] # returns (model, tokenizer, checkpoint)
    _tokenize_fn: Callable[..., tuple] # (tokenizer, seq) -> (input_ids: Tensor, seq_slice: slice)
    _attention_fn: Callable[..., tuple] # (model, input_ids, bf16) -> tuple of attention tensors
    _extract_seq_fn: Callable[
        ..., NDArray[np.float32]
    ] # (attentions, layer, head, seq_slice) -> (L, L)

    def load(self, **kw): # type: ignore[no-untyped-def]
        return self._loader(self.variant, **kw)

    def tokenize(self, tokenizer, sequence: str): # type: ignore[no-untyped-def]
        return self._tokenize_fn(tokenizer, sequence)

    def forward_attention(self, model, input_ids, bf16: bool): # type: ignore[no-untyped-def]
        return self._attention_fn(model, input_ids, bf16)

    def extract_seq_attention(self, attentions, layer: int, head: int, seq_slice: slice): # type: ignore[no-untyped-def]
        return self._extract_seq_fn(attentions, layer, head, seq_slice)


# ---------- ESM-family (ESM-2 + ESM-1b) ----------


def _esm_tokenize(tokenizer, sequence: str): # type: ignore[no-untyped-def]
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    L = len(sequence)
    # ESM tokenizers prepend <cls> and append <eos> → seq lives at slice(1, L+1)
    return enc["input_ids"], slice(1, L + 1)


def _esm_forward(model, input_ids, bf16: bool): # type: ignore[no-untyped-def]
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
            output_attentions=True,
        )
    return out.attentions # tuple of n_layers, each (1, H, T, T)


def _esm_extract(attentions, layer: int, head: int, seq_slice: slice): # type: ignore[no-untyped-def]
    return attentions[layer][0, head, seq_slice, seq_slice].to(torch.float32).cpu().numpy()


def make_esm_adapter(variant: str) -> ProteinModelAdapter:
    from sj.model import ESM2_VARIANTS, load_esm2_variant

    if variant not in ESM2_VARIANTS:
        raise ValueError(f"unknown ESM variant {variant!r}; known {list(ESM2_VARIANTS)}")
    spec = ESM2_VARIANTS[variant]
    family = "esm1b" if variant == "1b" else "esm2"

    def _loader(v, **kw):
        return load_esm2_variant(v, **kw)

    return ProteinModelAdapter(
        family=family,
        variant=variant,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        description=spec.description,
        _loader=_loader,
        _tokenize_fn=_esm_tokenize,
        _attention_fn=_esm_forward,
        _extract_seq_fn=_esm_extract,
    )


# ---------- AMPLIFY ----------


def _amplify_tokenize(tokenizer, sequence: str): # type: ignore[no-untyped-def]
    # AMPLIFY uses HuggingFace fast tokenizer; prepends [CLS] / [BOS] depending
    # on config. The tokenizer's add_special_tokens flag is honoured.
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    L = len(sequence)
    # AMPLIFY tokenizer prepends 1 special token. Match ESM convention.
    seq_slice = slice(1, L + 1)
    if enc["input_ids"].shape[1] == L + 2 or enc["input_ids"].shape[1] == L + 1:
        seq_slice = slice(1, L + 1)
    elif enc["input_ids"].shape[1] == L:
        seq_slice = slice(0, L)
    return enc["input_ids"], seq_slice


def _amplify_forward(model, input_ids, bf16: bool): # type: ignore[no-untyped-def]
    """AMPLIFY expects an ADDITIVE attention_mask (0 for valid positions,
    -inf for padding) rather than the standard HF binary 0/1 mask. We
    pass a flat 0 tensor since none of our inputs have padding."""
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
            output_attentions=True,
        )
    return out.attentions


def make_amplify_adapter(variant: str = "350M") -> ProteinModelAdapter:
    from sj.model_registry import AMPLIFY_VARIANTS, load_amplify_variant

    if variant not in AMPLIFY_VARIANTS:
        raise ValueError(f"unknown AMPLIFY variant {variant}")
    spec = AMPLIFY_VARIANTS[variant]

    def _loader(v, **kw):
        kw.pop("enable_gradient_checkpointing", None)
        return load_amplify_variant(v, **kw)

    return ProteinModelAdapter(
        family="amplify",
        variant=variant,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        description=spec.description,
        _loader=_loader,
        _tokenize_fn=_amplify_tokenize,
        _attention_fn=_amplify_forward,
        _extract_seq_fn=_esm_extract,
    )


# ---------- ProtT5 (encoder-only) ----------


def _prott5_tokenize(tokenizer, sequence: str): # type: ignore[no-untyped-def]
    # ProtT5 expects spaces between residues + sentencepiece encoding.
    # No BOS; appends </s>. So input length = L + 1.
    spaced = " ".join(sequence)
    enc = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
    L = len(sequence)
    # T5 sentencepiece prepends nothing and appends </s> → seq at slice(0, L)
    return enc["input_ids"], slice(0, L)


def _prott5_forward(model, input_ids, bf16: bool): # type: ignore[no-untyped-def]
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
            output_attentions=True,
        )
    return out.attentions # T5EncoderModel returns attentions if requested


def make_prott5_adapter(variant: str = "XL") -> ProteinModelAdapter:
    from sj.model_registry import PROTT5_VARIANTS, load_prott5_variant

    if variant not in PROTT5_VARIANTS:
        raise ValueError(f"unknown ProtT5 variant {variant}")
    spec = PROTT5_VARIANTS[variant]

    def _loader(v, **kw):
        kw.pop("enable_gradient_checkpointing", None)
        return load_prott5_variant(v, **kw)

    return ProteinModelAdapter(
        family="prott5",
        variant=variant,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        description=spec.description,
        _loader=_loader,
        _tokenize_fn=_prott5_tokenize,
        _attention_fn=_prott5_forward,
        _extract_seq_fn=_esm_extract,
    )


# ---------- ProGen2 (causal LM) ----------


def _progen2_tokenize(tokenizer, sequence: str): # type: ignore[no-untyped-def]
    # ProGen2 tokenizer: per-amino-acid + single BOS. Prepend "1" or "2" prefix
    # is sometimes recommended for ProGen2 to indicate forward / reverse strand;
    # we stay with raw sequence to match the "natural" generation distribution.
    enc = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    L = len(sequence)
    # If tokenizer prepended 1 BOS → seq at slice(1, L+1). Otherwise slice(0, L).
    if enc["input_ids"].shape[1] == L + 1:
        return enc["input_ids"], slice(1, L + 1)
    return enc["input_ids"], slice(0, L)


def _progen2_forward(model, input_ids, bf16: bool): # type: ignore[no-untyped-def]
    """Causal forward. Attention is lower-triangular (causal mask); we
    symmetrize downstream so that's fine — but the upper triangle is
    structurally zero for causal models."""
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
            output_attentions=True,
        )
    # ProGen2 / GPT-NeoX style returns out.attentions tuple; same convention.
    return out.attentions


def make_progen2_adapter(variant: str = "xlarge") -> ProteinModelAdapter:
    from sj.model_registry import PROGEN2_VARIANTS, load_progen2_variant

    if variant not in PROGEN2_VARIANTS:
        raise ValueError(f"unknown ProGen2 variant {variant}")
    spec = PROGEN2_VARIANTS[variant]

    def _loader(v, **kw):
        kw.pop("enable_gradient_checkpointing", None)
        return load_progen2_variant(v, **kw)

    return ProteinModelAdapter(
        family="progen2",
        variant=variant,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        description=spec.description,
        _loader=_loader,
        _tokenize_fn=_progen2_tokenize,
        _attention_fn=_progen2_forward,
        _extract_seq_fn=_esm_extract,
    )


# ---------- ESMC (ESM Cambrian, bidirectional masked LM) ----------
#
# ESMC uses a HuggingFace AutoTokenizer with the same <cls>…<eos> layout as
# ESM-2 (verified empirically: tokenized length == L+2, seq lives at
# slice(1, L+1)), and AutoModelForMaskedLM under attn_implementation="eager"
# returns per-layer (1, H, T, T) attentions. So the ESM helpers apply verbatim.
# Loading requires the Biohub transformers fork (see load_esmc_variant); this
# adapter is therefore only exercised from the isolated-image ESMC dispatch
# scripts, never the standard-image b2/cl7 runners.


def make_esmc_adapter(variant: str) -> ProteinModelAdapter:
    from sj.model_registry import ESMC_VARIANTS, load_esmc_variant

    if variant not in ESMC_VARIANTS:
        raise ValueError(f"unknown ESMC variant {variant!r}; known {list(ESMC_VARIANTS)}")
    spec = ESMC_VARIANTS[variant]

    def _loader(v, **kw):
        # ESMC's loader has no gradient-checkpointing knob; drop it if passed.
        kw.pop("enable_gradient_checkpointing", None)
        return load_esmc_variant(v, **kw)

    return ProteinModelAdapter(
        family="esmc",
        variant=variant,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        description=spec.description,
        _loader=_loader,
        _tokenize_fn=_esm_tokenize,
        _attention_fn=_esm_forward,
        _extract_seq_fn=_esm_extract,
    )


# ---------- factory ----------


FAMILY_ADAPTERS: dict[str, Callable[[str], ProteinModelAdapter]] = {
    "esm2": make_esm_adapter,
    "esm1b": make_esm_adapter,
    "amplify": make_amplify_adapter,
    "prott5": make_prott5_adapter,
    "progen2": make_progen2_adapter,
    "esmc": make_esmc_adapter,
}


def make_adapter(family: str, variant: str) -> ProteinModelAdapter:
    factory = FAMILY_ADAPTERS.get(family)
    if factory is None:
        raise ValueError(f"unknown family {family!r}; known {list(FAMILY_ADAPTERS)}")
    return factory(variant)


__all__ = [
    "FAMILY_ADAPTERS",
    "ProteinModelAdapter",
    "make_adapter",
    "make_amplify_adapter",
    "make_esm_adapter",
    "make_esmc_adapter",
    "make_progen2_adapter",
    "make_prott5_adapter",
]
