"""Multi-family protein model registry + load adapters.

Adds ESM-1b (drop-in to existing ESM2_VARIANTS via the same EsmForMaskedLM
class) and three non-ESM families:
  - AMPLIFY (chandar-lab MLM, custom architecture, trust_remote_code)
  - ProtT5 (Rostlab T5 encoder, sentencepiece tokenizer)
  - ProGen2 (causal-LM, sharded weights, trust_remote_code)

Each family exposes a `load_<family>_variant(variant_key, ...)` callable
returning ``(model, tokenizer, hf_checkpoint)`` plus an `Adapter` with
the small amount of family-specific glue our head-probe + fusion scripts
need (sequence slicing, attention extraction).

The ESM family continues to use ``sj.model.load_esm2_variant`` directly;
this module is the entry point for non-ESM models.

A weight-integrity gate: every loader hashes the on-disk weights against a committed
manifest. Sharded weight families use the sharded validator from
sj.model. trust_remote_code=True is required for AMPLIFY and ProGen2 —
this is acceptable risk because (a) we pin the HF revision
via snapshot_download and (b) the weight files are still SHA-validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sj.model import (
    CHECKPOINTS_DIR,
    HF_PYTORCH_INDEX,
    HF_SAFETENSORS_INDEX,
    HF_TOKENIZER_FILES,
    HF_WEIGHT_FALLBACK,
    HF_WEIGHT_PRIMARY,
    HFCheckpoint,
    WeightIntegrityError,
    validate_hf_snapshot,
)


@dataclass(frozen=True)
class ModelFamilySpec:
    """Static facts about a non-ESM protein-LM family variant."""

    family: str
    variant: str
    hf_model_id: str
    manifest_filename: str
    n_layers: int
    n_heads: int
    hidden_size: int
    description: str
    trust_remote_code: bool = False
    extra_allow_patterns: tuple[str, ...] = ()
    # Pin the HF repo to an exact commit so config.json/tokenizer.json (which
    # the weights-only SHA manifest does not cover) cannot drift from repo
    # HEAD between runs. None == HEAD (legacy behavior for already-pinned-by-
    # convention families). Set for third-party-namespace repos (e.g. ESMC).
    hf_revision: str | None = None


AMPLIFY_VARIANTS: dict[str, ModelFamilySpec] = {
    "350M": ModelFamilySpec(
        family="amplify",
        variant="350M",
        hf_model_id="chandar-lab/AMPLIFY_350M",
        manifest_filename="amplify_350M.sha256",
        n_layers=32,
        n_heads=15,
        hidden_size=960,
        description="AMPLIFY-350M (Fournier 2024, Chandar lab)",
        trust_remote_code=True,
        extra_allow_patterns=("amplify.py", "rmsnorm.py", "rotary.py", "tokenizer.json"),
    ),
}


PROTT5_VARIANTS: dict[str, ModelFamilySpec] = {
    "XL": ModelFamilySpec(
        family="prott5",
        variant="XL",
        hf_model_id="Rostlab/prot_t5_xl_uniref50",
        manifest_filename="prot_t5_xl_uniref50.sha256",
        n_layers=24,
        n_heads=32, # ProtT5-XL: 24 layers × 32 heads, hidden 1024 (encoder)
        hidden_size=1024,
        description="ProtT5-XL (Rostlab, T5 encoder, sentencepiece)",
        trust_remote_code=False,
        extra_allow_patterns=("spiece.model",),
    ),
}


PROGEN2_VARIANTS: dict[str, ModelFamilySpec] = {
    "large": ModelFamilySpec(
        family="progen2",
        variant="large",
        hf_model_id="hugohrban/progen2-large",
        manifest_filename="progen2_large.sha256",
        n_layers=32,
        n_heads=16,
        hidden_size=2560,
        description="ProGen2-large (Madani 2023, causal LM, 2.7B)",
        trust_remote_code=True,
        extra_allow_patterns=(
            "modeling_progen.py",
            "configuration_progen.py",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-*-of-*.safetensors",
        ),
    ),
    "xlarge": ModelFamilySpec(
        family="progen2",
        variant="xlarge",
        hf_model_id="hugohrban/progen2-xlarge",
        manifest_filename="progen2_xlarge.sha256",
        n_layers=32,
        n_heads=16,
        hidden_size=4096,
        description="ProGen2-xlarge (Madani 2023, causal LM, 6.4B)",
        trust_remote_code=True,
        extra_allow_patterns=(
            "modeling_progen.py",
            "configuration_progen.py",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-*-of-*.safetensors",
        ),
    ),
}


# ESMC (ESM Cambrian, EvolutionaryScale). Bidirectional masked LM with a
# categorical AA logit head (architectures: ESMCForMaskedLM) — so unlike the
# other non-ESM families it runs logit-CJ directly (not repr-CJ), exactly like
# ESM-2/ESM-1b. Loading requires EvolutionaryScale's transformers FORK
# (github.com/Biohub/transformers): the `esmc` model_type is NOT in upstream
# transformers, so this family is dispatched only from the isolated-image
# the ESMC load path.
ESMC_VARIANTS: dict[str, ModelFamilySpec] = {
    "300M": ModelFamilySpec(
        family="esmc",
        variant="300M",
        hf_model_id="biohub/ESMC-300M",
        manifest_filename="esmc_300m.sha256",
        n_layers=30,
        n_heads=15,
        hidden_size=960, # ESM-2-650M peer
        description="ESMC-300M (EvolutionaryScale Cambrian, bidirectional masked LM)",
        trust_remote_code=False, # class ships in the Biohub transformers fork
        extra_allow_patterns=("tokenizer.json",),
        hf_revision="05729d7947bd31aa2e1f59adb1b3099255390f2e",
    ),
    "600M": ModelFamilySpec(
        family="esmc",
        variant="600M",
        hf_model_id="biohub/ESMC-600M",
        manifest_filename="esmc_600m.sha256",
        n_layers=36,
        n_heads=18,
        hidden_size=1152, # ESM-2-3B peer
        description="ESMC-600M (EvolutionaryScale Cambrian, bidirectional masked LM)",
        trust_remote_code=False,
        extra_allow_patterns=("tokenizer.json",),
        hf_revision="465f75840fee10acc8c0db104ae244d8abb9288e",
    ),
}


# Family → variant_dict lookup.
NON_ESM_FAMILIES: dict[str, dict[str, ModelFamilySpec]] = {
    "amplify": AMPLIFY_VARIANTS,
    "prott5": PROTT5_VARIANTS,
    "progen2": PROGEN2_VARIANTS,
    "esmc": ESMC_VARIANTS,
}


def _resolve_with_extras(spec: ModelFamilySpec, cache_dir: Path | None) -> Path:
    """Like sj.model._resolve_hf_snapshot but with family-specific extras
    in allow_patterns (e.g., spiece.model, custom modeling_*.py)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise WeightIntegrityError(
            "huggingface_hub not installed; cannot resolve snapshot."
        ) from exc
    snapshot = snapshot_download(
        repo_id=spec.hf_model_id,
        revision=spec.hf_revision, # None == HEAD; pinned commit for ESMC
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=[
            HF_WEIGHT_PRIMARY,
            HF_SAFETENSORS_INDEX,
            "model-*-of-*.safetensors",
            HF_WEIGHT_FALLBACK,
            HF_PYTORCH_INDEX,
            "pytorch_model-*-of-*.bin",
            *HF_TOKENIZER_FILES,
            *spec.extra_allow_patterns,
        ],
    )
    return Path(snapshot)


def _validate_family_snapshot(spec: ModelFamilySpec, cache_dir: Path | None) -> HFCheckpoint:
    snapshot_dir = _resolve_with_extras(spec, cache_dir)
    manifest_path = CHECKPOINTS_DIR / spec.manifest_filename
    return validate_hf_snapshot(snapshot_dir, manifest_path=manifest_path)


def load_amplify_variant( # type: ignore[no-untyped-def]
    variant: str = "350M",
    *,
    cache_dir: Path | str | None = None,
    device: str = "cpu",
    attn_implementation: str | None = None,
):
    """Load AMPLIFY (Fournier 2024). Custom architecture; trust_remote_code=True."""
    if variant not in AMPLIFY_VARIANTS:
        raise ValueError(f"unknown AMPLIFY variant {variant}; known {list(AMPLIFY_VARIANTS)}")
    spec = AMPLIFY_VARIANTS[variant]
    checkpoint = _validate_family_snapshot(spec, Path(cache_dir) if cache_dir else None)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint.snapshot_dir), trust_remote_code=True)
    # AMPLIFY's auto_map only registers AutoConfig + AutoModel (not
    # AutoModelForMaskedLM). For attention extraction we only need the
    # encoder + output_attentions, so AutoModel is sufficient.
    #
    # low_cpu_mem_usage=False is required: AMPLIFY's amplify.py builds a
    # plain-attribute `self.freqs_cis = precompute_freqs_cis(...)` in
    # __init__ rather than registering it as a buffer. With HF's default
    # low_cpu_mem_usage=True the init runs on meta device, leaving
    # freqs_cis as a meta tensor that .to(device) cannot materialize.
    kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": False}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(str(checkpoint.snapshot_dir), **kwargs)

    # AMPLIFY's freqs_cis is a plain attribute (not a buffer). If __init__
    # happened on meta device anyway (HF behavior can override our setting
    # when trust_remote_code is in effect), .to() will fail at forward
    # time. Force-re-init it from the rotary helper that ships with the
    # model code, materializing on the real device. Idempotent if
    # freqs_cis is already non-meta.
    try:
        is_meta = getattr(model, "freqs_cis", None) is not None and model.freqs_cis.is_meta
    except Exception:
        is_meta = False
    if is_meta:
        import importlib

        # rotary.py is namespaced under transformers_modules.<rev_hash>
        mod_name = type(model).__module__.rsplit(".", 1)[0] + ".rotary"
        rotary = importlib.import_module(mod_name)
        cfg = model.config
        model.freqs_cis = rotary.precompute_freqs_cis(
            cfg.hidden_size // cfg.num_attention_heads, cfg.max_length
        )

    # Patch xformers.memory_efficient_attention with an SDPA equivalent on
    # the loaded amplify module. The original raises 'No operator found ...'
    # under the xformers + torch combo; SDPA is correct + native to torch.
    # Inputs/outputs are (B, M, H, K). attn_bias is additive (or None).
    import sys as _sys

    import torch.nn.functional as _F

    def _sdpa_xops(query, key, value, attn_bias=None, p: float = 0.0, **_kw):
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        out = _F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
        return out.transpose(1, 2).contiguous()

    for mod_name, mod in list(_sys.modules.items()):
        if "amplify" in mod_name and hasattr(mod, "memory_efficient_attention"):
            mod.memory_efficient_attention = _sdpa_xops

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, tokenizer, checkpoint


def load_prott5_variant( # type: ignore[no-untyped-def]
    variant: str = "XL",
    *,
    cache_dir: Path | str | None = None,
    device: str = "cpu",
    attn_implementation: str | None = None,
):
    """Load ProtT5 ENCODER (we don't need the decoder). Sentencepiece tokenizer."""
    if variant not in PROTT5_VARIANTS:
        raise ValueError(f"unknown ProtT5 variant {variant}; known {list(PROTT5_VARIANTS)}")
    spec = PROTT5_VARIANTS[variant]
    checkpoint = _validate_family_snapshot(spec, Path(cache_dir) if cache_dir else None)

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(str(checkpoint.snapshot_dir))
    kwargs: dict[str, Any] = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = T5EncoderModel.from_pretrained(str(checkpoint.snapshot_dir), **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, tokenizer, checkpoint


def load_progen2_variant( # type: ignore[no-untyped-def]
    variant: str = "xlarge",
    *,
    cache_dir: Path | str | None = None,
    device: str = "cpu",
    attn_implementation: str | None = None,
):
    """Load ProGen2 (Madani 2023). Causal LM; trust_remote_code=True.

    Compatibility shim: hugohrban/progen2-xlarge's modeling_progen.py was
    written for transformers <=4.45 — newer transformers reads
    ``model.all_tied_weights_keys`` during load and raises AttributeError
    on this model. We patch the PreTrainedModel base class with an empty
    default before load so the lookup succeeds (ProGen ties only
    transformer.wte.weight already declared in ``_tied_weights_keys``;
    the empty default matches HF's pre-4.50 behavior).
    """
    if variant not in PROGEN2_VARIANTS:
        raise ValueError(f"unknown ProGen2 variant {variant}; known {list(PROGEN2_VARIANTS)}")
    spec = PROGEN2_VARIANTS[variant]
    checkpoint = _validate_family_snapshot(spec, Path(cache_dir) if cache_dir else None)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        # Newer transformers expects a dict mapping (source -> target) weight names.
        # ProGen ties only transformer.wte.weight (declared in _tied_weights_keys);
        # the empty dict means no extra tying beyond what the subclass declared.
        PreTrainedModel.all_tied_weights_keys = {} # type: ignore[attr-defined]

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint.snapshot_dir), trust_remote_code=True)
    import torch as _torch

    # bf16 load: 6.4B fp32 weights (~25.6 GB) OOM'd the b2 eval-200 probe on long
    # proteins; bf16 (~12.8 GB) fits and matches the autocast dtype every forward uses.
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,
        "torch_dtype": _torch.bfloat16,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint.snapshot_dir), **kwargs)

    # transformers 5.x removed get_head_mask from PreTrainedModel; the
    # ProGen2 custom modeling code on HF Hub still calls it. Monkeypatch
    # a no-op compatible implementation onto the inner transformer block
    # (and the outer wrapper, for safety). When head_mask is None this
    # just returns [None] * n_layers, which is the standard semantics.
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        return head_mask

    import types as _types

    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = _types.MethodType(_get_head_mask, model)
    inner = getattr(model, "transformer", None)
    if inner is not None and not hasattr(inner, "get_head_mask"):
        inner.get_head_mask = _types.MethodType(_get_head_mask, inner)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    # ProGen2's modeling_progen.py creates `self.scale_attn` as a plain
    # torch.tensor (not a registered buffer) in each attention layer's
    # __init__. When from_pretrained runs __init__ on the meta device,
    # scale_attn stays on meta and the first forward crashes on the
    # attention-score division. Walk all modules and recompute it.
    import torch as _torch

    target = _torch.device(device)
    for mod in model.modules():
        if (
            hasattr(mod, "scale_attn")
            and isinstance(mod.scale_attn, _torch.Tensor)
            and (mod.scale_attn.device.type == "meta" or mod.scale_attn.device != target)
        ):
            head_dim = getattr(mod, "head_dim", None)
            if head_dim is not None:
                mod.scale_attn = _torch.sqrt(_torch.tensor(float(head_dim), device=target)).to(
                    model.dtype
                )
    return model, tokenizer, checkpoint


def load_esmc_variant( # type: ignore[no-untyped-def]
    variant: str = "300M",
    *,
    cache_dir: Path | str | None = None,
    device: str = "cpu",
    attn_implementation: str | None = None,
):
    """Load ESMC (EvolutionaryScale Cambrian). Bidirectional masked LM.

    Requires the Biohub transformers FORK to be installed (it carries the
    ``esmc`` model_type + ``ESMCForMaskedLM`` class; upstream transformers does
    not). We load via ``AutoModelForMaskedLM`` so ``out.logits`` is available
    for logit-CJ, exactly like ESM-2. ``attn_implementation="eager"`` is needed
    if a caller wants ``output_attentions`` (SDPA/FA2 return empty attentions).
    """
    if variant not in ESMC_VARIANTS:
        raise ValueError(f"unknown ESMC variant {variant}; known {list(ESMC_VARIANTS)}")
    spec = ESMC_VARIANTS[variant]
    checkpoint = _validate_family_snapshot(spec, Path(cache_dir) if cache_dir else None)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint.snapshot_dir))
    kwargs: dict[str, Any] = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForMaskedLM.from_pretrained(str(checkpoint.snapshot_dir), **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, tokenizer, checkpoint


# Family dispatch — the head-probe + fusion scripts call this.
LOADERS: dict[str, Any] = {
    "amplify": load_amplify_variant,
    "prott5": load_prott5_variant,
    "progen2": load_progen2_variant,
    "esmc": load_esmc_variant,
}


def load_protein_model(
    family: str,
    variant: str,
    *,
    cache_dir: Path | str | None = None,
    device: str = "cpu",
    attn_implementation: str | None = None,
    enable_gradient_checkpointing: bool = False,
):
    """Family dispatch. ESM family stays on sj.model.load_esm2_variant."""
    if family in ("esm2", "esm1b", "esm"):
        from sj.model import load_esm2_variant

        return load_esm2_variant(
            variant,
            cache_dir=cache_dir,
            device=device,
            attn_implementation=attn_implementation,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
    loader = LOADERS.get(family)
    if loader is None:
        raise ValueError(f"unknown family {family!r}; known {[*list(LOADERS), 'esm2', 'esm1b']}")
    # gradient_checkpointing is family-specific; only ESM path supports it
    # cleanly via the shared loader. For now, callers that need it should
    # call the family loader directly.
    return loader(
        variant,
        cache_dir=cache_dir,
        device=device,
        attn_implementation=attn_implementation,
    )


__all__ = [
    "AMPLIFY_VARIANTS",
    "ESMC_VARIANTS",
    "LOADERS",
    "NON_ESM_FAMILIES",
    "PROGEN2_VARIANTS",
    "PROTT5_VARIANTS",
    "ModelFamilySpec",
    "load_amplify_variant",
    "load_esmc_variant",
    "load_progen2_variant",
    "load_protein_model",
    "load_prott5_variant",
]
