"""ESM-2 loader with SHA256 weight-integrity validation.

Every model load goes through `load_esm2_variant` (or its alias
`load_esm2_650m`). It hashes the on-disk checkpoint and refuses to return
a model if the hash does not match the manifest at
`checkpoints/<variant_manifest>.sha256`; a mismatch is a halt, not a warning.

The variant registry (``ESM2_VARIANTS``) covers the public ESM-2 family.
Each variant carries its own committed manifest.

HuggingFace Transformers is the only supported load path: HF Hub publishes
LFS sha256 at upload time, giving the manifest a verifiable provenance chain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Repository-relative checkpoint directory. Tests override via argument.
CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
DEFAULT_MANIFEST_PATH = CHECKPOINTS_DIR / "esm2_t33_650M_UR50D.sha256"

# Buffer size for streaming SHA256 — large enough that a 2.5 GB checkpoint
# hashes in a few seconds on commodity disk, small enough not to spike RAM.
_HASH_CHUNK_BYTES = 1 << 20 # 1 MiB


class WeightIntegrityError(RuntimeError):
    """Raised when an on-disk checkpoint does not match its committed SHA256.

    This is a halt-execution condition, not a soft warning.
    """


@dataclass(frozen=True)
class CheckpointEntry:
    """One row in the SHA256 manifest: a checkpoint filename and its hash."""

    filename: str
    sha256: str


def parse_manifest(manifest_path: Path) -> list[CheckpointEntry]:
    """Parse a sha256sum-format manifest into typed entries.

    Lines look like: ``<64-hex> <filename>`` (two spaces, sha256sum default).
    Lines starting with ``#`` and empty lines are skipped so we can keep human
    notes near the hashes.
    """
    if not manifest_path.is_file():
        raise WeightIntegrityError(
            f"SHA256 manifest missing at {manifest_path}. "
            "The integrity gate requires the manifest committed to the repo before any model load."
        )
    entries: list[CheckpointEntry] = []
    for raw_line in manifest_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise WeightIntegrityError(f"Malformed manifest line in {manifest_path}: {raw_line!r}")
        entries.append(CheckpointEntry(filename=parts[1].strip(), sha256=parts[0].lower()))
    if not entries:
        raise WeightIntegrityError(f"SHA256 manifest at {manifest_path} contains no entries.")
    return entries


def sha256_file(path: Path) -> str:
    """Stream-hash a file. Returns lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> str:
    """Verify ``checkpoint_path`` matches the committed manifest. Halts on mismatch.

    Returns the validated SHA256 so callers can record it in their VariantResult
    without re-hashing.
    """
    if not checkpoint_path.is_file():
        raise WeightIntegrityError(f"Checkpoint missing at {checkpoint_path}.")
    entries = parse_manifest(manifest_path)
    by_name = {e.filename: e for e in entries}
    expected = by_name.get(checkpoint_path.name)
    if expected is None:
        raise WeightIntegrityError(
            f"Checkpoint {checkpoint_path.name!r} is not listed in {manifest_path}. "
            f"Known entries: {sorted(by_name)}"
        )
    actual = sha256_file(checkpoint_path)
    if actual != expected.sha256:
        raise WeightIntegrityError(
            f"SHA256 mismatch for {checkpoint_path.name}: "
            f"expected {expected.sha256}, got {actual}. "
            "Refusing to load. Check upstream replacement or local corruption."
        )
    return actual


@dataclass(frozen=True)
class HFCheckpoint:
    """Result of HuggingFace-side validation: which file we'll load and its sha256."""

    snapshot_dir: Path
    weights_file: Path # absolute path; either model.safetensors or pytorch_model.bin
    sha256: str
    revision: str # HF revision the snapshot was downloaded at


HF_TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)
HF_WEIGHT_PRIMARY = "model.safetensors"
HF_WEIGHT_FALLBACK = "pytorch_model.bin"
HF_SAFETENSORS_INDEX = "model.safetensors.index.json"
HF_PYTORCH_INDEX = "pytorch_model.bin.index.json"


@dataclass(frozen=True)
class Esm2VariantSpec:
    """Static facts about one ESM-2 variant: id, manifest, architecture sizes."""

    key: str
    hf_model_id: str
    manifest_filename: str
    n_layers: int
    n_heads: int
    description: str


ESM2_VARIANTS: dict[str, Esm2VariantSpec] = {
    "8M": Esm2VariantSpec(
        "8M", "facebook/esm2_t6_8M_UR50D", "esm2_t6_8M_UR50D.sha256", 6, 20, "ESM-2-8M (tiny)"
    ),
    "35M": Esm2VariantSpec(
        "35M",
        "facebook/esm2_t12_35M_UR50D",
        "esm2_t12_35M_UR50D.sha256",
        12,
        20,
        "ESM-2-35M (small)",
    ),
    "150M": Esm2VariantSpec(
        "150M", "facebook/esm2_t30_150M_UR50D", "esm2_t30_150M_UR50D.sha256", 30, 20, "ESM-2-150M"
    ),
    "650M": Esm2VariantSpec(
        "650M",
        "facebook/esm2_t33_650M_UR50D",
        "esm2_t33_650M_UR50D.sha256",
        33,
        20,
        "ESM-2-650M (project default)",
    ),
    "3B": Esm2VariantSpec(
        "3B", "facebook/esm2_t36_3B_UR50D", "esm2_t36_3B_UR50D.sha256", 36, 40, "ESM-2-3B"
    ),
    "15B": Esm2VariantSpec(
        "15B", "facebook/esm2_t48_15B_UR50D", "esm2_t48_15B_UR50D.sha256", 48, 40, "ESM-2-15B"
    ),
    # ESM-1b (Rives 2021) — predecessor to ESM-2, same EsmForMaskedLM HF class
    # but absolute positional encoding (vs ESM-2 rotary). Loads via the same path.
    "1b": Esm2VariantSpec(
        "1b",
        "facebook/esm1b_t33_650M_UR50S",
        "esm1b_t33_650M_UR50S.sha256",
        33,
        20,
        "ESM-1b (Rives 2021 predecessor)",
    ),
}

HF_MODEL_ID = ESM2_VARIANTS["650M"].hf_model_id # back-compat alias


def _enumerate_shards(snapshot_dir: Path, index_filename: str) -> list[Path]:
    """Return the unique shard files referenced by an HF weight-shard index, sorted.

    HF index files are JSON with shape ``{"weight_map": {layer_name: shard_filename}}``.
    The same shard appears many times in the map; we only need each one once.
    """
    import json as _json

    index_path = snapshot_dir / index_filename
    with index_path.open() as fh:
        data = _json.load(fh)
    weight_map = data.get("weight_map", {})
    if not weight_map:
        raise WeightIntegrityError(f"empty weight_map in {index_path}; cannot enumerate shards")
    shard_names = sorted(set(weight_map.values()))
    shards = [snapshot_dir / name for name in shard_names]
    missing = [p for p in shards if not p.is_file()]
    if missing:
        raise WeightIntegrityError(
            f"shards listed in {index_path.name} are missing on disk: {[p.name for p in missing]}"
        )
    return shards


def validate_hf_snapshot(
    snapshot_dir: Path,
    *,
    prefer_safetensors: bool = True,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> HFCheckpoint:
    """Validate the HF Transformers cache directory against the SHA256 manifest.

    Single-file weights (small + medium ESM-2 variants) work as before — pick
    the safetensors file or fall back to ``pytorch_model.bin``, hash, validate.

    Sharded weights (large ESM-2 variants like 3B / 15B which ship as
    ``pytorch_model-0000N-of-000NN.bin`` plus a ``pytorch_model.bin.index.json``)
    are validated by hashing every shard listed in the index and checking
    each against its committed manifest entry. The returned ``HFCheckpoint``
    points ``weights_file`` at the index file (descriptive — the loader uses
    the snapshot dir, not the file path) and ``sha256`` is the SHA-256 of
    the sorted concatenation of per-shard hashes (a stable digest of the
    full sharded checkpoint). The same hash construction is used by
    the committed manifest under ``checkpoints/``.
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir():
        raise WeightIntegrityError(f"HF snapshot directory missing: {snapshot_dir}")
    safetensors = snapshot_dir / HF_WEIGHT_PRIMARY
    pytorch_bin = snapshot_dir / HF_WEIGHT_FALLBACK
    safetensors_index = snapshot_dir / HF_SAFETENSORS_INDEX
    pytorch_index = snapshot_dir / HF_PYTORCH_INDEX

    if prefer_safetensors and safetensors.is_file():
        weights_file = safetensors
    elif prefer_safetensors and safetensors_index.is_file():
        return _validate_sharded(snapshot_dir, safetensors_index.name, manifest_path=manifest_path)
    elif pytorch_bin.is_file():
        weights_file = pytorch_bin
    elif pytorch_index.is_file():
        return _validate_sharded(snapshot_dir, pytorch_index.name, manifest_path=manifest_path)
    elif safetensors.is_file():
        weights_file = safetensors
    elif safetensors_index.is_file():
        return _validate_sharded(snapshot_dir, safetensors_index.name, manifest_path=manifest_path)
    else:
        raise WeightIntegrityError(
            f"No weights present in {snapshot_dir} ({HF_WEIGHT_PRIMARY}, "
            f"{HF_WEIGHT_FALLBACK}, or shard index). Re-download the snapshot."
        )
    sha = validate_checkpoint(weights_file, manifest_path=manifest_path)
    revision = snapshot_dir.name # HF Hub names snapshot dirs by revision sha
    return HFCheckpoint(
        snapshot_dir=snapshot_dir,
        weights_file=weights_file,
        sha256=sha,
        revision=revision,
    )


def _validate_sharded(
    snapshot_dir: Path,
    index_filename: str,
    *,
    manifest_path: Path,
) -> HFCheckpoint:
    """Validate every shard of a sharded HF checkpoint against the manifest."""
    shards = _enumerate_shards(snapshot_dir, index_filename)
    per_shard_shas: list[str] = []
    for shard in shards:
        per_shard_shas.append(validate_checkpoint(shard, manifest_path=manifest_path))
    combined = combine_shard_shas(per_shard_shas)
    return HFCheckpoint(
        snapshot_dir=snapshot_dir,
        weights_file=snapshot_dir / index_filename,
        sha256=combined,
        revision=snapshot_dir.name,
    )


def combine_shard_shas(per_shard_shas: list[str]) -> str:
    """Stable digest for a sharded checkpoint: sha256 of sorted shard hashes joined.

    Sorting makes the digest insensitive to discovery order; joining with a
    single newline is a delimiter that cannot appear inside a hex hash.
    """
    payload = "\n".join(sorted(per_shard_shas)).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve_hf_snapshot(
    cache_dir: Path | None,
    hf_model_id: str = HF_MODEL_ID,
) -> Path:
    """Find the cached snapshot dir for ``hf_model_id``, downloading on a miss.

    Kept separate from the validate path so unit tests can exercise validation
    against a fixture directory without touching huggingface_hub.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise WeightIntegrityError(
            "huggingface_hub not installed; cannot resolve ESM-2 snapshot. "
            "Install via `pip install -e '.[dev]'`."
        ) from exc
    snapshot = snapshot_download(
        repo_id=hf_model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        # Skip TensorFlow weights — the LFS oid for tf_model.h5 is in the
        # manifest only as informational; we don't load it. Sharded variants
        # (3B, 15B) need the index + every pytorch_model-*.bin shard.
        allow_patterns=[
            HF_WEIGHT_PRIMARY,
            HF_SAFETENSORS_INDEX,
            "model-*-of-*.safetensors",
            HF_WEIGHT_FALLBACK,
            HF_PYTORCH_INDEX,
            "pytorch_model-*-of-*.bin",
            *HF_TOKENIZER_FILES,
        ],
    )
    return Path(snapshot)


def load_esm2_variant( # type: ignore[no-untyped-def]
    variant: str = "650M",
    *,
    cache_dir: Path | str | None = None,
    prefer_safetensors: bool = True,
    manifest_path: Path | None = None,
    device: str = "cpu",
    attn_implementation: str | None = None,
    enable_gradient_checkpointing: bool = False,
):
    """Load an ESM-2 variant from HF Transformers with SHA256 weight-integrity validation.

    Returns ``(model, tokenizer, hf_checkpoint)``. ``variant`` keys into
    :data:`ESM2_VARIANTS`. The default is ``"650M"`` so existing callers
    keep their behavior unchanged.
    """
    if variant not in ESM2_VARIANTS:
        raise ValueError(f"Unknown ESM-2 variant {variant!r}. Known: {sorted(ESM2_VARIANTS)}")
    spec = ESM2_VARIANTS[variant]
    if manifest_path is None:
        manifest_path = CHECKPOINTS_DIR / spec.manifest_filename
    snapshot_dir = _resolve_hf_snapshot(
        Path(cache_dir) if cache_dir else None,
        hf_model_id=spec.hf_model_id,
    )
    checkpoint = validate_hf_snapshot(
        snapshot_dir,
        prefer_safetensors=prefer_safetensors,
        manifest_path=manifest_path,
    )
    try:
        from transformers import EsmForMaskedLM, EsmTokenizer
    except ImportError as exc:
        raise WeightIntegrityError("transformers not installed; cannot instantiate ESM-2.") from exc
    tokenizer = EsmTokenizer.from_pretrained(str(checkpoint.snapshot_dir))
    from_pretrained_kwargs: dict = {
        "use_safetensors": checkpoint.weights_file.name == HF_WEIGHT_PRIMARY,
    }
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    model = EsmForMaskedLM.from_pretrained(
        str(checkpoint.snapshot_dir),
        **from_pretrained_kwargs,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    return model, tokenizer, checkpoint


def load_esm2_650m( # type: ignore[no-untyped-def]
    *,
    cache_dir: Path | str | None = None,
    prefer_safetensors: bool = True,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    device: str = "cpu",
    attn_implementation: str | None = None,
    enable_gradient_checkpointing: bool = False,
):
    """Load ESM-2-650M from HF Transformers with SHA256 weight-integrity validation.

    Returns ``(model, tokenizer, hf_checkpoint)``. The model is loaded on
    ``device``, in eval mode, with gradients disabled at the parameter level.
    The HFCheckpoint return value carries the validated sha256 so callers
    can stamp it into a VariantResult without re-hashing.

    Validation order: resolve snapshot → hash weight file → compare against
    manifest → only then call from_pretrained. This guarantees no weight
    bytes are unpickled before the integrity check.

    ``attn_implementation``: pass through to from_pretrained. None = HF
    default (SDPA in transformers 5.4+). "eager" is required when callers
    need ``output_attentions=True`` to return non-empty maps — SDPA fuses
    softmax+matmul and silently returns an empty attentions list.

    FlashAttention 2: ``attn_implementation='flash_attention_2'`` is
    supported on transformers 5.4+ — EsmPreTrainedModel sets
    ``_supports_flash_attn = True`` and the per-layer attention dispatch
    in ``models/esm/modeling_esm.py`` resolves the implementation via
    ``ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, …)``,
    so FA2 swaps in cleanly under the existing residual-edit hook
    (which fires on ``EsmLayer`` outputs, post-attention residual — FA2
    fuses softmax+matmul inside the layer but the hooked tensor is
    unchanged in shape and semantics). Verified at L=1000 with
    an optional FlashAttention path. Requires the ``flash-attn``
    package; SDPA default already calls FA2 underneath via
    PyTorch's scaled_dot_product_attention when flash-attn is installed,
    so the explicit setting matters mainly when comparing backends or
    forcing a particular kernel path.

    ``enable_gradient_checkpointing``: when True, calls
    model.gradient_checkpointing_enable() so each transformer block
    re-computes activations on the backward pass instead of caching them.
    Trades ~30% extra forward time for ~3× memory headroom — useful for
    pushing beyond L≈2000 on A100-80GB. We run inference-only so the
    'gradient' framing is misleading; HF still calls the knob this name.
    """
    return load_esm2_variant(
        "650M",
        cache_dir=cache_dir,
        prefer_safetensors=prefer_safetensors,
        manifest_path=manifest_path,
        device=device,
        attn_implementation=attn_implementation,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )


__all__ = [
    "CHECKPOINTS_DIR",
    "DEFAULT_MANIFEST_PATH",
    "ESM2_VARIANTS",
    "HF_MODEL_ID",
    "HF_PYTORCH_INDEX",
    "HF_SAFETENSORS_INDEX",
    "HF_TOKENIZER_FILES",
    "HF_WEIGHT_FALLBACK",
    "HF_WEIGHT_PRIMARY",
    "CheckpointEntry",
    "Esm2VariantSpec",
    "HFCheckpoint",
    "WeightIntegrityError",
    "combine_shard_shas",
    "load_esm2_650m",
    "load_esm2_variant",
    "parse_manifest",
    "sha256_file",
    "validate_checkpoint",
    "validate_hf_snapshot",
]
