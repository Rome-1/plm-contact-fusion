"""Result schema for one (protein, variant) evaluation row.

`VariantResult` plus its `HardwareInfo` are the JSON-serializable shape every
method writes, so per-cell results are comparable across architectures and runs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Frozen hardware/precision context for one VariantResult.

    Captured at job-launch time. Goes into every result JSON so a row's
    numbers can be re-attributed to a specific run environment.
    """

    gpu_name: str
    gpu_count: int
    precision: str # "bf16", "fp16", "fp32"
    cuda_version: str | None
    cudnn_version: str | None
    torch_version: str
    transformers_version: str | None
    esm_version: str | None
    image_tag: str | None


@dataclass
class VariantResult:
    """Single (protein, variant) result row. one JSON per row in the results store."""

    variant_id: str
    protein_id: str
    dataset: str # "zhang_1431" | "casp14_fm" | "cameo_pta"
    sequence_length: int
    top_L_2_short: float
    top_L_2_medium: float
    top_L_2_long: float
    bootstrap_ci_long: tuple[float, float]
    forwards_count: int
    gpu_seconds: float
    peak_memory_gb: float
    layer: int
    contact_map_path: str
    timestamp_utc: str
    git_sha: str
    checkpoint_sha256: str
    image_tag: str | None
    hardware: HardwareInfo


__all__ = [
    "HardwareInfo",
    "VariantResult",
]
