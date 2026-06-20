"""End-to-end eval pipeline that turns a contact-score map into a VariantResult.

Shared between every variant + baseline so reported numbers are computed
identically. Inputs:

  - protein_id, sequence, ground-truth contact mask (from a DatasetLoader)
  - score map (from a variant or baseline)
  - compute log (forwards count, wall-clock, peak memory, layer)

Outputs a VariantResult-shaped dict ready to write to the results store.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sj.eval.contacts import bootstrap_delta_ci, top_lk_precision


def evaluate_score_map(
    score_map: NDArray[np.float32],
    ground_truth: NDArray[np.bool_],
    *,
    k: int = 2,
    valid_residues: NDArray[np.bool_] | None = None,
) -> dict[str, float | tuple[float, float]]:
    """Compute top-L/k precision broken out by range for a single protein.

    Returns the 3-band split (short/medium/long) AND the finer 5-band split
    (very_short / short_fine / medium_fine / long_fine / ultra_long) for
    long-protein analysis. Both splits are emitted so callers can read either
    the standard bands or the finer breakdown.

    ``valid_residues`` is forwarded to top_lk_precision; see that
    function for the eval-time semantics.
    """
    score = score_map.astype(np.float64)
    out: dict[str, float | tuple[float, float]] = {}
    # Legacy 3-band split (preserved for back-compat).
    for r in ("short", "medium", "long"):
        res = top_lk_precision(
            score, ground_truth, range_name=r, k=k, valid_residues=valid_residues
        )
        out[f"top_L_{k}_{r}"] = res.precision
        if r == "long":
            out["n_long_predicted"] = res.predicted
            out["n_long_true_positives"] = res.true_positives
    # Finer 5-band split.
    for r in ("very_short", "short_fine", "medium_fine", "long_fine", "ultra_long"):
        res = top_lk_precision(
            score, ground_truth, range_name=r, k=k, valid_residues=valid_residues
        )
        out[f"top_L_{k}_{r}"] = res.precision
        out[f"n_{r}_predicted"] = res.predicted
        out[f"n_{r}_true_positives"] = res.true_positives
    return out


def per_protein_long_range_deltas(
    variant_long: list[float],
    baseline_long: list[float],
) -> NDArray[np.float64]:
    """Paired per-protein deltas for bootstrap CI of the variant-vs-baseline lift."""
    if len(variant_long) != len(baseline_long):
        raise ValueError(
            f"variant ({len(variant_long)}) and baseline ({len(baseline_long)}) "
            "must have the same per-protein cardinality"
        )
    return np.asarray(variant_long, dtype=np.float64) - np.asarray(baseline_long, dtype=np.float64)


def assemble_variant_result(
    *,
    variant_id: str,
    protein_id: str,
    dataset: str,
    sequence_length: int,
    score_map: NDArray[np.float32],
    ground_truth: NDArray[np.bool_],
    forwards_count: int,
    gpu_seconds: float,
    peak_memory_bytes: int,
    layer: int,
    git_sha: str,
    checkpoint_sha256: str,
    image_tag: str | None,
    hardware: dict[str, Any],
    contact_map_path: str = "",
    valid_residues: NDArray[np.bool_] | None = None,
    mlm_loss: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the JSON-ready dict matching sj.interfaces.VariantResult.

    ``mlm_loss``: optional dict of MLM-loss aggregates from the variant's
    native forward. Caller passes one or more of:
      - ``mean_all_positions``: -log p(native AA at j | clean input),
        averaged over all valid j.
      - ``mean_long_range_non_contacts``: same but only at j with
        no long-range contact partner (sequence-distant non-partners).
      - ``mean_long_range_contacts``: same but only at j IS a long-range
        contact partner of some i.
    Useful for long-protein analysis since it detects ESM-2 OOD behavior
    at L > training context.
    """
    metrics = evaluate_score_map(score_map, ground_truth, valid_residues=valid_residues)
    # Per-protein long-range CI is degenerate (1 sample); just store the
    # point estimate as a tuple of (long, long). Cross-protein CI is
    # computed at aggregation time.
    long_pt = float(metrics["top_L_2_long"])
    result: dict[str, Any] = {
        "variant_id": variant_id,
        "protein_id": protein_id,
        "dataset": dataset,
        "sequence_length": sequence_length,
        "top_L_2_short": float(metrics["top_L_2_short"]),
        "top_L_2_medium": float(metrics["top_L_2_medium"]),
        "top_L_2_long": long_pt,
        "bootstrap_ci_long": (long_pt, long_pt),
        "forwards_count": int(forwards_count),
        "gpu_seconds": float(gpu_seconds),
        "peak_memory_gb": float(peak_memory_bytes) / (1 << 30),
        "layer": int(layer),
        "contact_map_path": contact_map_path,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": git_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "image_tag": image_tag,
        "hardware": hardware,
    }
    # Fine 5-band breakdown (additive — old consumers ignore unknown keys).
    for r in ("very_short", "short_fine", "medium_fine", "long_fine", "ultra_long"):
        result[f"top_L_2_{r}"] = float(metrics[f"top_L_2_{r}"])
    if mlm_loss is not None:
        result["mlm_loss"] = {k: float(v) for k, v in mlm_loss.items()}
    return result


def aggregate_long_range_with_ci(
    long_per_protein: list[float],
    *,
    baseline_long_per_protein: list[float] | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute mean + bootstrap 95% CI for long-range top-L/2 across proteins.

    If a baseline list is supplied, also computes the per-protein delta CI
    (variant - baseline). the design requires this CI on the Zhang reproduction
    delta and on every variant-vs-baseline delta.
    """
    arr = np.asarray(long_per_protein, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    if arr.size == 0:
        return {"mean_long": 0.0, "n_proteins": 0}
    lo, hi = bootstrap_delta_ci(arr - arr.mean(), n_resamples=n_resamples, confidence=confidence)
    out: dict[str, Any] = {
        "mean_long": mean,
        "ci_long_lo": float(mean + lo),
        "ci_long_hi": float(mean + hi),
        "n_proteins": int(arr.size),
    }
    if baseline_long_per_protein is not None:
        deltas = per_protein_long_range_deltas(long_per_protein, baseline_long_per_protein)
        delta_lo, delta_hi = bootstrap_delta_ci(
            deltas, n_resamples=n_resamples, confidence=confidence
        )
        out.update(
            {
                "mean_delta_long": float(deltas.mean()),
                "delta_ci_lo": float(delta_lo),
                "delta_ci_hi": float(delta_hi),
            }
        )
    return out


__all__ = [
    "aggregate_long_range_with_ci",
    "assemble_variant_result",
    "evaluate_score_map",
    "per_protein_long_range_deltas",
]
