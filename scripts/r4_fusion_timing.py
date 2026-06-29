"""r4 Job 1: per-protein FUSION wall-clock + peak memory (paired-timing fix).

Reviewer (TMLR-style) flagged that Table A18 / Fig F2 assign fusion a single
fixed per-forward time per backbone, forcing a flat "L-independent" curve. CJ's
per-protein wall-clock + peak memory are ALREADY cached (every
results/cl7_phase15_cj_*.json carries wall_clock_seconds + peak_gpu_gb per
protein, across L=30..2180). The only missing measurement is FUSION's
per-protein timing. This script supplies exactly that, on the same A100-80GB and
the same proteins, so r4_timing_table.py can emit a real per-L paired curve.

Design (cost discipline): the existing fusion_one_protein reloads the model on
every protein (del model at the end), so a 200-protein starmap would burn hours
of GPU on reloads. This function loads the model ONCE per (variant, dataset) and
loops proteins, timing only fusion's map-production path:

    forward (eager attentions) -> extract top-K head maps (sym+APC) -> naive_mean

which is what CJ's cached wall_clock_seconds measures on its side (compute ->
(L,L) map, excluding model load and excluding scoring against ground truth).

Outputs: results/r4_fusion_timing_<variant>_<dataset>.json with per-protein
{protein_id, sequence_length, fusion_wall_clock_seconds, fusion_peak_gpu_gb}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

# Reuse the dataset->proteins mapping and topk-key logic from the main dispatch.
def _proteins_for_dataset(dataset: str):  # inlined to avoid cross-script import
    if dataset == "zhang_eval200":
        from sj.data.zhang_splits import eval_proteins

        return list(eval_proteins())
    if dataset == "zhang_select10":
        from sj.data.zhang_splits import select_proteins

        return list(select_proteins())
    if dataset == "casp14_fm":
        from sj.data.casp14_fm import CASP14FMLoader, default_casp14_fm_root

        return list(CASP14FMLoader(data_root=default_casp14_fm_root()))
    if dataset in ("cameo_pta_select10", "cameo_pta_eval"):
        from sj.data.cameo_pretraining_aware import cameo_pta_split

        return list(
            cameo_pta_split(
                "select10" if dataset == "cameo_pta_select10" else "eval",
                data_root=Path("data/cameo_pta"),
            )
        )
    raise ValueError(f"unknown dataset {dataset!r}")


_APP_NAME = "plm-contact-fusion-r4-timing"

try:
    import modal as _modal  # type: ignore[import-untyped]

    _lock_path = REPO_ROOT / "requirements.lock"
    _checkpoints_dir = REPO_ROOT / "checkpoints"
    # ESM-2-only: no xformers/sentencepiece/tiktoken (those are ProtT5/ProGen2
    # deps and xformers builds from source ~30 min). model_adapters imports the
    # per-family loaders lazily, so the esm2 path never touches them.
    _image = (
        _modal.Image.from_registry("python:3.11-slim-bookworm")
        .pip_install_from_requirements(str(_lock_path))
        .add_local_python_source("sj", "baselines")
        .add_local_dir(str(_checkpoints_dir), "/checkpoints")
    )
    _app = _modal.App(name=_APP_NAME, image=_image)
    _hf_secret = _modal.Secret.from_name("hf_token")
    _weights_volume = _modal.Volume.from_name("plm-contact-fusion-weights", create_if_missing=True)

    @_app.function(
        gpu="A100-80GB",
        timeout=3600,
        secrets=[_hf_secret],
        volumes={"/vol/weights": _weights_volume},
        max_containers=int(os.environ.get("SJ_MAX_CONTAINERS", "4")),
    )
    def fusion_timing_batch(
        family: str,
        variant: str,
        proteins: list[tuple[str, str]],
        topk_indices: list[tuple[int, int]],
        k: int = 10,
        bf16: bool = True,
    ) -> list[dict]:
        """Time fusion's map-production path per protein; model loaded once."""
        import time

        import numpy as np
        import torch

        from sj.eval.contacts import apc_correction
        from sj.probes.head_fusion import fuse_naive_mean
        from sj.probes.model_adapters import make_adapter

        adapter = make_adapter(family, variant)
        load_kwargs = {
            "cache_dir": Path("/vol/weights/hf_cache"),
            "device": "cuda",
            "attn_implementation": "eager",
        }
        import time as _t
        print(f"[{_t.strftime('%H:%M:%S')}] loading {family}/{variant} ...", flush=True)
        model, tokenizer, _ = adapter.load(**load_kwargs)
        print(f"[{_t.strftime('%H:%M:%S')}] loaded; timing {len(proteins)} proteins", flush=True)
        max_pos = int(
            getattr(model.config, "max_position_embeddings", 0)
            or getattr(model.config, "max_length", 0)
            or 0
        )
        topk = [tuple(t) for t in topk_indices][:k]

        def _head_map(attentions, seq_slice, ell: int, h: int) -> np.ndarray:
            a = attentions[ell][0, h, seq_slice, seq_slice].to(torch.float32).cpu().numpy()
            a_sym = 0.5 * (a + a.T)
            a_sym = apc_correction(a_sym.astype(np.float64))
            a_sym = 0.5 * (a_sym + a_sym.T)
            np.fill_diagonal(a_sym, 0.0)
            return a_sym

        def _fusion_map(sequence: str):
            """The timed region: forward -> top-K maps -> naive_mean -> (L,L)."""
            input_ids, seq_slice = adapter.tokenize(tokenizer, sequence)
            attentions = adapter.forward_attention(model, input_ids, bf16=bf16)
            stack = np.stack([_head_map(attentions, seq_slice, int(e), int(h)) for (e, h) in topk])
            fused = fuse_naive_mean(stack)
            n_layers = len(attentions)
            n_heads = int(attentions[0].shape[1])
            return fused, n_layers, n_heads

        # Warmup (kernel autotune / cache) on the first non-skipped protein so
        # the first timed measurement is not inflated by one-time CUDA overhead.
        warmed = False
        out: list[dict] = []
        for _idx, (pid, seq) in enumerate(proteins):
            if _idx % 50 == 0:
                print(f"[{_t.strftime('%H:%M:%S')}] protein {_idx}/{len(proteins)}", flush=True)
            L = len(seq)
            if max_pos and max_pos < L + 2:
                out.append(
                    {
                        "protein_id": pid,
                        "sequence_length": L,
                        "skipped": True,
                        "reason": f"L={L} exceeds {family}/{variant} max_pos={max_pos}",
                    }
                )
                continue
            if not warmed:
                try:
                    _fusion_map(seq)
                except Exception:
                    pass
                warmed = True
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                _fused, n_layers, n_heads = _fusion_map(seq)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                out.append(
                    {
                        "protein_id": pid,
                        "sequence_length": L,
                        "fusion_wall_clock_seconds": float(wall),
                        "fusion_peak_gpu_gb": float(peak_gb),
                        "n_layers": int(n_layers),
                        "n_heads": int(n_heads),
                        "k": int(len(topk)),
                    }
                )
            except Exception as exc:  # OOM or other — record, keep going
                torch.cuda.empty_cache()
                out.append(
                    {
                        "protein_id": pid,
                        "sequence_length": L,
                        "skipped": True,
                        "reason": f"error: {exc!r}",
                    }
                )
        return out

    @_app.local_entrypoint()
    def dispatch(
        variants: str = "35M,150M,650M,3B",
        datasets: str = "zhang_eval200,casp14_fm",
        k: int = 10,
        family: str = "esm2",
    ):
        """Real dispatch via `modal run` (the proven pattern; `with app.run()`
        from a plain script does not execute the function on modal 1.3.5)."""
        topk_file = REPO_ROOT / "results" / "cl7_phase15_topk_heads.json"
        out_dir = REPO_ROOT / "results"
        with topk_file.open() as fh:
            topk_lookup = json.load(fh)
        vlist = [v for v in variants.split(",") if v]
        dlist = [d for d in datasets.split(",") if d]
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant in vlist:
            for dataset in dlist:
                key = _topk_key(family, variant)
                tkd = _topk_dataset_for(dataset)
                if key not in topk_lookup or tkd not in topk_lookup[key]:
                    print(f"  no top-K for {key}/{tkd}; skip {variant}/{dataset}", flush=True)
                    continue
                topk = [tuple(t) for t in topk_lookup[key][tkd]][:k]
                proteins = _proteins_for_dataset(dataset)
                payload = [(ex.protein_id, ex.sequence) for ex in proteins]
                print(f"\n=== timing {key}/{dataset} N={len(payload)} ===", flush=True)
                rows = fusion_timing_batch.remote(family, variant, payload, list(topk), k)
                out_path = out_dir / f"r4_fusion_timing_{variant}_{dataset}.json"
                out_path.write_text(
                    json.dumps(
                        {"family": family, "variant": variant, "dataset": dataset,
                         "k": k, "per_protein": rows},
                        indent=2,
                    )
                )
                timed = [r for r in rows if "fusion_wall_clock_seconds" in r]
                print(f"  wrote {out_path} (timed {len(timed)}/{len(rows)})", flush=True)

except ImportError:
    _modal = None  # type: ignore[assignment]
    _app = None  # type: ignore[assignment]
    fusion_timing_batch = None  # type: ignore[assignment]


def _topk_key(family: str, variant: str) -> str:
    return variant if family == "esm2" else f"{family}_{variant}"


def _topk_dataset_for(dataset: str) -> str:
    if dataset == "zhang_eval200":
        return "zhang_select10"
    if dataset == "cameo_pta_eval":
        return "cameo_pta_select10"
    if dataset in ("zhang_random_50_seed43", "cameo_pta"):
        return "zhang_random_50"
    return dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="r4 fusion paired-timing dispatch.")
    p.add_argument("--family", default="esm2")
    p.add_argument("--variants", nargs="+", default=["35M", "150M", "650M", "3B"])
    p.add_argument("--datasets", nargs="+", default=["zhang_eval200", "casp14_fm"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--topk-file",
        type=Path,
        default=REPO_ROOT / "results" / "cl7_phase15_topk_heads.json",
    )
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    with args.topk_file.open() as fh:
        topk_lookup = json.load(fh)

    # Dry-run cost prior: fusion is one forward/protein. Per-protein seconds
    # scale ~ with the cached CJ time / (19L) but we just need a coarse ceiling.
    _CJ_ANCHOR_S = {"8M": 0.3, "35M": 0.6, "150M": 1.5, "650M": 0.5, "3B": 1.5}
    total_calls = 0
    plan = []
    for variant in args.variants:
        for dataset in args.datasets:
            key = _topk_key(args.family, variant)
            tkd = _topk_dataset_for(dataset)
            if key not in topk_lookup or tkd not in topk_lookup[key]:
                print(f"  no top-K cache for {key}/{tkd}; skipping {variant}/{dataset}")
                continue
            proteins = _proteins_for_dataset(dataset)
            plan.append((variant, dataset, key, tkd, len(proteins)))
            total_calls += 1

    print("\n=== r4 fusion-timing plan ===")
    for variant, dataset, key, tkd, n in plan:
        anchor = _CJ_ANCHOR_S.get(variant, 1.0)
        # ~ n forwards + 1 model load (~60-90s); coarse ceiling.
        est_s = n * anchor + 90
        print(f"  {key:>14} / {dataset:<14} N={n:<4} topk<-{tkd:<18} ~{est_s:.0f}s GPU")
    grand = sum(
        n * _CJ_ANCHOR_S.get(v, 1.0) + 90 for (v, d, k_, t, n) in plan
    )
    print(f"  TOTAL ~{grand:.0f}s GPU = {grand/3600:.2f} GPU-hr  ~${grand/3600*4.0:.2f} at $4/hr")

    if args.dry_run:
        return 0
    if _modal is None or fusion_timing_batch is None:
        print("modal not available", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with _app.run():
        for variant, dataset, key, tkd, _n in plan:
            topk = [tuple(t) for t in topk_lookup[key][tkd]][: args.k]
            proteins = _proteins_for_dataset(dataset)
            payload = [(ex.protein_id, ex.sequence) for ex in proteins]
            print(f"\n=== timing {key} / {dataset} (N={len(payload)}) ===")
            rows = fusion_timing_batch.remote(args.family, variant, payload, list(topk), args.k)
            out_path = args.out_dir / f"r4_fusion_timing_{variant}_{dataset}.json"
            out_path.write_text(json.dumps({"family": args.family, "variant": variant,
                                            "dataset": dataset, "k": args.k,
                                            "per_protein": rows}, indent=2))
            timed = [r for r in rows if "fusion_wall_clock_seconds" in r]
            skipped = [r for r in rows if r.get("skipped")]
            if timed:
                ws = [r["fusion_wall_clock_seconds"] for r in timed]
                ms = [r["fusion_peak_gpu_gb"] for r in timed]
                print(f"  timed {len(timed)}  wall[min/med/max]="
                      f"{min(ws):.3f}/{sorted(ws)[len(ws)//2]:.3f}/{max(ws):.3f}s  "
                      f"peak_gb[max]={max(ms):.2f}  skipped={len(skipped)}")
            print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
