"""r4 native-long-context check: ProtT5-XL fusion on long proteins in ONE forward.

ProtT5 uses T5 relative-position attention -> no hard context cap, so it can run
the two CASP14-FM targets that exceed ESM-2's 1024 window (EXT-CAS9 L=1372,
T1044 L=2180) in a single forward, no stitching. We run top-K attention fusion
(same zhang_select10 head selection regime as the ESM-2 stitch) and score
P@L/2 long, to put native-long-context vs stitched side by side. A few short
CASP14 targets are included as a sanity check that ProtT5 fusion behaves.

Run: modal run scripts/r4_longctx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def _casp():
    from sj.data.casp14_fm import CASP14FMLoader, default_casp14_fm_root

    return list(CASP14FMLoader(data_root=default_casp14_fm_root()))


try:
    import modal as _modal

    _image = (
        _modal.Image.from_registry("python:3.11-slim-bookworm")
        .pip_install_from_requirements(str(REPO_ROOT / "requirements.lock"))
        .pip_install("sentencepiece>=0.1.99")  # ProtT5 T5 tokenizer needs it
        .add_local_python_source("sj", "baselines")
        .add_local_dir(str(REPO_ROOT / "checkpoints"), "/checkpoints")
    )
    _app = _modal.App("plm-contact-fusion-r4-longctx", image=_image)
    _hf = _modal.Secret.from_name("hf_token")
    _wvol = _modal.Volume.from_name("plm-contact-fusion-weights", create_if_missing=True)

    @_app.function(
        gpu="A100-80GB", timeout=3600, secrets=[_hf],
        volumes={"/vol/weights": _wvol},
        max_containers=int(__import__("os").environ.get("SJ_MAX_CONTAINERS", "2")),
    )
    def fusion_prott5(proteins: list, topk_indices: list, k: int, bf16: bool = True) -> list:
        import io
        import time

        import numpy as np
        import torch

        from sj.eval.contacts import apc_correction, top_lk_precision
        from sj.probes.head_fusion import fuse_naive_mean
        from sj.probes.model_adapters import make_adapter

        adapter = make_adapter("prott5", "XL")
        print(f"[{time.strftime('%H:%M:%S')}] loading ProtT5-XL ...", flush=True)
        model, tok, _ = adapter.load(
            cache_dir=Path("/vol/weights/hf_cache"), device="cuda", attn_implementation="eager"
        )
        device = next(model.parameters()).device
        topk = [tuple(t) for t in topk_indices][:k]
        print(f"[{time.strftime('%H:%M:%S')}] loaded; {len(proteins)} proteins", flush=True)

        def _head_map(att, sl, ell, h):
            a = att[ell][0, h, sl, sl].to(torch.float32).cpu().numpy()
            a = 0.5 * (a + a.T)
            a = apc_correction(a.astype(np.float64))
            a = 0.5 * (a + a.T)
            np.fill_diagonal(a, 0.0)
            return a

        out = []
        for i, (pid, seq, gt) in enumerate(proteins):
            L = len(seq)
            with np.load(io.BytesIO(gt)) as data:
                contacts = data["contacts"].astype(bool)
                valid = (data["valid_residues"].astype(bool)
                         if "valid_residues" in data.files else np.ones(L, dtype=bool))
            print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(proteins)} {pid} L={L}", flush=True)
            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                ids, sl = adapter.tokenize(tok, seq)
                att = adapter.forward_attention(model, ids.to(device), bf16=bf16)
                stack = np.stack([_head_map(att, sl, int(e), int(h)) for (e, h) in topk])
                fused = fuse_naive_mean(stack)
                wall = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 1e9
                r = top_lk_precision(fused.astype(np.float64), contacts, range_name="long",
                                     k=2, valid_residues=valid)
                out.append({"protein_id": pid, "sequence_length": L, "skipped": False,
                            "p_long": float(r.precision), "n_layers": len(att),
                            "n_heads": int(att[0].shape[1]),
                            "wall_s": round(wall, 2), "peak_gpu_gb": round(float(peak), 2)})
                del att
                torch.cuda.empty_cache()
            except Exception as exc:
                torch.cuda.empty_cache()
                out.append({"protein_id": pid, "sequence_length": L, "skipped": True,
                            "reason": f"{type(exc).__name__}: {str(exc)[:120]}"})
        return out

    @_app.local_entrypoint()
    def dispatch(k: int = 10, n_short: int = 4):
        import io
        import json

        topk = json.loads((REPO_ROOT / "results" / "cl7_phase15_topk_heads.json").read_text())
        topk = topk["prott5_XL"]["zhang_select10"][:k]
        casp = _casp()
        longs = sorted([e for e in casp if e.length > 1024], key=lambda e: e.length)
        shorts = sorted([e for e in casp if e.length <= 1024], key=lambda e: e.length)
        # sanity: a few short targets spread by length
        pick = shorts[:: max(1, len(shorts) // n_short)][:n_short] if shorts else []
        proteins = longs + pick
        payload = []
        for ex in proteins:
            buf = io.BytesIO()
            np.savez_compressed(buf, contacts=ex.contact_map.astype(bool),
                                valid_residues=(ex.valid_residues.astype(bool)
                                                if ex.valid_residues is not None
                                                else np.ones(ex.length, dtype=bool)))
            payload.append((ex.protein_id, ex.sequence, buf.getvalue()))
        print(f"=== ProtT5-XL fusion: {len(longs)} long + {len(pick)} short sanity ===", flush=True)
        rows = fusion_prott5.remote(payload, list(topk), k)
        (REPO_ROOT / "results" / "r4_longctx_prott5.json").write_text(
            json.dumps({"model": "prott5_XL", "k": k, "per_protein": rows}, indent=2))
        for r in rows:
            if r.get("skipped"):
                print(f"  {r['protein_id']} L={r['sequence_length']}: SKIPPED {r.get('reason')}")
            else:
                print(f"  {r['protein_id']} L={r['sequence_length']}: P@L/2={r['p_long']:.3f} "
                      f"(1 forward, {r['wall_s']}s, {r['peak_gpu_gb']}GB)")
        print("wrote r4_longctx_prott5.json", flush=True)

except ImportError:
    _modal = None
    _app = None
    fusion_prott5 = None


def main(argv=None):
    casp = _casp()
    longs = [e for e in casp if e.length > 1024]
    print(f"long targets (L>1024): {[(e.protein_id, e.length) for e in longs]}")
    print("ProtT5-XL: T5 relative positions, no hard context cap -> single forward each.")
    print("run: modal run scripts/r4_longctx.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
