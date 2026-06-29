"""r4 stitching experiments: attention-fusion contact prediction beyond the
context window via overlapping-window tiling + stitch.

E1 (chunking-penalty ablation, Zhang eval-200, full forward feasible): for each
protein compute the full-forward fusion map (baseline) and stitched maps at
several window sizes W; report full vs stitched P@L/2 long and the fraction of
true long-range contacts with |i-j|>W (the inherent window ceiling).

E2 (real long proteins, CASP14-FM L>1024): the two targets (EXT-CAS9 L=1372,
T1044 L=2180) that exceed ESM-2's 1024 context and cannot be run in one forward;
stitched fusion produces a contact map and is scored vs ground truth -- a
proof-of-concept that the method reaches proteins beyond the context window.

Stitch: windows of length W (<=1024), stride W*(1-overlap); per-window fusion
sub-map (top-K heads, APC within window); overlap-averaged into the (L,L) map.
Pairs with |i-j|>W are never co-covered (NaN -> excluded), the honest ceiling.

Run: modal run scripts/r4_stitch.py --n-eval 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def _zhang_eval(n: int):
    from sj.data.zhang_splits import eval_proteins

    ps = list(eval_proteins())
    if n >= len(ps):
        return ps
    ordered = sorted(ps, key=lambda e: e.length)
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(idx.tolist()))]


def _casp_long():
    from sj.data.casp14_fm import CASP14FMLoader, default_casp14_fm_root

    return [e for e in CASP14FMLoader(data_root=default_casp14_fm_root()) if e.length > 1024]


try:
    import modal as _modal

    _image = (
        _modal.Image.from_registry("python:3.11-slim-bookworm")
        .pip_install_from_requirements(str(REPO_ROOT / "requirements.lock"))
        .add_local_python_source("sj", "baselines")
        .add_local_dir(str(REPO_ROOT / "checkpoints"), "/checkpoints")
    )
    _app = _modal.App("plm-contact-fusion-r4-stitch", image=_image)
    _hf = _modal.Secret.from_name("hf_token")
    _wvol = _modal.Volume.from_name("plm-contact-fusion-weights", create_if_missing=True)

    @_app.function(
        gpu="A100-80GB", timeout=10800, secrets=[_hf],
        volumes={"/vol/weights": _wvol},
        max_containers=int(__import__("os").environ.get("SJ_MAX_CONTAINERS", "4")),
    )
    def stitch_batch(
        variant: str,
        proteins: list,
        topk_indices: list,
        k: int,
        window_sizes: list,
        overlap: float,
        full_baseline: bool,
        bf16: bool = True,
    ) -> list:
        import io
        import time

        import numpy as np
        import torch

        from sj.eval.contacts import apc_correction, top_lk_precision
        from sj.probes.head_fusion import fuse_naive_mean
        from sj.probes.model_adapters import make_adapter

        adapter = make_adapter("esm2", variant)
        print(f"[{time.strftime('%H:%M:%S')}] loading esm2/{variant} (eager)...", flush=True)
        model, tok, _ = adapter.load(
            cache_dir=Path("/vol/weights/hf_cache"), device="cuda", attn_implementation="eager"
        )
        device = next(model.parameters()).device
        max_pos = int(getattr(model.config, "max_position_embeddings", 1026) or 1026)
        feasible = max_pos - 2
        topk = [tuple(t) for t in topk_indices][:k]
        print(f"[{time.strftime('%H:%M:%S')}] loaded; feasible L<= {feasible}; {len(proteins)} proteins",
              flush=True)

        def _head_map(att, sl, ell, h):
            a = att[ell][0, h, sl, sl].to(torch.float32).cpu().numpy()
            a = 0.5 * (a + a.T)
            a = apc_correction(a.astype(np.float64))
            a = 0.5 * (a + a.T)
            np.fill_diagonal(a, 0.0)
            return a

        def _fusion_submap(subseq):
            ids, sl = adapter.tokenize(tok, subseq)
            att = adapter.forward_attention(model, ids.to(device), bf16=bf16)
            stack = np.stack([_head_map(att, sl, int(e), int(h)) for (e, h) in topk])
            return fuse_naive_mean(stack)

        def _windows(L, W):
            stride = max(1, int(round(W * (1.0 - overlap))))
            starts = list(range(0, max(1, L - W + 1), stride))
            last = max(0, L - W)
            if not starts or starts[-1] != last:
                starts.append(last)
            return sorted(set(s for s in starts if 0 <= s <= last)), stride

        def _stitch(seq, W):
            L = len(seq)
            acc = np.zeros((L, L)); cnt = np.zeros((L, L))
            starts, _ = _windows(L, W)
            for s in starts:
                e = min(s + W, L)
                sub = _fusion_submap(seq[s:e])
                acc[s:e, s:e] += sub
                cnt[s:e, s:e] += 1.0
            m = cnt > 0
            out = np.full((L, L), np.nan)
            out[m] = acc[m] / cnt[m]
            return out, len(starts)

        def _score(M, contacts, valid, L):
            Ms = np.where(np.isnan(M), -1e9, M)
            r = top_lk_precision(Ms.astype(np.float64), contacts, range_name="long", k=2,
                                 valid_residues=valid)
            return float(r.precision)

        def _frac_beyond(contacts, valid, L, W):
            ii, jj = np.indices((L, L))
            long = (jj - ii) >= 24
            vm = valid[:, None] & valid[None, :]
            mask = long & vm
            true = contacts & mask
            n = int(true.sum())
            if n == 0:
                return float("nan")
            beyond = int((true & ((jj - ii) > W)).sum())
            return beyond / n

        out = []
        for i, (pid, seq, gt) in enumerate(proteins):
            L = len(seq)
            with np.load(io.BytesIO(gt)) as data:
                contacts = data["contacts"].astype(bool)
                valid = (data["valid_residues"].astype(bool)
                         if "valid_residues" in data.files else np.ones(L, dtype=bool))
            print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(proteins)} {pid} L={L}", flush=True)
            rec = {"protein_id": pid, "sequence_length": L, "stitched": {}}
            try:
                if full_baseline and L <= feasible:
                    full = _fusion_submap(seq)  # one forward, whole protein
                    rec["full_p"] = _score(full, contacts, valid, L)
                for W in window_sizes:
                    if W > feasible:
                        continue
                    t0 = time.perf_counter()
                    M, nwin = _stitch(seq, W)
                    rec["stitched"][str(W)] = {
                        "p": _score(M, contacts, valid, L),
                        "n_windows": nwin,
                        "frac_true_beyond_W": _frac_beyond(contacts, valid, L, W),
                        "stitch_wall_s": round(time.perf_counter() - t0, 3),
                    }
                out.append(rec)
            except Exception as exc:
                torch.cuda.empty_cache()
                out.append({"protein_id": pid, "sequence_length": L, "skipped": True,
                            "reason": f"error: {exc!r}"})
        return out

    @_app.local_entrypoint()
    def dispatch(variant: str = "650M", n_eval: int = 50, overlap: float = 0.5, k: int = 10):
        import io
        import json

        topk = json.loads((REPO_ROOT / "results" / "cl7_phase15_topk_heads.json").read_text())
        topk = topk[variant]["zhang_select10"][:k]

        def payload(proteins):
            out = []
            for ex in proteins:
                buf = io.BytesIO()
                np.savez_compressed(buf, contacts=ex.contact_map.astype(bool),
                                    valid_residues=(ex.valid_residues.astype(bool)
                                                    if ex.valid_residues is not None
                                                    else np.ones(ex.length, dtype=bool)))
                out.append((ex.protein_id, ex.sequence, buf.getvalue()))
            return out

        out_dir = REPO_ROOT / "results"
        # E1: chunking-penalty ablation on Zhang eval-200 (full forward feasible)
        e1_prot = _zhang_eval(n_eval)
        print(f"=== E1 stitch-ablation Zhang eval-200 N={len(e1_prot)} ===", flush=True)
        e1 = stitch_batch.remote(variant, payload(e1_prot), list(topk), k,
                                 [128, 192, 256, 384], overlap, True)
        (out_dir / "r4_stitch_e1_zhang.json").write_text(
            json.dumps({"variant": variant, "overlap": overlap, "per_protein": e1}, indent=2))
        # E2: real long proteins (L>1024, infeasible in one forward); reuse if done.
        if not (out_dir / "r4_stitch_e2_long.json").exists():
            e2_prot = _casp_long()
            print(f"=== E2 stitch real long CASP14 N={len(e2_prot)} ===", flush=True)
            e2 = stitch_batch.remote(variant, payload(e2_prot), list(topk), k,
                                     [768, 1000], overlap, False)
            (out_dir / "r4_stitch_e2_long.json").write_text(
                json.dumps({"variant": variant, "overlap": overlap, "per_protein": e2}, indent=2))
        print("wrote r4_stitch_e1_zhang.json (+ E2 reused)", flush=True)

except ImportError:
    _modal = None
    _app = None
    stitch_batch = None


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    e1 = _zhang_eval(args.n_eval)
    e2 = _casp_long()
    # rough forward counts: E1 full(1)+windows; E2 windows.
    print(f"E1 Zhang N={len(e1)} (L {min(e.length for e in e1)}-{max(e.length for e in e1)})")
    print(f"E2 long CASP14 N={len(e2)}: {[(e.protein_id, e.length) for e in e2]}")
    print("cost: fusion forwards are ~0.05-1s each; E1 ~ a few hundred forwards, E2 ~tens. ~$1-2.")
    print("run: modal run scripts/r4_stitch.py --n-eval 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
