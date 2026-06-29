"""r4 Job 2: repr-CJ pair-level agreement + WtW intermediate baseline.

Grounds the r4 metric-geometry theory paragraph empirically. One perturbation
sweep per protein produces THREE (L,L) contact maps under a single, shared
post-processing pipeline (mean over alt AAs -> symmetrize -> APC -> zero diag),
so the comparison isolates the readout METRIC, not the pipeline:

  - repr-CJ  : per-position response = ||dh||_2                 (Euclidean metric I)
  - WtW-CJ   : per-position response = ||W dh||_2               (Mahalanobis W^T W;
               W = lm_head.decoder.weight, the tied output embedding -- the exact
               "z = W h" linear head in the theory paragraph)
  - logit-CJ : per-position response = ||d logits[AA]||_2       (TRUE head:
               dense -> gelu -> layer_norm -> decoder; the full nonlinear head)

Theory decomposition the three readouts test:
  repr  vs WtW   -> how anisotropic W^T W is (Euclidean == Mahalanobis iff isotropic)
  WtW   vs logit -> how much the dense+gelu+layernorm nonlinearity contributes

Per protein we return scalars only (precisions + pairwise agreement: top-L/2 long
Jaccard, Pearson on long-range pair scores), so failure cases = low-agreement
proteins, identifiable directly. Maps stay in-container.

Applies only to MLM families with an EsmLMHead-style decoder (esm2, esm1b).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

def _proteins_for_dataset(dataset: str):  # inlined to avoid cross-script import
    if dataset == "zhang_eval200":
        from sj.data.zhang_splits import eval_proteins

        return list(eval_proteins())
    if dataset == "casp14_fm":
        from sj.data.casp14_fm import CASP14FMLoader, default_casp14_fm_root

        return list(CASP14FMLoader(data_root=default_casp14_fm_root()))
    raise ValueError(f"unknown dataset {dataset!r}")


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_APP_NAME = "plm-contact-fusion-r4-cjmaps"

try:
    import modal as _modal  # type: ignore[import-untyped]

    _lock_path = REPO_ROOT / "requirements.lock"
    _checkpoints_dir = REPO_ROOT / "checkpoints"
    # ESM-2-only image: no xformers/sentencepiece/tiktoken (ProtT5/ProGen2 deps;
    # xformers builds from source ~30 min). esm2 adapter loads via transformers.
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
        timeout=10800,
        secrets=[_hf_secret],
        volumes={"/vol/weights": _weights_volume},
        max_containers=int(os.environ.get("SJ_MAX_CONTAINERS", "4")),
    )
    def cj_maps_batch(
        family: str,
        variant: str,
        proteins: list[tuple[str, str, bytes]],
        layer: int = -1,
        max_perturbs_per_batch: int = 64,
        bf16: bool = True,
    ) -> list[dict]:
        import time

        import numpy as np
        import torch

        from sj.eval.contacts import apc_correction, top_lk_precision
        from sj.probes.model_adapters import make_adapter

        adapter = make_adapter(family, variant)
        import time as _t
        print(f"[{_t.strftime('%H:%M:%S')}] loading {family}/{variant} ...", flush=True)
        model, tokenizer, _ = adapter.load(
            cache_dir=Path("/vol/weights/hf_cache"),
            device="cuda",
            attn_implementation=None,
        )
        device = next(model.parameters()).device
        print(f"[{_t.strftime('%H:%M:%S')}] loaded; {len(proteins)} proteins (~19L fwd each)", flush=True)
        max_pos = int(
            getattr(model.config, "max_position_embeddings", 0)
            or getattr(model.config, "max_length", 0)
            or 0
        )

        # W = tied output embedding (decoder weight), restricted to 20 AA rows.
        aa_ids = [tokenizer.convert_tokens_to_ids(a) for a in STANDARD_AA]
        decoder = model.lm_head.decoder  # EsmLMHead.decoder: Linear(D, vocab, bias=False)
        W_full = decoder.weight.detach()  # (vocab, D)
        W_aa = W_full[aa_ids, :].to(torch.float32)  # (20, D)

        def _postprocess(score_LL: np.ndarray) -> np.ndarray:
            s = 0.5 * (score_LL + score_LL.T)
            s = apc_correction(s.astype(np.float64))
            s = 0.5 * (s + s.T)
            np.fill_diagonal(s, 0.0)
            return s.astype(np.float64)

        def _long_mask(L: int) -> np.ndarray:
            ii, jj = np.indices((L, L))
            return (jj - ii) >= 24  # upper-triangle long-range

        def _topl2_long_set(score: np.ndarray, L: int) -> set:
            m = _long_mask(L)
            idx = np.argwhere(m)
            vals = score[m]
            k = max(1, L // 2)
            order = np.argsort(-vals)[:k]
            return {tuple(idx[o]) for o in order}

        def _jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 1.0
            return len(a & b) / len(a | b)

        def _pearson_long(x: np.ndarray, y: np.ndarray, L: int) -> float:
            m = _long_mask(L)
            xv, yv = x[m], y[m]
            if xv.std() < 1e-12 or yv.std() < 1e-12:
                return float("nan")
            return float(np.corrcoef(xv, yv)[0, 1])

        out: list[dict] = []
        for _i, (pid, seq, gt_bytes) in enumerate(proteins):
            print(f"[{_t.strftime('%H:%M:%S')}] protein {_i+1}/{len(proteins)} {pid} L={len(seq)}", flush=True)
            L = len(seq)
            if max_pos and max_pos < L + 2:
                out.append({"protein_id": pid, "sequence_length": L, "skipped": True,
                            "reason": f"L={L} exceeds max_pos={max_pos}"})
                continue
            with np.load(io.BytesIO(gt_bytes)) as data:
                contacts = data["contacts"].astype(bool)
                valid = (data["valid_residues"].astype(bool)
                         if "valid_residues" in data.files else np.ones(L, dtype=bool))
            try:
                native_ids, seq_slice = adapter.tokenize(tokenizer, seq)

                def _forward(ids):
                    ids = ids.to(device)
                    with (torch.inference_mode(),
                          torch.autocast(device_type=device.type,
                                         dtype=torch.bfloat16, enabled=bf16)):
                        o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  output_hidden_states=True)
                    h = o.hidden_states[layer][:, seq_slice, :].to(torch.float32)
                    lg = o.logits[:, seq_slice, :][..., aa_ids].to(torch.float32)
                    return h, lg  # (B,L,D), (B,L,20)

                t0 = time.perf_counter()
                h_nat, lg_nat = _forward(native_ids)
                h_nat, lg_nat = h_nat[0], lg_nat[0]  # (L,D),(L,20)
                forwards = 1

                raw_repr = np.zeros((L, 20, L), dtype=np.float32)
                raw_wtw = np.zeros((L, 20, L), dtype=np.float32)
                raw_logit = np.zeros((L, 20, L), dtype=np.float32)
                work = [(i, a) for i in range(L) for a in range(20)]
                for cs in range(0, len(work), max_perturbs_per_batch):
                    chunk = work[cs:cs + max_perturbs_per_batch]
                    ids_list = []
                    for i, a in chunk:
                        sp = seq[:i] + STANDARD_AA[a] + seq[i + 1:]
                        ids_p, _ = adapter.tokenize(tokenizer, sp)
                        ids_list.append(ids_p[0])
                    batch = torch.stack(ids_list, dim=0)
                    h_p, lg_p = _forward(batch)  # (B,L,D),(B,L,20)
                    forwards += 1
                    dh = h_p - h_nat.unsqueeze(0)  # (B,L,D)
                    dlg = lg_p - lg_nat.unsqueeze(0)  # (B,L,20)
                    repr_resp = dh.norm(dim=-1)  # (B,L)
                    wtw_resp = torch.einsum("ad,bld->bla", W_aa, dh).norm(dim=-1)  # (B,L)
                    logit_resp = dlg.norm(dim=-1)  # (B,L)
                    repr_resp = repr_resp.cpu().numpy()
                    wtw_resp = wtw_resp.cpu().numpy()
                    logit_resp = logit_resp.cpu().numpy()
                    for k, (i, a) in enumerate(chunk):
                        raw_repr[i, a] = repr_resp[k]
                        raw_wtw[i, a] = wtw_resp[k]
                        raw_logit[i, a] = logit_resp[k]

                maps = {}
                for name, raw in (("repr", raw_repr), ("wtw", raw_wtw), ("logit", raw_logit)):
                    maps[name] = _postprocess(raw.mean(axis=1))
                wall = time.perf_counter() - t0

                prec = {}
                for name, mp in maps.items():
                    r = top_lk_precision(mp, contacts, range_name="long", k=2,
                                         valid_residues=valid)
                    prec[name] = float(r.precision)

                sets = {n: _topl2_long_set(maps[n], L) for n in maps}
                jac = {
                    "logit_repr": _jaccard(sets["logit"], sets["repr"]),
                    "logit_wtw": _jaccard(sets["logit"], sets["wtw"]),
                    "repr_wtw": _jaccard(sets["repr"], sets["wtw"]),
                }
                pear = {
                    "logit_repr": _pearson_long(maps["logit"], maps["repr"], L),
                    "logit_wtw": _pearson_long(maps["logit"], maps["wtw"], L),
                    "repr_wtw": _pearson_long(maps["repr"], maps["wtw"], L),
                }
                out.append({
                    "protein_id": pid, "sequence_length": L, "skipped": False,
                    "precision": prec, "jaccard_topl2_long": jac,
                    "pearson_long_pairs": pear,
                    "wall_clock_seconds": float(wall), "forwards_count": int(forwards),
                })
            except Exception as exc:
                torch.cuda.empty_cache()
                out.append({"protein_id": pid, "sequence_length": L, "skipped": True,
                            "reason": f"error: {exc!r}"})
        return out

    @_app.local_entrypoint()
    def dispatch(
        variant: str = "650M",
        dataset: str = "zhang_eval200",
        n_proteins: int = 50,
        family: str = "esm2",
        max_perturbs_per_batch: int = 64,
    ):
        """Real dispatch via `modal run` (with app.run() does not execute on 1.3.5)."""
        import io as _io

        import numpy as _np

        out_dir = REPO_ROOT / "results"
        proteins = _stratified_subsample(_proteins_for_dataset(dataset), n_proteins)
        payload = []
        for ex in proteins:
            buf = _io.BytesIO()
            _np.savez_compressed(
                buf,
                contacts=ex.contact_map.astype(bool),
                valid_residues=(
                    ex.valid_residues.astype(bool)
                    if ex.valid_residues is not None
                    else _np.ones(ex.length, dtype=bool)
                ),
            )
            payload.append((ex.protein_id, ex.sequence, buf.getvalue()))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== cj-maps {family}/{variant} {dataset} N={len(payload)} ===", flush=True)
        rows = cj_maps_batch.remote(family, variant, payload, -1, max_perturbs_per_batch)
        out_path = out_dir / f"r4_cj_maps_{variant}_{dataset}.json"
        out_path.write_text(
            json.dumps(
                {"family": family, "variant": variant, "dataset": dataset,
                 "n_proteins": len(payload), "per_protein": rows},
                indent=2,
            )
        )
        scored = [r for r in rows if not r.get("skipped")]
        print(f"wrote {out_path} (scored {len(scored)}/{len(rows)})", flush=True)

except ImportError:
    _modal = None  # type: ignore[assignment]
    _app = None  # type: ignore[assignment]
    cj_maps_batch = None  # type: ignore[assignment]


def _stratified_subsample(proteins, n: int, seed: int = 42):
    """Pick n proteins spread across the length range (deterministic)."""
    if n >= len(proteins):
        return list(proteins)
    ordered = sorted(proteins, key=lambda ex: ex.length)
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(idx.tolist()))]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="r4 repr/WtW/logit-CJ pair-level agreement.")
    p.add_argument("--family", default="esm2")
    p.add_argument("--variant", required=True, help="e.g. 650M or 3B")
    p.add_argument("--dataset", default="zhang_eval200")
    p.add_argument("--n-proteins", type=int, default=50)
    p.add_argument("--max-perturbs-per-batch", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    proteins = _stratified_subsample(_proteins_for_dataset(args.dataset), args.n_proteins)
    Ls = [ex.length for ex in proteins]
    # Cost prior from cached CJ s/protein anchors (~19L forwards).
    anchor = {"650M": 33.0, "3B": 100.0, "150M": 8.0, "35M": 3.0}.get(args.variant, 50.0)
    # scale anchor (measured at mean L~300) by this subsample's mean L.
    scale = (sum(Ls) / len(Ls)) / 300.0
    est_s = len(proteins) * anchor * scale + 90
    print(f"=== r4 cj-maps {args.family}/{args.variant} {args.dataset} ===")
    print(f"  N={len(proteins)}  L=[{min(Ls)},{max(Ls)}] mean={sum(Ls)/len(Ls):.0f}")
    print(f"  est ~{est_s:.0f}s GPU = {est_s/3600:.2f} GPU-hr  ~${est_s/3600*4.0:.2f} at $4/hr")
    if args.dry_run:
        return 0
    if _modal is None or cj_maps_batch is None:
        print("modal not available", file=sys.stderr)
        return 1

    payload = []
    for ex in proteins:
        buf = io.BytesIO()
        np.savez_compressed(
            buf, contacts=ex.contact_map.astype(bool),
            valid_residues=(ex.valid_residues.astype(bool)
                            if ex.valid_residues is not None else np.ones(ex.length, dtype=bool)),
        )
        payload.append((ex.protein_id, ex.sequence, buf.getvalue()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with _app.run():
        rows = cj_maps_batch.remote(args.family, args.variant, payload, -1,
                                    args.max_perturbs_per_batch)
    out_path = args.out_dir / f"r4_cj_maps_{args.variant}_{args.dataset}.json"
    out_path.write_text(json.dumps({"family": args.family, "variant": args.variant,
                                    "dataset": args.dataset, "n_proteins": len(payload),
                                    "per_protein": rows}, indent=2))
    scored = [r for r in rows if not r.get("skipped")]
    if scored:
        def _mean(key, sub):
            vals = [r[key][sub] for r in scored if not np.isnan(r[key][sub])]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"\n  scored {len(scored)}/{len(rows)}")
        print(f"  mean precision: logit={_mean('precision','logit'):.3f} "
              f"repr={_mean('precision','repr'):.3f} wtw={_mean('precision','wtw'):.3f}")
        print(f"  mean top-L/2 Jaccard: logit~repr={_mean('jaccard_topl2_long','logit_repr'):.3f} "
              f"logit~wtw={_mean('jaccard_topl2_long','logit_wtw'):.3f} "
              f"repr~wtw={_mean('jaccard_topl2_long','repr_wtw'):.3f}")
        print(f"  mean Pearson(long pairs): logit~repr={_mean('pearson_long_pairs','logit_repr'):.3f} "
              f"logit~wtw={_mean('pearson_long_pairs','logit_wtw'):.3f} "
              f"repr~wtw={_mean('pearson_long_pairs','repr_wtw'):.3f}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
