"""r4 confidence analysis: per-method (L,L) maps + confidence metrics (ESM-2-650M).

For each protein, produces FIVE score maps under matched post-processing
(symmetrize -> APC -> zero-diag):
  fusion (mean top-K APC'd attention), top1 (best single head),
  logit-CJ (||d logits||), repr-CJ (||dh||), WtW-CJ (||W dh||).
Fusion/top1 come from one eager attention forward; the CJ trio from one ~19L
perturbation sweep. Eager attn is used throughout (needed for output_attentions).

Per method, per protein, computes "confidence" on three axes:
  - sharpness: cutoff_z = (score at rank L/2 - mean)/std; gini of the score mass
  - separation: dprime between true-contact and non-contact scores
  - calibration: precision per score-decile (top_decile_precision + full curve)
plus P@L/2 long for reference. Saves full fp16 maps for a few example proteins.

Run: modal run scripts/r4_confidence_maps.py --n-proteins 40 --n-save 6
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def _proteins_for_dataset(dataset: str):
    if dataset == "zhang_eval200":
        from sj.data.zhang_splits import eval_proteins

        return list(eval_proteins())
    raise ValueError(f"unknown dataset {dataset!r}")


def _stratified(proteins, n: int):
    if n >= len(proteins):
        return list(proteins)
    ordered = sorted(proteins, key=lambda ex: ex.length)
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(idx.tolist()))]


try:
    import modal as _modal

    _image = (
        _modal.Image.from_registry("python:3.11-slim-bookworm")
        .pip_install_from_requirements(str(REPO_ROOT / "requirements.lock"))
        .add_local_python_source("sj", "baselines")
        .add_local_dir(str(REPO_ROOT / "checkpoints"), "/checkpoints")
    )
    _app = _modal.App("plm-contact-fusion-r4-confmaps", image=_image)
    _hf = _modal.Secret.from_name("hf_token")
    _wvol = _modal.Volume.from_name("plm-contact-fusion-weights", create_if_missing=True)

    @_app.function(
        gpu="A100-80GB", timeout=10800, secrets=[_hf],
        volumes={"/vol/weights": _wvol},
        max_containers=int(__import__("os").environ.get("SJ_MAX_CONTAINERS", "4")),
    )
    def confidence_batch(
        variant: str,
        proteins: list,
        topk_indices: list,
        k: int,
        save_ids: list,
        max_perturbs_per_batch: int = 64,
        bf16: bool = True,
    ) -> list:
        import io
        import time

        import numpy as np
        import torch

        from sj.eval.contacts import apc_correction
        from sj.probes.head_fusion import fuse_naive_mean
        from sj.probes.model_adapters import make_adapter

        adapter = make_adapter("esm2", variant)
        print(f"[{time.strftime('%H:%M:%S')}] loading esm2/{variant} (eager)...", flush=True)
        model, tok, _ = adapter.load(
            cache_dir=Path("/vol/weights/hf_cache"), device="cuda", attn_implementation="eager"
        )
        device = next(model.parameters()).device
        aa_ids = [tok.convert_tokens_to_ids(a) for a in STANDARD_AA]
        W_aa = model.lm_head.decoder.weight.detach()[aa_ids, :].to(torch.float32)
        topk = [tuple(t) for t in topk_indices][:k]
        save_ids = set(save_ids)
        print(f"[{time.strftime('%H:%M:%S')}] loaded; {len(proteins)} proteins", flush=True)

        def _post(score_LL):
            s = 0.5 * (score_LL + score_LL.T)
            s = apc_correction(s.astype(np.float64))
            s = 0.5 * (s + s.T)
            np.fill_diagonal(s, 0.0)
            return s.astype(np.float64)

        def _head_map(attn, sl, ell, h):
            a = attn[ell][0, h, sl, sl].to(torch.float32).cpu().numpy()
            return _post(a)

        def _conf(M, contacts, valid, L):
            ii, jj = np.indices((L, L))
            long = (jj - ii) >= 24
            if valid is not None:
                vm = valid[:, None] & valid[None, :]
                long = long & vm
            s = M[long].astype(np.float64)
            y = contacts[long].astype(bool)
            if len(s) < 4 or y.sum() == 0:
                return None
            kk = max(1, L // 2)
            order = np.argsort(-s)
            precision = float(y[order[:kk]].mean())
            mu, sd = float(s.mean()), float(s.std() + 1e-12)
            cutoff_z = float((np.sort(s)[::-1][kk - 1] - mu) / sd)
            sh = s - s.min()
            ssum = sh.sum() + 1e-12
            ss = np.sort(sh)
            n = len(sh)
            gini = float((2.0 * np.sum(np.arange(1, n + 1) * ss) / (n * ssum)) - (n + 1) / n)
            s1, s0 = s[y], s[~y]
            dprime = float((s1.mean() - s0.mean()) / np.sqrt(0.5 * (s1.var() + s0.var()) + 1e-12))
            deciles = np.array_split(order, 10)
            dec_prec = [float(y[d].mean()) if len(d) else float("nan") for d in deciles]
            return {
                "precision": precision, "cutoff_z": cutoff_z, "gini": gini,
                "dprime": dprime, "top_decile_precision": dec_prec[0],
                "decile_precisions": dec_prec,
            }

        out = []
        for i, (pid, seq, gt) in enumerate(proteins):
            L = len(seq)
            with np.load(io.BytesIO(gt)) as data:
                contacts = data["contacts"].astype(bool)
                valid = (data["valid_residues"].astype(bool)
                         if "valid_residues" in data.files else np.ones(L, dtype=bool))
            if i % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] protein {i+1}/{len(proteins)} {pid} L={L}",
                      flush=True)
            try:
                ids, sl = adapter.tokenize(tok, seq)
                ids_d = ids.to(device)
                # (1) native eager forward -> attentions (fusion/top1) + hidden + logits
                with (torch.inference_mode(),
                      torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)):
                    o = model(input_ids=ids_d, attention_mask=torch.ones_like(ids_d),
                              output_attentions=True, output_hidden_states=True)
                attn = o.attentions
                head_maps = [_head_map(attn, sl, int(e), int(h)) for (e, h) in topk]
                fusion_map = fuse_naive_mean(np.stack(head_maps))
                top1_map = head_maps[0]
                h_nat = o.hidden_states[-1][:, sl, :][0].to(torch.float32)
                lg_nat = o.logits[:, sl, :][0][:, aa_ids].to(torch.float32)
                del o, attn
                # (2) perturbation sweep -> logit/repr/wtw
                raw_r = np.zeros((L, 20, L), dtype=np.float32)
                raw_w = np.zeros((L, 20, L), dtype=np.float32)
                raw_l = np.zeros((L, 20, L), dtype=np.float32)
                work = [(p, a) for p in range(L) for a in range(20)]
                for cs in range(0, len(work), max_perturbs_per_batch):
                    chunk = work[cs:cs + max_perturbs_per_batch]
                    batch = torch.stack([
                        adapter.tokenize(tok, seq[:p] + STANDARD_AA[a] + seq[p + 1:])[0][0]
                        for (p, a) in chunk
                    ], dim=0).to(device)
                    with (torch.inference_mode(),
                          torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)):
                        op = model(input_ids=batch, attention_mask=torch.ones_like(batch),
                                   output_hidden_states=True)
                    hp = op.hidden_states[-1][:, sl, :].to(torch.float32)
                    lp = op.logits[:, sl, :][..., aa_ids].to(torch.float32)
                    dh = hp - h_nat.unsqueeze(0)
                    dl = lp - lg_nat.unsqueeze(0)
                    rr = dh.norm(dim=-1).cpu().numpy()
                    ww = torch.einsum("ad,bld->bla", W_aa, dh).norm(dim=-1).cpu().numpy()
                    ll = dl.norm(dim=-1).cpu().numpy()
                    for j, (p, a) in enumerate(chunk):
                        raw_r[p, a] = rr[j]; raw_w[p, a] = ww[j]; raw_l[p, a] = ll[j]
                    del op, hp, lp, dh, dl
                maps = {
                    "fusion": fusion_map, "top1": top1_map,
                    "logit": _post(raw_l.mean(axis=1)),
                    "repr": _post(raw_r.mean(axis=1)),
                    "wtw": _post(raw_w.mean(axis=1)),
                }
                metrics = {}
                for m, M in maps.items():
                    c = _conf(M, contacts, valid, L)
                    if c is not None:
                        metrics[m] = c
                row = {"protein_id": pid, "sequence_length": L, "skipped": False,
                       "metrics": metrics}
                if pid in save_ids:
                    blobs = {}
                    for m, M in maps.items():
                        buf = io.BytesIO()
                        np.savez_compressed(buf, m=M.astype(np.float16))
                        blobs[m] = buf.getvalue()
                    cbuf = io.BytesIO()
                    np.savez_compressed(cbuf, c=contacts, v=valid)
                    row["map_blobs"] = blobs
                    row["contact_blob"] = cbuf.getvalue()
                out.append(row)
            except Exception as exc:
                torch.cuda.empty_cache()
                out.append({"protein_id": pid, "sequence_length": L, "skipped": True,
                            "reason": f"error: {exc!r}"})
        return out

    @_app.local_entrypoint()
    def dispatch(variant: str = "650M", dataset: str = "zhang_eval200",
                 n_proteins: int = 40, n_save: int = 6, k: int = 10,
                 max_perturbs_per_batch: int = 64):
        import io
        import json

        topk_lookup = json.loads((REPO_ROOT / "results" / "cl7_phase15_topk_heads.json").read_text())
        topk = topk_lookup[variant]["zhang_select10"][:k]
        proteins = _stratified(_proteins_for_dataset(dataset), n_proteins)
        # save maps for n_save proteins evenly spread across the (already
        # length-sorted) subsample.
        save_idx = np.linspace(0, len(proteins) - 1, n_save).round().astype(int)
        save_ids = [proteins[i].protein_id for i in sorted(set(save_idx.tolist()))]
        payload = []
        for ex in proteins:
            buf = io.BytesIO()
            np.savez_compressed(buf, contacts=ex.contact_map.astype(bool),
                                valid_residues=(ex.valid_residues.astype(bool)
                                                if ex.valid_residues is not None
                                                else np.ones(ex.length, dtype=bool)))
            payload.append((ex.protein_id, ex.sequence, buf.getvalue()))
        print(f"=== confidence {variant}/{dataset} N={len(payload)} save={save_ids} ===", flush=True)
        rows = confidence_batch.remote(variant, payload, list(topk), k, save_ids,
                                       max_perturbs_per_batch)
        # split maps out to .npz files; keep metrics in the JSON
        out_dir = REPO_ROOT / "results"
        maps_dir = out_dir / "r4_confmaps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        clean = []
        for r in rows:
            blobs = r.pop("map_blobs", None)
            cblob = r.pop("contact_blob", None)
            if blobs:
                for m, b in blobs.items():
                    (maps_dir / f"{r['protein_id']}_{m}.npz").write_bytes(b)
                if cblob:
                    (maps_dir / f"{r['protein_id']}_contacts.npz").write_bytes(cblob)
            clean.append(r)
        (out_dir / f"r4_confidence_{variant}_{dataset}.json").write_text(
            json.dumps({"variant": variant, "dataset": dataset, "k": k,
                        "save_ids": save_ids, "per_protein": clean}, indent=2))
        scored = [r for r in clean if not r.get("skipped")]
        print(f"wrote r4_confidence_{variant}_{dataset}.json (scored {len(scored)}/{len(clean)}); "
              f"maps for {len(save_ids)} in results/r4_confmaps/", flush=True)

except ImportError:
    _modal = None
    _app = None
    confidence_batch = None


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-proteins", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    proteins = _stratified(_proteins_for_dataset("zhang_eval200"), args.n_proteins)
    Ls = [ex.length for ex in proteins]
    # ~19L forwards/protein, eager; 650M anchor ~1.3 min at mean L.
    est = len(proteins) * 1.3 * (sum(Ls) / len(Ls) / 300.0) + 1.5
    print(f"N={len(proteins)} L=[{min(Ls)},{max(Ls)}] mean={sum(Ls)/len(Ls):.0f}; "
          f"~{est:.0f} min GPU = ${est/60*4.0:.2f} at $4/hr (run via: modal run {Path(__file__).name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
