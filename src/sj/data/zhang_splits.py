"""Canonical Zhang-1431 select/eval split ( rerun protocol).

The headline in-distribution protocol: SELECT top-K attention heads on a small
labeled set, EVALUATE on a disjoint, length-representative 200-protein draw from
the full Zhang set. This module is the single source of truth for that split so
every dispatcher (logit-CJ, repr-CJ, fusion; every architecture) scores the SAME
200 proteins — required for paired cross-method/cross-architecture comparisons.

Design:
  - Length-stratified, not uniform-random: we sort the 1430 chains by length and
    draw one protein per equal-rank stratum, so the eval set's length
    distribution matches the full set's by construction (no length bias).
  - Disjoint by construction: the selection set is drawn from the complement of
    the eval set (assertion enforced).
  - Deterministic: a fixed seed makes the split reproducible across runs and
    machines; every dispatcher recomputes the identical split.

N_EVAL=200 gives a bootstrap-of-mean CI half-width of ~0.010 on P@L/2 long
(measured per-protein sigma ~0.074), about half the N=50 width, at ~1/7 the cost
of the full set. N_SELECT=10 is the label budget (Appendix O: ~10 proteins
suffice for head selection).
"""

from __future__ import annotations

import itertools

import numpy as np

DEFAULT_SEED = 4231 # distinct from the legacy seed-42/43 Zhang-50 draws
N_EVAL = 200
N_SELECT = 10


def _stratified_pick(items, k, rng):
    """One item per equal-rank stratum over a length-sorted list (representative)."""
    n = len(items)
    if k >= n:
        return list(items)
    # k contiguous strata over the rank axis; pick a random index within each.
    edges = np.linspace(0, n, k + 1).astype(int)
    out = []
    for lo, hi in itertools.pairwise(edges):
        hi = max(hi, lo + 1)
        out.append(items[int(rng.integers(lo, hi))])
    return out


LEGACY_SEED = 42 # the original Zhang-50 draw (results already computed for it)
LEGACY_N = 50


def legacy_zhang50_ids():
    """Reproduce the original seed-42 Zhang-50 protein IDs exactly (loader order,
    np default_rng(42).choice(N, 50)). These are the proteins whose logit-CJ /
    repr-CJ results are already cached and can be reused in the eval-200 set."""
    from sj.data.zhang_1431 import Zhang1431Loader, default_zhang_1431_root

    proteins = list(Zhang1431Loader(data_root=default_zhang_1431_root())) # native order
    idx = np.random.default_rng(LEGACY_SEED).choice(len(proteins), size=LEGACY_N, replace=False)
    return {proteins[i].protein_id for i in sorted(idx)}


def zhang_select_eval_split(n_eval=N_EVAL, n_select=N_SELECT, seed=DEFAULT_SEED, reuse_legacy=True):
    """Return (select_proteins, eval_proteins): disjoint, length-representative.

    eval (n_eval): when reuse_legacy, the original seed-42 Zhang-50 are FORCED in
    (so their cached CJ/repr-CJ results can be reused) and the remaining
    n_eval-50 are drawn length-stratified from the rest so the *union* still spans
    the full length distribution. select (n_select): length-stratified from the
    disjoint complement (the labeled head-selection budget).
    """
    from sj.data.zhang_1431 import Zhang1431Loader, default_zhang_1431_root

    proteins = list(Zhang1431Loader(data_root=default_zhang_1431_root()))
    proteins.sort(key=lambda p: (p.length, p.protein_id)) # length-sorted for stratification
    rng = np.random.default_rng(seed)

    if reuse_legacy:
        legacy = legacy_zhang50_ids()
        forced = [p for p in proteins if p.protein_id in legacy]
        pool = [p for p in proteins if p.protein_id not in legacy]
        fill = _stratified_pick(pool, n_eval - len(forced), rng)
        eval_set = sorted(forced + fill, key=lambda p: (p.length, p.protein_id))
    else:
        eval_set = _stratified_pick(proteins, n_eval, rng)

    eval_ids = {p.protein_id for p in eval_set}
    remainder = [p for p in proteins if p.protein_id not in eval_ids]
    select_set = _stratified_pick(remainder, n_select, rng)

    sel_ids = {p.protein_id for p in select_set}
    assert not (eval_ids & sel_ids), "select/eval sets overlap"
    assert len(eval_ids) == len(eval_set) == n_eval, "wrong eval count / duplicates"
    assert len(sel_ids) == len(select_set) == n_select, "wrong select count / duplicates"
    return select_set, eval_set


EVAL_SLUG = "zhang_eval200" # honest result-slug for the N=200 eval set
SELECT_SLUG = "zhang_select10" # the disjoint head-selection set

# Module-level cache so every dispatcher in a process sees the identical split.
_SPLIT = None


def _split():
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = zhang_select_eval_split()
    return _SPLIT


def eval_proteins():
    """The canonical 200-protein evaluation set (contains the legacy-50)."""
    return _split()[1]


def select_proteins():
    """The canonical 10-protein head-selection set (disjoint from eval)."""
    return _split()[0]


def proteins_for_slug(slug):
    """Dispatch slug -> protein list. New slugs route through the canonical split;
    legacy slugs are left to each script's own loader."""
    if slug == EVAL_SLUG:
        return eval_proteins()
    if slug == SELECT_SLUG:
        return select_proteins()
    raise ValueError(f"proteins_for_slug: unknown slug {slug!r}")


def legacy_reuse_map(old_json_path, id_key="protein_id"):
    """Map {protein_id: per_protein_entry} for the legacy-50 found in an existing
    *_zhang_random_50.json, to pre-seed reuse of selection-INDEPENDENT cells
    (logit-CJ, repr-CJ). Returns {} if the file is absent. Caller asserts the
    entries are among the eval-200 (they are, by construction)."""
    import json
    from pathlib import Path

    p = Path(old_json_path)
    if not p.is_file():
        return {}
    d = json.loads(p.read_text())
    rows = d.get("per_protein", d if isinstance(d, list) else [])
    legacy = legacy_zhang50_ids()
    out = {}
    for r in rows:
        pid = r.get(id_key)
        if pid in legacy and not r.get("skipped"):
            out[pid] = r
    return out


def load_existing_per_protein(out_json_path):
    """Read the ``per_protein`` rows from an existing output JSON for
    idempotency (resume-on-crash). Returns [] if the file is absent. Accepts
    either ``{"per_protein": [...]}`` or a bare list at the top level."""
    import json
    from pathlib import Path

    p = Path(out_json_path)
    if not p.is_file():
        return []
    d = json.loads(p.read_text())
    if isinstance(d, list):
        return d
    return d.get("per_protein", [])


def carry_forward_existing(rows, target_ids, id_key="protein_id"):
    """Build the resume done-set for resume idempotency.

    Restrict carried-forward rows to (a) protein_ids that ARE in ``target_ids``
    (drop stale rows from a different split), and (b) rows that are NOT
    ``skipped`` (so skipped proteins retry rather than counting as done).
    Returns {protein_id: row} keyed by the last-seen non-skipped row in target.
    """
    target = set(target_ids)
    out: dict = {}
    for r in rows:
        pid = r.get(id_key)
        if pid is None or pid not in target:
            continue
        if r.get("skipped"):
            continue
        out[pid] = r
    return out


def assert_complete_split(per_protein, target_ids, slug, id_key="protein_id"):
    """After a run, assert per-protein ids EXACTLY cover the target set (FIX 1).

    Only enforced for the canonical zhang_eval200 / zhang_select10 slugs; for
    other slugs this is a no-op (legacy behavior unchanged). A skipped protein
    still emits a row carrying its protein_id, so the coverage check counts
    every row with a protein_id (skipped or not) — what it guards against is a
    missing/extra protein, not a skip.
    """
    if slug not in (EVAL_SLUG, SELECT_SLUG):
        return
    target = set(target_ids)
    got = {r.get(id_key) for r in per_protein if r.get(id_key) is not None}
    if got != target:
        missing = sorted(target - got)
        extra = sorted(got - target)
        raise RuntimeError(
            f"{slug}: per-protein set does not match target "
            f"(n_got={len(got)}, n_target={len(target)}; "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}; "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})"
        )
    if len(got) != len(target):
        raise RuntimeError(f"{slug}: count mismatch n_got={len(got)} != n_target={len(target)}")


def length_summary(proteins):
    """(n, mean, p10, median, p90, max) of sequence lengths — for representativeness checks."""
    L = np.array([p.length for p in proteins], dtype=float)
    return (
        len(L),
        float(L.mean()),
        float(np.percentile(L, 10)),
        float(np.median(L)),
        float(np.percentile(L, 90)),
        float(L.max()),
    )
