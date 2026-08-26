"""Vectorised model-label permutation tests.

Mathematically identical to the naive loop in common.py -- Spearman's rho is
Pearson correlation on (tie-averaged) ranks, so once the alignment ranks are
fixed each permutation reduces to a dot product. That turns a per-draw
scipy.stats.spearmanr call into one batched matmul, which is ~100x faster and
lets the sweep use the same 200k draws without dominating the runtime.

common.py keeps the naive loops as _perm_p_pairs_naive and
_perm_p_models_naive; the two paths draw their permutations differently, so
they agree to Monte-Carlo error rather than bit-exactly.
"""
import numpy as np
from scipy.stats import rankdata

CHUNK = 20_000


def _unit_rank(x, axis=-1):
    """Rank, centre, and scale to unit norm so Pearson becomes a dot product."""
    r = rankdata(x, axis=axis).astype(np.float64)
    r = r - r.mean(axis=axis, keepdims=True)
    n = np.linalg.norm(r, axis=axis, keepdims=True)
    return r / np.maximum(n, 1e-300)


def perm_p_models(align, hs_vals, rho_obs, n_perm=200_000, seed=0):
    """Permute competence across models; alignment is one value per model."""
    rng = np.random.default_rng(seed)
    ra = _unit_rank(np.asarray(align, dtype=float))
    rc = _unit_rank(np.asarray(hs_vals, dtype=float))
    n = rc.shape[0]
    thr = abs(rho_obs) - 1e-12
    hits = 0
    done = 0
    while done < n_perm:
        k = min(CHUNK, n_perm - done)
        idx = np.argsort(rng.random((k, n)), axis=1)
        rho = rc[idx] @ ra                       # (k,)
        hits += int(np.count_nonzero(np.abs(rho) >= thr))
        done += k
    return (hits + 1) / (n_perm + 1)


def perm_p_pairs(align, hs, pair_names, rho_obs, n_perm=200_000, seed=0):
    """Permute competence across models, then rebuild the pair means."""
    rng = np.random.default_rng(seed)
    ra = _unit_rank(np.asarray(align, dtype=float))
    models = sorted(hs)
    vals = np.array([hs[m] for m in models], dtype=float)
    idx_of = {m: i for i, m in enumerate(models)}
    ia = np.array([idx_of[u] for u, _ in pair_names])
    ib = np.array([idx_of[v] for _, v in pair_names])
    n = len(models)
    thr = abs(rho_obs) - 1e-12
    hits = 0
    done = 0
    while done < n_perm:
        k = min(CHUNK, n_perm - done)
        pidx = np.argsort(rng.random((k, n)), axis=1)
        pv = vals[pidx]                          # (k, n_models)
        y = (pv[:, ia] + pv[:, ib]) / 2.0        # (k, n_pairs)
        ry = _unit_rank(y, axis=1)               # (k, n_pairs)
        rho = ry @ ra                            # (k,)
        hits += int(np.count_nonzero(np.abs(rho) >= thr))
        done += k
    return (hits + 1) / (n_perm + 1)
