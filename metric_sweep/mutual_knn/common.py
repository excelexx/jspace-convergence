"""Shared machinery for the metric sweep: paper-faithful decomposition plus
alignment metrics ported from minyoungg/platonic-rep (metrics.py).

Everything upstream of the metric is held identical to the paper:
  text-text     : ../acts_{model}.pt, band layers, 0.95-quantile clamp,
                  k=25 non-negative OMP + NNLS against the vocab-mean-centred
                  W_U.diag(w).J_L dictionary, then prep() = clamp + l2 norm
  text-vision   : ../cache/text_acts/{model}_L{L}_pool.npy captions decomposed
                  the same way (caption side), against ../cache/vision/
                  {enc}/eval_acts_L{L}.npy raw pooled patches preprocessed with
                  the per-coordinate 0.5/99.5 clip + l2 norm of ../xkernels.py
  aggregation   : mean over the band x band layer-pair grid, per model pair
Only the metric that scores a layer pair changes.

Metric ports are faithful to the reference implementation; the batched LCS and
edit-distance kernels are verified against reference Python versions by
verify_metrics.py.
"""
import json
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_SPARSE, TOPK, CCA_DIM = 25, 10, 10
V_CHUNK = 65536


def _find_up(start, marker, isdir=True):
    """Walk upward from `start` until `marker` exists, so a metric subfolder
    and the sweep root both resolve the same paths."""
    d = os.path.abspath(start)
    for _ in range(6):
        p = os.path.join(d, marker)
        if (os.path.isdir(p) if isdir else os.path.exists(p)):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise RuntimeError("could not locate " + marker + " above " + start)


_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _find_up(_HERE, os.path.join("results", "lmeval"))
FEATURES = os.path.join(_find_up(_HERE, "_features"), "_features")

MODELS = {
    "gpt2": dict(acts="acts_gpt2.pt", hf="gpt2",
        lens="lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
    "gemma": dict(acts="acts_gemma.pt", hf="google/gemma-3-1b-pt",
        lens="lenses/gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt"),
    "gemma270": dict(acts="acts_gemma270.pt", hf="google/gemma-3-270m",
        lens="lenses/gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt"),
    "pythia70m": dict(acts="acts_pythia70m.pt", hf="EleutherAI/pythia-70m-deduped",
        lens="lenses/pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"),
    "qwen08b": dict(acts="acts_qwen08b.pt", hf="Qwen/Qwen3.5-0.8B",
        lens="lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"),
    "qwen17b": dict(acts="acts_qwen17b.pt", hf="Qwen/Qwen3-1.7B",
        lens="lenses/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"),
    "qwen2b": dict(acts="acts_qwen2b.pt", hf="Qwen/Qwen3.5-2B-Base",
        lens="lenses/qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt"),
    "gemma2_2b": dict(acts="acts_gemma2_2b.pt", hf="google/gemma-2-2b",
        lens="lenses/gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt"),
    "qwen4b": dict(acts="acts_qwen4b.pt", hf="Qwen/Qwen3-4B",
        lens="lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"),
    "qwen35_4b": dict(acts="acts_qwen35_4b.pt", hf="Qwen/Qwen3.5-4B",
        lens="lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"),
    "gemma3_4b": dict(acts="acts_gemma3_4b.pt", hf="google/gemma-3-4b-pt",
        lens="lenses/gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt"),
}
ENCODERS = ["dinov2", "mae", "clip", "siglip"]
COMPS = ["full", "J", "perp"]
NORM_PATHS = ["transformer.ln_f", "gpt_neox.final_layer_norm", "model.norm",
              "model.language_model.norm", "language_model.model.norm"]


# ---------------- paper-faithful decomposition (from ../step7_align.py) -----
def get_WU_and_w(hf):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(hf, dtype="auto")
    W = m.get_output_embeddings().weight.detach().float().clone()
    norm = None
    for attr in NORM_PATHS:
        obj = m
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            norm = obj
            break
        except AttributeError:
            continue
    assert norm is not None, "no final norm for " + hf
    w = norm.weight.detach().float().clone()
    if "gemma" in type(norm).__name__.lower():
        w = 1.0 + w
    del m
    return W, w


def nnls_refit(A, h, iters=80):
    At = A.transpose(1, 2)
    G = At @ A
    b = At @ h
    step = 1.0 / G.diagonal(dim1=1, dim2=2).sum(-1).clamp(min=1e-8)
    c = torch.zeros_like(b)
    for _ in range(iters):
        c = torch.clamp(c - step.view(-1, 1, 1) * (G @ c - b), min=0)
    return c


def best_atom(r, D, forbid):
    n = r.shape[0]
    best_v = torch.full((n,), -1e30, device=DEV)
    best_i = torch.zeros(n, dtype=torch.long, device=DEV)
    for start in range(0, D.shape[0], V_CHUNK):
        S = r @ D[start:start + V_CHUNK].T
        if forbid is not None:
            inchunk = (forbid >= start) & (forbid < start + S.shape[1])
            rows, cols = inchunk.nonzero(as_tuple=True)
            S[rows, forbid[rows, cols] - start] = -1e30
        v, i = S.max(dim=1)
        upd = v > best_v
        best_v = torch.where(upd, v, best_v)
        best_i = torch.where(upd, i + start, best_i)
    return best_i


def nnomp_batch(H, D):
    r = H.clone()
    sel = torch.zeros(H.shape[0], K_SPARSE, dtype=torch.long, device=DEV)
    for s in range(K_SPARSE):
        sel[:, s] = best_atom(r, D, sel[:, :s] if s else None)
        A = D[sel[:, :s + 1]].transpose(1, 2)
        c = nnls_refit(A, H.unsqueeze(-1))
        r = H - (A @ c).squeeze(-1)
    return H - r


def prep(X):
    """Paper text/caption-side preprocessing: 0.95-quantile clamp + l2 norm."""
    q = torch.quantile(X.abs().flatten().float(), 0.95)
    X = X.clamp(-q, q)
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)


def prep_image(X):
    """Paper image-side preprocessing (../xkernels.py preprocess())."""
    lo = torch.quantile(X, 0.005, dim=0, keepdim=True)
    hi = torch.quantile(X, 0.995, dim=0, keepdim=True)
    Xc = torch.clamp(X, min=lo, max=hi)
    return Xc / Xc.norm(dim=1, keepdim=True).clamp(min=1e-8)


def unembed_eff(hf):
    WU, w = get_WU_and_w(hf)
    WUeff = WU.to(DEV) * w.to(DEV)
    del WU
    WUeff -= WUeff.mean(dim=0, keepdim=True)      # vocab-mean centring
    return WUeff


def dictionary(lens, L, WUeff):
    J = lens["J"][L].to(DEV, torch.float32)
    D = WUeff @ J
    return D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)


def decompose(H, D):
    """(full, J, perp) prepped components for one clamped activation block."""
    HJ = nnomp_batch(H, D)
    return {"full": prep(H), "J": prep(HJ), "perp": prep(H - HJ)}


# ---------------- metrics (ported from platonic-rep/metrics.py) -------------
def hsic_biased(K, L):
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def hsic_unbiased(K, L):
    m = K.shape[0]
    Kt = K.clone().fill_diagonal_(0)
    Lt = L.clone().fill_diagonal_(0)
    v = ((Kt * Lt.T).sum()
         + Kt.sum() * Lt.sum() / ((m - 1) * (m - 2))
         - 2 * (Kt @ Lt).sum() / (m - 2))
    return v / (m * (m - 3))


def cka_from_gram(K, L, unbiased=False):
    f = hsic_unbiased if unbiased else hsic_biased
    return (f(K, L) / (torch.sqrt(f(K, K) * f(L, L)) + 1e-6)).item()


def cknna_from_gram(K, L, topk=TOPK, unbiased=True):
    n = K.shape[0]

    def sim(A, B):
        if unbiased:
            Ah = A.clone().fill_diagonal_(float("-inf"))
            Bh = B.clone().fill_diagonal_(float("-inf"))
        else:
            Ah, Bh = A, B
        ia = torch.topk(Ah, topk, dim=1).indices
        ib = torch.topk(Bh, topk, dim=1).indices
        ma = torch.zeros(n, n, device=A.device, dtype=A.dtype).scatter_(1, ia, 1)
        mb = torch.zeros(n, n, device=A.device, dtype=A.dtype).scatter_(1, ib, 1)
        m = ma * mb
        if unbiased:
            return hsic_unbiased(m * A, m * B)
        return hsic_biased(m * A, m * B)

    return (sim(K, L) / (torch.sqrt(sim(K, K) * sim(L, L)) + 1e-6)).item()


def knn_from_gram(K, topk=TOPK):
    """Reference convention: diagonal -1e8, argsort descending, take topk."""
    A = K.clone().fill_diagonal_(-1e8)
    return A.argsort(dim=1, descending=True)[:, :topk]


def mutual_knn(knn_A, knn_B):
    n, topk = knn_A.shape
    r = torch.arange(n, device=knn_A.device).unsqueeze(1)
    ma = torch.zeros(n, n, device=knn_A.device)
    mb = torch.zeros(n, n, device=knn_A.device)
    ma[r, knn_A] = 1.0
    mb[r, knn_B] = 1.0
    return ((ma * mb).sum(dim=1) / topk).mean().item()


def cycle_knn(knn_A, knn_B):
    n = knn_A.shape[0]
    stacked = knn_A[knn_B]                        # (n, topk, topk)
    acc = stacked == torch.arange(n, device=knn_A.device).view(-1, 1, 1)
    return acc.float().view(n, -1).max(dim=1).values.mean().item()


def lcs_knn(knn_A, knn_B):
    """Mean longest-common-subsequence length, batched over items."""
    X, Y = knn_A, knn_B
    n, k = X.shape
    dp = torch.zeros(n, k + 1, k + 1, device=X.device, dtype=torch.float64)
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            eq = X[:, i - 1] == Y[:, j - 1]
            dp[:, i, j] = torch.where(
                eq, dp[:, i - 1, j - 1] + 1,
                torch.maximum(dp[:, i - 1, j], dp[:, i, j - 1]))
    return dp[:, k, k].mean().item()


def edit_knn(knn_A, knn_B):
    """1 - mean Levenshtein distance / topk, batched over items."""
    X, Y = knn_A, knn_B
    n, k = X.shape
    dp = torch.zeros(n, k + 1, k + 1, device=X.device, dtype=torch.float64)
    ar = torch.arange(k + 1, device=X.device, dtype=torch.float64)
    dp[:, :, 0] = ar.unsqueeze(0)
    dp[:, 0, :] = ar.unsqueeze(0)
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            cost = (X[:, i - 1] != Y[:, j - 1]).double()
            dp[:, i, j] = torch.minimum(
                torch.minimum(dp[:, i - 1, j] + 1, dp[:, i, j - 1] + 1),
                dp[:, i - 1, j - 1] + cost)
    return (1 - dp[:, k, k].mean() / k).item()


def svcca_basis(feats, cca_dim=CCA_DIM):
    """Centre/scale then top-cca_dim left singular vectors (reference order)."""
    a = feats - feats.mean(axis=0)
    a = a / (a.std(axis=0) + 1e-8)
    U, _, _ = torch.svd_lowrank(a, q=cca_dim)
    return U


def svcca_from_basis(U1, U2, cca_dim=CCA_DIM, seed=0):
    from sklearn.cross_decomposition import CCA
    rng = np.random.default_rng(seed)
    cca = CCA(n_components=cca_dim)
    cca.fit(U1, U2)
    a, b = cca.transform(U1, U2)
    a = a + 1e-10 * rng.standard_normal(a.shape)
    b = b + 1e-10 * rng.standard_normal(b.shape)
    return float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1]
                          for i in range(cca_dim)]))


# ---------------- stats -----------------------------------------------------
def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def perm_p_pairs(align, hs, pair_names, rho_obs, n_perm=200_000, seed=0):
    """Model-label permutation for pair-level stats (the paper's null).

    Delegates to the vectorised implementation; _perm_p_pairs_naive below is
    the reference loop it reproduces."""
    import fastperm
    return fastperm.perm_p_pairs(align, hs, pair_names, rho_obs, n_perm, seed)


def _perm_p_pairs_naive(align, hs, pair_names, rho_obs, n_perm=200_000, seed=0):
    """Reference implementation, one scipy spearmanr per draw."""
    rng = np.random.default_rng(seed)
    models = sorted(hs)
    vals = np.array([hs[m] for m in models])
    idx = {m: i for i, m in enumerate(models)}
    ia = np.array([idx[u] for u, _ in pair_names])
    ib = np.array([idx[v] for _, v in pair_names])
    a = np.asarray(align)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(vals)
        y = (p[ia] + p[ib]) / 2
        if abs(spearman(y, a)) >= abs(rho_obs) - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def perm_p_models(align, hs_vals, rho_obs, n_perm=200_000, seed=0):
    """Model-level permutation for the 11-point cross-modal stats.

    Delegates to the vectorised implementation; _perm_p_models_naive below is
    the reference loop it reproduces."""
    import fastperm
    return fastperm.perm_p_models(align, hs_vals, rho_obs, n_perm, seed)


def _perm_p_models_naive(align, hs_vals, rho_obs, n_perm=200_000, seed=0):
    """Reference implementation, one scipy spearmanr per draw."""
    rng = np.random.default_rng(seed)
    v = np.asarray(hs_vals)
    a = np.asarray(align)
    hits = 0
    for _ in range(n_perm):
        if abs(spearman(rng.permutation(v), a)) >= abs(rho_obs) - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, ic = np.polyfit(x, y, 1)
    resid = y - (slope * x + ic)
    se = float((resid @ resid / (len(y) - 2)) ** 0.5
               / ((x - x.mean()) ** 2).sum() ** 0.5)
    return float(slope), se


def hellaswag(names):
    out = {}
    for n in names:
        with open(os.path.join(ROOT, "results/lmeval/" + n + ".json")) as f:
            out[n] = json.load(f)["hellaswag_acc_norm"]
    return out


def text_pairs():
    ns = list(MODELS)
    return [(a, b) for i, a in enumerate(ns) for b in ns[i + 1:]]


def band_layers():
    with open(os.path.join(ROOT, "cache/text_acts/layers.json")) as f:
        return json.load(f)


def vision_layers():
    with open(os.path.join(ROOT, "cache/vision/layers.json")) as f:
        return json.load(f)
