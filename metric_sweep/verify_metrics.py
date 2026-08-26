"""Verify the metric ports against reference implementations before the sweep.

Checks:
  1. batched LCS  == reference O(mn) Python DP, per item
  2. batched edit == reference Levenshtein DP, per item
  3. cka_from_gram(K,L) == reference feature-space CKA (biased and unbiased)
  4. cknna_from_gram   == reference feature-space CKNNA
  5. mutual_knn / cycle_knn == reference implementations
  6. sanity: every metric is 1.0 (or maximal) for identical inputs
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import common as C


# ---- reference implementations (transcribed from platonic-rep/metrics.py) --
def ref_lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def ref_edit_distance(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n]


def ref_cka(fa, fb, unbiased=False):
    K = fa @ fa.T
    L = fb @ fb.T
    f = C.hsic_unbiased if unbiased else C.hsic_biased
    return (f(K, L) / (torch.sqrt(f(K, K) * f(L, L)) + 1e-6)).item()


def ref_cknna(fa, fb, topk=10, unbiased=True):
    n = fa.shape[0]
    K = fa @ fa.T
    L = fb @ fb.T

    def similarity(A, B, topk):
        if unbiased:
            Ah = A.clone().fill_diagonal_(float("-inf"))
            Bh = B.clone().fill_diagonal_(float("-inf"))
        else:
            Ah, Bh = A, B
        _, ia = torch.topk(Ah, topk, dim=1)
        _, ib = torch.topk(Bh, topk, dim=1)
        ma = torch.zeros(n, n, device=A.device, dtype=A.dtype).scatter_(1, ia, 1)
        mb = torch.zeros(n, n, device=A.device, dtype=A.dtype).scatter_(1, ib, 1)
        m = ma * mb
        if unbiased:
            return C.hsic_unbiased(m * A, m * B)
        return C.hsic_biased(m * A, m * B)

    return (similarity(K, L, topk)
            / (torch.sqrt(similarity(K, K, topk) * similarity(L, L, topk))
               + 1e-6)).item()


def ref_compute_nn(feats, topk):
    return (feats @ feats.T).fill_diagonal_(-1e8).argsort(
        dim=1, descending=True)[:, :topk]


def ref_mutual_knn(fa, fb, topk):
    ka, kb = ref_compute_nn(fa, topk), ref_compute_nn(fb, topk)
    n = ka.shape[0]
    r = torch.arange(n, device=ka.device).unsqueeze(1)
    ma = torch.zeros(n, n, device=ka.device)
    mb = torch.zeros(n, n, device=ka.device)
    ma[r, ka] = 1.0
    mb[r, kb] = 1.0
    return ((ma * mb).sum(dim=1) / topk).mean().item()


def ref_cycle_knn(fa, fb, topk):
    ka, kb = ref_compute_nn(fa, topk), ref_compute_nn(fb, topk)
    n = ka.shape[0]
    acc = ka[kb] == torch.arange(n, device=ka.device).view(-1, 1, 1)
    return acc.float().view(n, -1).max(dim=1).values.mean().item()


def main():
    torch.manual_seed(0)
    dev = C.DEV
    n, d, k = 200, 64, 10
    fa = torch.randn(n, d, device=dev, dtype=torch.float64)
    fb = torch.randn(n, d, device=dev, dtype=torch.float64)
    fa = fa / fa.norm(dim=1, keepdim=True)
    fb = fb / fb.norm(dim=1, keepdim=True)
    Ka, Kb = fa @ fa.T, fb @ fb.T
    ka, kb = C.knn_from_gram(Ka, k), C.knn_from_gram(Kb, k)

    ok = True

    # 1 + 2: sequence kernels vs reference DP
    A, B = ka.cpu().numpy(), kb.cpu().numpy()
    ref_l = np.mean([ref_lcs_length(A[i], B[i]) for i in range(n)])
    got_l = C.lcs_knn(ka, kb)
    ref_e = 1 - np.mean([ref_edit_distance(A[i], B[i]) for i in range(n)]) / k
    got_e = C.edit_knn(ka, kb)
    for label, r, g, tol in [("lcs_knn", ref_l, got_l, 1e-9),
                             ("edit_knn", ref_e, got_e, 1e-9)]:
        good = abs(r - g) < tol
        ok &= good
        print(f"{'OK ' if good else 'FAIL'} {label:12s} ref {r:.10f}  got {g:.10f}")

    # 3: CKA from Gram vs reference from features
    for ub in (False, True):
        r = ref_cka(fa, fb, ub)
        g = C.cka_from_gram(Ka, Kb, ub)
        good = abs(r - g) < 1e-9
        ok &= good
        label = "unbiased_cka" if ub else "cka"
        print(f"{'OK ' if good else 'FAIL'} {label:12s} ref {r:.10f}  got {g:.10f}")

    # 4: CKNNA
    r, g = ref_cknna(fa, fb, k), C.cknna_from_gram(Ka, Kb, k)
    good = abs(r - g) < 1e-9
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {'cknna':12s} ref {r:.10f}  got {g:.10f}")

    # 5: knn metrics
    for label, r, g in [("mutual_knn", ref_mutual_knn(fa.clone(), fb.clone(), k),
                         C.mutual_knn(ka, kb)),
                        ("cycle_knn", ref_cycle_knn(fa.clone(), fb.clone(), k),
                         C.cycle_knn(ka, kb))]:
        good = abs(r - g) < 1e-9
        ok &= good
        print(f"{'OK ' if good else 'FAIL'} {label:12s} ref {r:.10f}  got {g:.10f}")

    # 6: identical-input maxima
    # the reference divides by (sqrt(.) + 1e-6); with small HSIC values that
    # epsilon pulls self-similarity a hair under 1, so allow 1e-3 there
    ident = [("cka", C.cka_from_gram(Ka, Ka), 1.0, 1e-3),
             ("unbiased_cka", C.cka_from_gram(Ka, Ka, True), 1.0, 1e-3),
             ("cknna", C.cknna_from_gram(Ka, Ka, k), 1.0, 1e-3),
             ("mutual_knn", C.mutual_knn(ka, ka), 1.0, 1e-9),
             ("lcs_knn", C.lcs_knn(ka, ka), float(k), 1e-9),
             ("edit_knn", C.edit_knn(ka, ka), 1.0, 1e-9)]
    for label, got, want, tol in ident:
        good = abs(got - want) < tol
        ok &= good
        print(f"{'OK ' if good else 'FAIL'} {label:12s} self {got:.8f} "
              f"(want {want})")

    # svcca sanity: identical features -> ~1
    U = C.svcca_basis(fa.float())
    s = C.svcca_from_basis(U.cpu().numpy(), U.cpu().numpy())
    good = s > 0.99
    ok &= good
    print(f"{'OK ' if good else 'FAIL'} {'svcca':12s} self {s:.8f} (want ~1)")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
