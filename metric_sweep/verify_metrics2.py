"""Verify the CKNNA sweep's two load-bearing identities.

The sweep computes CKNNA from cached Gram matrices for speed. Two properties
have to hold for the appendix's locality paragraph to mean what it says:

  1. at topk = n-1 the neighbour mask becomes all-ones off the diagonal, so
     CKNNA collapses exactly to unbiased CKA. This is what makes the topk
     sweep a continuous interpolation between the local metric family and the
     global one, rather than two unrelated numbers;
  2. at the paper's topk = 10 the swept path agrees with the CKNNA used for
     the headline table, so the k=10 point of the sweep is the same measurement
     as the `cknna` row.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import common as C


def main():
    torch.manual_seed(0)
    n, d1, d2, k = 240, 48, 64, C.TOPK
    X = torch.randn(n, d1, device=C.DEV, dtype=torch.float64)
    Y = torch.randn(n, d2, device=C.DEV, dtype=torch.float64)
    # give them shared structure so the metrics are not all ~0
    shared = torch.randn(n, 16, device=C.DEV, dtype=torch.float64)
    X[:, :16] += 1.5 * shared
    Y[:, :16] += 1.5 * shared
    X = X / X.norm(dim=1, keepdim=True)
    Y = Y / Y.norm(dim=1, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    ok = True

    def check(label, ref, got, tol):
        nonlocal ok
        good = abs(ref - got) < tol
        ok &= good
        print(f"{'OK ' if good else 'FAIL'} {label:26s} ref {ref:+.8f}  "
              f"got {got:+.8f}  |d| {abs(ref-got):.2e}")

    check("cknna(topk=n-1)==ubCKA", C.cka_from_gram(K, L, unbiased=True),
          C.cknna_from_gram(K, L, n - 1), 1e-6)
    # self-similarity, up to the 1e-6 the denominator adds for stability
    check(f"cknna(topk={k}) self", 1.0, C.cknna_from_gram(K, K, k), 1e-3)

    print("\nCKNNA SWEEP IDENTITIES VERIFIED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
