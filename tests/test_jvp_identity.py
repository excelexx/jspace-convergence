"""Correctness gate for crossmodal_jacobian.estimate_J_item, the averaged-
Jacobian estimator xlc_phase1.py uses to refit half-corpus lenses for the
lens-fitting control (appendix `app:lensfit`).

The constant-tangent JVP identity must agree with an explicit autograd
Jacobian summed over reachable position pairs, on a toy 2-layer causal
transformer, to 1e-5. Lens refits are blocked until this passes.
Run from the project root: python tests/test_jvp_identity.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crossmodal_jacobian import estimate_J_item

torch.manual_seed(0)
T, D = 4, 8
DTYPE = torch.float64
TOL = 1e-5


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(D, dtype=DTYPE)
        self.ln2 = torch.nn.LayerNorm(D, dtype=DTYPE)
        self.q = torch.nn.Linear(D, D, dtype=DTYPE)
        self.k = torch.nn.Linear(D, D, dtype=DTYPE)
        self.v = torch.nn.Linear(D, D, dtype=DTYPE)
        self.o = torch.nn.Linear(D, D, dtype=DTYPE)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(D, 4 * D, dtype=DTYPE), torch.nn.GELU(),
            torch.nn.Linear(4 * D, D, dtype=DTYPE))

    def forward(self, x):                        # (T, D)
        h = self.ln1(x)
        att = (self.q(h) @ self.k(h).T) / math.sqrt(D)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        x = x + self.o(att.softmax(dim=-1) @ self.v(h))
        return x + self.mlp(self.ln2(x))


def run_case():
    torch.manual_seed(1)
    blocks = torch.nn.Sequential(ToyBlock(), ToyBlock())
    for p in blocks.parameters():
        p.requires_grad_(False)
    H0 = torch.randn(T, D, dtype=DTYPE)

    # explicit reference: full (T,D,T,D) Jacobian, summed over position pairs
    jac = torch.autograd.functional.jacobian(blocks, H0)   # [p', i, p, j]
    for p_out in range(T):                       # unreachable blocks must be 0
        for p_in in range(p_out + 1, T):
            zmax = jac[p_out, :, p_in, :].abs().max().item()
            assert zmax < 1e-12, f"causal leak at ({p_out},{p_in}): {zmax}"
    n_pairs = T * (T + 1) // 2
    J_ref = jac.sum(dim=(0, 2)) / n_pairs

    # estimator under test, chunk=3 to force the chunked-vmap path
    J_est = estimate_J_item(blocks, H0, chunk=3)

    err = (J_est - J_ref).abs().max().item()
    rel = ((J_est - J_ref).norm() / J_ref.norm()).item()
    print(f"  causal: max abs err {err:.2e}, rel fro {rel:.2e}, "
          f"n_pairs {n_pairs}, ||J_ref||_F {J_ref.norm():.4f}")
    assert err < TOL, f"JVP identity disagrees with explicit Jacobian: {err}"


if __name__ == "__main__":
    print("JVP-identity correctness gate (toy 2-layer causal transformer, "
          f"T={T}, d={D}, fp64):")
    run_case()
    print("PASS - lens refits are unblocked")
