"""Head-free averaged-Jacobian estimator.

This is the estimator xlc_phase1.py uses to refit half-corpus J-lenses for
the lens-fitting control (paper section 4.4, appendix `app:lensfit`); it is
the only remaining consumer. Correctness is gated by
tests/test_jvp_identity.py against explicit autograd Jacobians.

Identity: with g(H) = mean_{p'} suffix(H)_{p'} and a tangent that places the
SAME vector v at every position, JVP(g, v (x) 1_T) = (1/T) * sum_{p',p}
(d h_{p'} / d h_p) v. Hence for the causal position-pair-averaged Jacobian
J = (1/N) sum_{p<=p'}, we have J v = 2 JVP / (T+1), where N = T(T+1)/2 is the
number of reachable pairs (unreachable blocks are exactly zero, so the
constant-tangent sweep picks up only the causal average automatically)."""
import os

import torch


def estimate_J_item(suffix_fn, H_seq, chunk):
    """One item's d-basis JVP sweep -> (d, d) fp64 CPU Jacobian estimate.

    suffix_fn: (T, d) -> (T, d), applying blocks L..end (no head, no final
    pooling). H_seq: (T, d) cached activation entering the suffix."""
    T, d = H_seq.shape

    def g(H):
        return suffix_fn(H).mean(dim=0)          # (d,)

    def jvp_one(v_seq):
        return torch.func.jvp(g, (H_seq,), (v_seq,))[1]

    eye = torch.eye(d, device=H_seq.device, dtype=H_seq.dtype)
    rows = []
    for s in range(0, d, chunk):
        V = eye[s:s + chunk]                     # (c, d) basis vectors
        tangents = V.unsqueeze(1).expand(-1, T, -1)  # same v at every position
        # detach: if any module param has requires_grad, the JVP forward would
        # otherwise retain a reverse-mode graph across the whole basis sweep
        rows.append(torch.vmap(jvp_one)(tangents).detach())
    Jhat = torch.cat(rows, 0).T                  # column j = JVP for e_j
    assert torch.isfinite(Jhat).all(), "non-finite JVP output"
    return (Jhat * (2.0 / (T + 1))).double().cpu()


def load_mean_J(path):
    """Mean of a JAccumulator checkpoint -> (d, d) fp32."""
    ck = torch.load(path, weights_only=False)
    return (ck["sum"] / ck["count"]).float()


class JAccumulator:
    """Running-mean fp64 CPU accumulator with resume. One instance per
    (model, band layer, fitting-corpus half) in xlc_phase1.py."""

    def __init__(self, d, path):
        self.path = path
        self.sum = torch.zeros(d, d, dtype=torch.float64)
        self.count = 0
        if os.path.exists(path):
            ckpt = torch.load(path, weights_only=False)
            self.sum, self.count = ckpt["sum"], ckpt["count"]
            assert self.sum.shape == (d, d), f"checkpoint shape mismatch at {path}"

    def add(self, J):
        assert J.dtype == torch.float64 and torch.isfinite(J).all()
        self.sum += J
        self.count += 1
        if self.count % 32 == 0:
            self.save()

    def save(self):
        tmp = self.path + ".tmp"
        torch.save({"sum": self.sum, "count": self.count}, tmp)
        os.replace(tmp, self.path)

    @property
    def mean(self):
        assert self.count > 0, "empty accumulator"
        return self.sum / self.count
