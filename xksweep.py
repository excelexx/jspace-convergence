"""Lens-degradation control: sparsity sweep on real text (appendix C.2).

Builds the instrument-only null. Varying k on the REAL corpus produces
deliberately degraded J codes with content fully intact, so alignment can be
read off at the k whose variance share matches what the ablated corpus produces
at k = 25, and at k = 5, both of which the appendix quotes.

Orthogonal matching pursuit is greedy and nested: after round i the running
reconstruction is the k = i+1 solution, so every k is recorded in one pass at
the cost of a single k = 25 decomposition.

Output is cache/ksweep/{model}.pt, the per-k neighbour lists and variance
shares. xksweep_mean.py turns the cache into the numbers the paper reports.
"""
import os

import torch

from xsurrogate import best_atom, get_WU_and_w, neighbors, nnls_refit, prep
from xsurrogate_all import M, NAMES, log

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [5, 10, 15]
NB_DIR = "cache/ksweep"


def nnomp_snapshots(H, D, ks):
    """Greedy NN-OMP recording the reconstruction at each k in ks."""
    r = H.clone()
    kmax = max(ks)
    sel = torch.zeros(H.shape[0], kmax, dtype=torch.long, device=DEV)
    out = {}
    for i in range(kmax):
        sel[:, i] = best_atom(r, D, sel[:, :i] if i else None)
        A = D[sel[:, :i + 1]].transpose(1, 2)
        c = nnls_refit(A, H.unsqueeze(-1))
        r = H - (A @ c).squeeze(-1)
        if i + 1 in ks:
            out[i + 1] = (H - r).clone()
    return out


def sweep(name):
    """-> {k: {layer: neighbors}}, {k: {layer: health}} on the REAL corpus."""
    cache = f"{NB_DIR}/{name}.pt"
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        return d["nbrs"], d["health"]
    hf, real_path, _, lens_path = M[name]
    acts = torch.load(real_path, weights_only=False)
    lens = torch.load(lens_path, map_location="cpu", weights_only=False)
    WU, w = get_WU_and_w(hf)
    WUeff = WU.to(DEV) * w.to(DEV)
    del WU
    WUeff -= WUeff.mean(dim=0, keepdim=True)

    nbrs = {k: {} for k in KS}
    health = {k: {} for k in KS}
    for L in sorted(acts.keys()):
        H = acts[L].to(DEV, torch.float32)
        q = torch.quantile(H.abs().flatten(), 0.95)
        H = H.clamp(-q, q)
        J = lens["J"][L].to(DEV, torch.float32)
        D = WUeff @ J
        D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
        snaps = nnomp_snapshots(H, D, KS)
        for k, HJ in snaps.items():
            health[k][L] = dict(
                var_share=float((HJ.norm(dim=1) ** 2
                                 / H.norm(dim=1) ** 2).mean()))
            nbrs[k][L] = neighbors(prep(HJ)).cpu()
        del H, J, D, snaps
        torch.cuda.empty_cache()
    del WUeff
    torch.cuda.empty_cache()
    torch.save(dict(nbrs=nbrs, health=health), cache)
    return nbrs, health


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    log("=== sparsity sweep on the real corpus (11 models) ===")
    for i, name in enumerate(NAMES):
        sweep(name)
        log(f"  [{i+1}/11] {name:>10}  done")
    log(f"\nwrote {NB_DIR}/; run xksweep_mean.py for the alignment numbers "
        "the appendix quotes")


if __name__ == "__main__":
    main()
