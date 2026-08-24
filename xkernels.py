"""Shared m-NN machinery for the text-vision stages (paper section 4.2).

Consumers: xstage6_measB.py (measurement B), xstage7_controls.py (the
shuffled image-caption control) and xlc_phase2.py (the lens-fitting control's
text-side stability check). `mnn` is the pilot's own implementation, loaded
verbatim out of step7_align.py, at kappa = 10.

`preprocess` here clips each dimension at the 0.5/99.5 percentiles of the
eval set and then l2-normalises the rows; it is what the IMAGE side of
measurement B runs. The caption side is preprocessed with the pilot's
`prep` (global 0.95-quantile clamp) via `pilot_nbrs` instead.
"""
import numpy as np
import torch

from crossmodal_utils import load_pilot

_pilot = load_pilot(dev="cpu")
K_NN = _pilot.K_NN                                # 10
mnn = _pilot.mnn                                  # pilot implementation, verbatim


def preprocess(X):
    """(n, d) float32 tensor -> per-coordinate 0.5/99.5 percentile clip, then
    row-normalized."""
    lo = torch.quantile(X, 0.005, dim=0, keepdim=True)
    hi = torch.quantile(X, 0.995, dim=0, keepdim=True)
    Xc = torch.clamp(X, min=lo, max=hi)
    return Xc / Xc.norm(dim=1, keepdim=True).clamp(min=1e-8)


def neighbors(X, k=K_NN):
    K = X @ X.T
    K.fill_diagonal_(-torch.inf)
    return K.topk(k, dim=1).indices


def raw_nbrs(X):
    """One layer's raw features -> preprocessed (n, k) neighbour indices."""
    return neighbors(preprocess(X))


def pilot_nbrs(X, dev="cuda"):
    """Pilot preprocessing (0.95-quantile clamp + l2), then neighbour list."""
    Xp = _pilot.prep(X.to(dev))
    K = Xp @ Xp.T
    K.fill_diagonal_(-torch.inf)
    return K.topk(K_NN, dim=1).indices.cpu()


def grid_scores(nbrs_a, nbrs_b):
    """m-NN over all (layer_a, layer_b) combinations of two {layer: neighbour
    index} dicts. The paper's cross-modal numbers are the MEAN over `grid`,
    recomputed from the stored cells by xmeanmain.py."""
    return {"grid": {f"{La}x{Lb}": mnn(sa, sb)
                     for La, sa in nbrs_a.items()
                     for Lb, sb in nbrs_b.items()}}


def grid_max(nbrs_a, nbrs_b):
    """Max m-NN alignment over the layer-pair grid of two neighbour sets."""
    return max(mnn(sa, sb)
               for sa in nbrs_a.values() for sb in nbrs_b.values())


def grid_median(nbrs_a, nbrs_b):
    """Median m-NN alignment over the layer-pair grid of two neighbour sets."""
    return float(np.median([mnn(sa, sb)
                            for sa in nbrs_a.values()
                            for sb in nbrs_b.values()]))
