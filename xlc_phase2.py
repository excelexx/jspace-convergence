"""Lens-fitting control, step 2: stability gate on the half-fitted lenses.

Per (model, band layer) this compares the h1 and h2 refits two ways:

  * subspace overlap Omega between their top-r right-singular subspaces, with
    the chance level r/d divided out: Omega* = (Omega - r/d) / (1 - r/d);
  * m-NN agreement of the two induced kernels on the 1,000 Pile evaluation
    activations, reported per cell.

A cell passes when Omega* >= 0.5. xlc_phase4.py decomposes only the cells that
pass, so this threshold fixes the layer grid behind Appendix C.1's +0.0002.

Writes results/lenscontrol/phase2_gates.json.
"""
import json

import numpy as np
import torch

from crossmodal_jacobian import load_mean_J
from crossmodal_utils import LENSFIT_SCOPE, load_pilot
from xkernels import mnn, neighbors, preprocess

pilot = load_pilot(dev="cpu")
OUT = "results/lenscontrol/phase2_gates.json"
stage1 = json.load(open("cache/text_acts/layers.json", encoding="utf-8"))


def topr_V(J, r):
    return torch.linalg.svd(J)[2][:r]


out = {}
for name in LENSFIT_SCOPE:
    band = stage1[name]["band_layers"]
    pile = torch.load(pilot.MODELS[name]["acts"], weights_only=False)
    for L in band:
        J1 = load_mean_J(f"results/lenscontrol/jfit/{name}_L{L}_h1.pt")
        J2 = load_mean_J(f"results/lenscontrol/jfit/{name}_L{L}_h2.pt")
        d = J1.shape[0]
        S = torch.linalg.svdvals(0.5 * (J1 + J2))
        r = int(np.ceil(float((S.sum() ** 2) / (S ** 2).sum())))
        s = torch.linalg.svdvals(topr_V(J1, r) @ topr_V(J2, r).T).clamp(0, 1)
        omega = float((s ** 2).sum() / r)
        omega_star = (omega - r / d) / (1 - r / d)

        Hp = pile[L].float()                      # (1000, d) Pile eval acts
        k1 = neighbors(preprocess(Hp @ J1.T))
        k2 = neighbors(preprocess(Hp @ J2.T))
        knn12 = mnn(k1, k2)

        cell = {"r": r, "omega_star": float(omega_star),
                "knn_h1h2_pile": knn12,
                "pass": bool(omega_star >= 0.5)}
        out.setdefault(name, {})[str(L)] = cell
        print(f"{name} L{L}: Omega*={omega_star:.3f} (r={r}/{d}) | knn "
              f"{knn12:.3f} | "
              f"{'PASS' if cell['pass'] else '** FAIL - excluded **'}",
              flush=True)

n_pass = sum(c["pass"] for m in out.values() for c in m.values())
n_all = sum(len(m) for m in out.values())
print(f"\nphase 2: {n_pass}/{n_all} cells pass the stability gate")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"wrote {OUT}")
