"""Lens-fitting control, step 4: decompose the eval activations under each
fitting condition.

For every in-scope model and every layer that passed the xlc_phase2.py gate,
the 1,000 Pile evaluation activations are decomposed against three
dictionaries: one built from the downloaded full-corpus lens, and one from
each half-corpus refit. The code path is the main experiment's, unchanged --
WUeff = W_U * folded norm weights with the vocabulary mean subtracted,
non-negative OMP with an NNLS refit, k <= 25 atoms, activations clipped at
their 95th absolute percentile. What is stored is the m-NN neighbour set of
the J component, per (variant, layer); xlc_phase45_score.py and
xlc_median_delta.py pair those sets across models into the shared-full,
shared-half and crossed-half conditions of Appendix C.1.

The random-dictionary null draws random rows of WUeff and never touches a
lens, so it is identical under all three conditions: it is computed once per
(model, layer) from R=19 draws (seeds 1000+i) and reused.

Outputs: results/lenscontrol/sparse/{model}.pt -- a neighbour-set cache, not
shipped with the release.
"""
import json
import os

import torch

from crossmodal_jacobian import load_mean_J
from crossmodal_utils import LENSFIT_SCOPE, load_pilot
from xkernels import pilot_nbrs

DEV = "cuda"
pilot = load_pilot(dev=DEV)
GATES = "results/lenscontrol/phase2_gates.json"
R_NULL = 19
gates = json.load(open(GATES, encoding="utf-8"))
os.makedirs("results/lenscontrol/sparse", exist_ok=True)

for name in LENSFIT_SCOPE:
    ckpt = f"results/lenscontrol/sparse/{name}.pt"
    if os.path.exists(ckpt):
        print(f"{name}: cached, skipping")
        continue
    band = [int(L) for L in gates[name] if gates[name][L]["pass"]]
    cfg = pilot.MODELS[name]
    print(f"=== sparse decomposition: {name}, layers {band} ===", flush=True)
    WU, w = pilot.get_WU_and_w(cfg["hf"])
    WUeff = WU.to(DEV) * w.to(DEV)
    del WU
    WUeff -= WUeff.mean(dim=0, keepdim=True)
    Vsz = WUeff.shape[0]
    lens = torch.load(cfg["lens"], map_location="cpu", weights_only=False)
    pile = torch.load(cfg["acts"], weights_only=False)

    Drs = []                                      # lens-independent nulls
    for r in range(R_NULL):
        g = torch.Generator().manual_seed(1000 + r)
        rows = torch.randperm(Vsz, generator=g)[: min(Vsz, 20000)]
        Dr = WUeff[rows]
        Drs.append(Dr / Dr.norm(dim=1, keepdim=True).clamp(min=1e-8))

    cond = {v: {} for v in ("full", "h1", "h2")}
    null = [dict() for _ in range(R_NULL)]
    for L in band:
        H0 = pile[L].to(DEV, torch.float32)
        q = torch.quantile(H0.abs().flatten(), 0.95)
        H = H0.clamp(-q, q)
        Js = {"full": lens["J"][L].to(DEV, torch.float32),
              "h1": load_mean_J(
                  f"results/lenscontrol/jfit/{name}_L{L}_h1.pt").to(DEV),
              "h2": load_mean_J(
                  f"results/lenscontrol/jfit/{name}_L{L}_h2.pt").to(DEV)}
        for variant, J in Js.items():
            D = WUeff @ J
            D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
            HJ = pilot.nnomp_batch(H, D)
            share = (HJ.norm(dim=1) ** 2 / H.norm(dim=1) ** 2).mean().item()
            assert share < 1.0, f"{name} L{L} {variant}: share {share}"
            cond[variant][L] = pilot_nbrs(HJ.cpu(), DEV)
            del D, HJ
            torch.cuda.empty_cache()
        for r, Dr in enumerate(Drs):
            null[r][L] = pilot_nbrs(pilot.nnomp_batch(H, Dr).cpu(), DEV)
        del H0, H, Js
        torch.cuda.empty_cache()
    torch.save({"cond": cond, "null": null}, ckpt)
    print(f"  saved {ckpt}", flush=True)
    del WUeff, Drs
    torch.cuda.empty_cache()

print("phase 4 decompositions complete")
