"""Measurement B: caption-side components against RAW pooled image kernels.

This is the paper's cross-modal measurement (section 4.2 and `tab:perencoder`
in appendix A). No vision-side Jacobian is involved anywhere: the J-lens is
applied only to the caption side, and the image side enters as raw pooled
patch features.

Per text model, at the Stage 1 caption band layers, the 1,024 pooled caption
activations are decomposed against the pilot dictionary (WUeff folding,
vocab-mean centering, real NNLS, k <= 25, 0.95 clip) into full / J / perp
components, preprocessed with the pilot's prep(), and scored by m-NN against
every vision encoder's raw features at all cached image layers.

Writes results/measB.json: one layer-pair grid per (text model, encoder,
component). The paper's numbers are the MEAN over each grid, taken downstream
by xmeanmain.py, xplot_pairlevel_fig1.py and verify_paper_numbers.py.
Idempotent per text model.
"""
import json
import os

import numpy as np
import torch

from crossmodal_utils import load_pilot
from xkernels import grid_scores, pilot_nbrs, raw_nbrs

DEV = "cuda"
pilot = load_pilot(dev=DEV)
OUT = "results/measB.json"
os.makedirs("results", exist_ok=True)


def decompose_model(name, cfg, band):
    """Returns {component: {L: neighbours}}."""
    WU, w = pilot.get_WU_and_w(cfg["hf"])
    WUeff = WU.to(DEV) * w.to(DEV)
    del WU
    WUeff -= WUeff.mean(dim=0, keepdim=True)      # pilot vocab-mean fix
    lens = torch.load(cfg["lens"], map_location="cpu", weights_only=False)

    comps = {c: {} for c in ("full", "J", "perp")}
    for L in band:
        H = torch.tensor(np.load(f"cache/text_acts/{name}_L{L}_pool.npy"),
                         device=DEV)
        q = torch.quantile(H.abs().flatten(), 0.95)
        H = H.clamp(-q, q)
        J = lens["J"][L].to(DEV, torch.float32)
        D = WUeff @ J
        D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)

        HJ = pilot.nnomp_batch(H, D)
        share = (HJ.norm(dim=1) ** 2 / H.norm(dim=1) ** 2).mean().item()
        assert share < 1.0, f"{name} L{L}: variance share {share} > 1, broken"

        perp = H - HJ
        comps["full"][L] = pilot_nbrs(H.cpu(), DEV)
        comps["J"][L] = pilot_nbrs(HJ.cpu(), DEV)
        comps["perp"][L] = pilot_nbrs(perp.cpu(), DEV)
        del D, HJ, perp
        torch.cuda.empty_cache()
    del WUeff
    torch.cuda.empty_cache()
    return comps


def main():
    results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    stage1 = json.load(open("cache/text_acts/layers.json", encoding="utf-8"))
    vision_layers = json.load(open("cache/vision/layers.json", encoding="utf-8"))
    vision_raw = {
        vname: {L: raw_nbrs(torch.tensor(np.load(
            f"cache/vision/{vname}/eval_acts_L{L}.npy")))
            for L in cfg["layers"]}
        for vname, cfg in vision_layers.items()}

    for tname, cfg in pilot.MODELS.items():
        if tname in results:
            print(f"{tname}: done, skipping")
            continue
        print(f"\n=== measurement B: {tname} ===", flush=True)
        comps = decompose_model(tname, cfg, stage1[tname]["band_layers"])
        results[tname] = {
            vname: {c: grid_scores(nbrs, vraw) for c, nbrs in comps.items()}
            for vname, vraw in vision_raw.items()}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(f"\nmeasurement B complete -> {OUT}")


if __name__ == "__main__":
    main()
