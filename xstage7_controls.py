"""Shuffled image-caption control for the text-vision measurement (section 4.2).

--shuffle : the shuffled image-caption control the paper reports. For each of
            the 44 (encoder, text model) pairs it finds the best raw layer
            pair, then re-pairs images with captions under 20 seeded
            permutations. Raw kernels on both sides; neighbour sets are built
            here straight from the Stage 1 activations, so no Jacobian and no
            cached feature blob is needed.
            -> results/control_shuffle.json
"""
import argparse
import json
import os

import numpy as np
import torch

from crossmodal_utils import load_pilot
from xkernels import mnn, raw_nbrs

pilot = load_pilot(dev="cpu")
N_SHUFFLE = 20
os.makedirs("results", exist_ok=True)


def permuted_mnn(sa, sb, perm):
    """m-NN when item i is re-paired with perm[i] on side b."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    sb_re = inv[sb[perm]]                         # neighbors of re-paired items
    return mnn(sa, sb_re)


def raw_nbrs_vision(vname):
    """Raw neighbour sets per layer, rebuilt from the Stage 1 activations."""
    layers = json.load(open("cache/vision/layers.json",
                            encoding="utf-8"))[vname]["layers"]
    return {L: raw_nbrs(torch.tensor(np.load(
        f"cache/vision/{vname}/eval_acts_L{L}.npy"))) for L in layers}


def raw_nbrs_text(tname):
    """Same for a text model: band layers plus the final block."""
    cfg = json.load(open("cache/text_acts/layers.json",
                         encoding="utf-8"))[tname]
    layers = list(cfg["band_layers"]) + [cfg["final"]]
    return {L: raw_nbrs(torch.tensor(np.load(
        f"cache/text_acts/{tname}_L{L}_pool.npy"))) for L in layers}


def shuffle_all():
    out = {}
    vnames = list(json.load(open("cache/vision/layers.json", encoding="utf-8")))
    text_nbrs = {t: raw_nbrs_text(t) for t in pilot.MODELS}
    for vname in vnames:
        vis = raw_nbrs_vision(vname)
        for tname, txt in text_nbrs.items():
            # observed max layer pair, then permute at that pair
            best, arg = -1.0, None
            for La, sa in vis.items():
                for Lb, sb in txt.items():
                    v = mnn(sa, sb)
                    if v > best:
                        best, arg = v, (La, Lb)
            sa, sb = vis[arg[0]], txt[arg[1]]
            vals = []
            for s_i in range(N_SHUFFLE):
                g = torch.Generator().manual_seed(s_i)
                vals.append(permuted_mnn(sa, sb,
                                         torch.randperm(1024, generator=g)))
            out[f"{vname}|{tname}"] = {"raw": {"observed": best,
                                               "shuffled_mean": float(np.mean(vals))}}
            print(f"  {vname} x {tname}: raw {best:.4f} "
                  f"(shuf {np.mean(vals):.4f})", flush=True)
    with open("results/control_shuffle.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()
    if args.shuffle:
        shuffle_all()
