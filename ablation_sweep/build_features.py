"""Build the feature cache for the content-ablation / random-dictionary sweep.

Reproduces the paper's Table 1 setup, but caches Gram matrices (and SVCCA
bases) so any of the eight metrics can be scored on it, not just mutual
kappa-NN.

Only the parts that are NOT already known are computed here. The real-corpus,
J-lens-dictionary alignment was produced by the 8-metric sweep and is copied in
by copy_real.py, so this builds:

  abl/{model}_{full,J,perp}_L{L}   content-ablated corpus, J-lens dictionary
  real/{model}_gaussJ_L{L}         real corpus, Gaussian dictionary
  abl/{model}_gaussJ_L{L}          ablated corpus, Gaussian dictionary

Inputs (read-only): ../acts_{model}.pt, ../acts_surr_{model}.pt, ../lenses/.
Nothing outside ablation_sweep/ is written.

The Gaussian dictionary follows ../xrandnull.py exactly: seeded from
md5("rand|{model}|{layer}"), so the SAME random dictionary serves the real and
ablated corpora -- which is what makes their retention comparable.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import common as C

OUT = "_features"
RAND_VARIANT = "rand"                       # matches ../xrandnull.py VARIANT
JLENS_COMPS = ("full", "J", "perp")
# what this script is responsible for: (corpus, component)
TARGETS = [("abl", "full"), ("abl", "J"), ("abl", "perp"),
           ("real", "gaussJ"), ("abl", "gaussJ")]


def seed_of(name, L):
    """../xrandnull.py seed_of(variant, name, L)."""
    h = hashlib.md5(f"{RAND_VARIANT}|{name}|{L}".encode()).hexdigest()[:8]
    return int(h, 16)


def gauss_dict(vocab, width, seed):
    g = torch.Generator(device=C.DEV).manual_seed(seed)
    D = torch.randn(vocab, width, generator=g, device=C.DEV)
    return D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)


def gpath(corpus, key):
    return f"{OUT}/{corpus}/{key}_gram.npy"


def upath(corpus, key):
    return f"{OUT}/{corpus}/{key}_u.npy"


def emit(corpus, key, feats):
    np.save(gpath(corpus, key), (feats @ feats.T).cpu().numpy().astype(np.float32))
    U = C.svcca_basis(feats)                # cca_dim=10, as in the 8-metric sweep
    np.save(upath(corpus, key), U.cpu().numpy().astype(np.float32))


def have(corpus, key):
    return os.path.exists(gpath(corpus, key)) and os.path.exists(upath(corpus, key))


def done_layer(name, L, targets=TARGETS):
    return all(have(c, f"{name}_{comp}_L{L}") for c, comp in targets)


def clamped(acts, L):
    H = acts[L].to(C.DEV, torch.float32)
    q = torch.quantile(H.abs().flatten(), 0.95)
    return H.clamp(-q, q)


def build_model(name, cfg, also_real_jlens=False):
    """also_real_jlens rebuilds the copied real columns, for validation only."""
    targets = list(TARGETS)
    if also_real_jlens:
        targets += [("real", c) for c in JLENS_COMPS]
    real = torch.load(os.path.join(C.ROOT, cfg["acts"]), weights_only=False)
    abl = torch.load(os.path.join(C.ROOT, f"acts_surr_{name}.pt"),
                     weights_only=False)
    layers = sorted(real.keys())
    assert sorted(abl.keys()) == layers, f"layer mismatch for {name}"
    if all(done_layer(name, L, targets) for L in layers):
        print(f"{name}: cached", flush=True)
        return
    lens = torch.load(os.path.join(C.ROOT, cfg["lens"]), map_location="cpu",
                      weights_only=False)
    WUeff = C.unembed_eff(cfg["hf"])
    vocab, width = WUeff.shape
    for L in layers:
        if done_layer(name, L, targets):
            continue
        H = {"real": clamped(real, L), "abl": clamped(abl, L)}

        # --- J-lens dictionary --------------------------------------------
        need_j = [c for c in ("abl", "real")
                  if any(cc == c and comp in JLENS_COMPS for cc, comp in targets)]
        if need_j:
            D = C.dictionary(lens, L, WUeff)
            for corpus in need_j:
                HJ = C.nnomp_batch(H[corpus], D)
                emit(corpus, f"{name}_full_L{L}", C.prep(H[corpus]))
                emit(corpus, f"{name}_J_L{L}", C.prep(HJ))
                emit(corpus, f"{name}_perp_L{L}", C.prep(H[corpus] - HJ))
                del HJ
            del D
            torch.cuda.empty_cache()

        # --- Gaussian dictionary, J component only -------------------------
        Dg = gauss_dict(vocab, width, seed_of(name, L))
        for corpus in ("real", "abl"):
            HJg = C.nnomp_batch(H[corpus], Dg)
            emit(corpus, f"{name}_gaussJ_L{L}", C.prep(HJg))
            del HJg
        del Dg, H
        torch.cuda.empty_cache()
        print(f"  {name} L{L}", flush=True)
    del WUeff, real, abl, lens
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list; default all")
    ap.add_argument("--also_real_jlens", action="store_true",
                    help="also rebuild the copied real columns (validation)")
    args = ap.parse_args()
    names = ([n.strip() for n in args.models.split(",")] if args.models
             else list(C.MODELS))

    for c in ("real", "abl"):
        os.makedirs(f"{OUT}/{c}", exist_ok=True)
    t0 = time.time()
    for name in names:
        build_model(name, C.MODELS[name], args.also_real_jlens)
        print(f"[{name} done {time.time()-t0:.0f}s]", flush=True)

    if not args.models:
        def layers_on_disk(name):
            pre, suf = f"{name}_J_L", "_gram.npy"
            return sorted(int(f[len(pre):-len(suf)])
                          for f in os.listdir(f"{OUT}/abl")
                          if f.startswith(pre) and f.endswith(suf))

        with open(f"{OUT}/manifest.json", "w") as f:
            json.dump(dict(layers={n: layers_on_disk(n) for n in C.MODELS},
                           targets=[list(t) for t in TARGETS],
                           rand_variant=RAND_VARIANT, cca_dim=C.CCA_DIM,
                           note="ablated corpus (J-lens dict) and Gaussian "
                                "dictionary (both corpora); real J-lens "
                                "columns are copied by copy_real.py"),
                      f, indent=1)
        print(f"feature cache complete ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
