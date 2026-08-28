"""Gaussian-dictionary control for the content ablation (Table 1, lower block).

Runs the identical pipeline with the dictionary replaced by a Gaussian one:
same cached activations, same NNOMP (k = 25), same preprocessing, neighbours and
m-NN, same band layers, same 55 pairs. Only the dictionary changes, to
randn(m_A, d_A) with unit-norm rows.

Only the J component is decomposed. Table 1's Gaussian row is the J row, and
the full activation does not depend on the dictionary at all. The J-lens side
of the comparison is not recomputed here; xmedlayer.py reads it from
cache/surr55, as written by xsurrogate_all.py.

Dictionaries are seeded from (variant, model, layer) so the real and ablated
corpora are decomposed against the SAME dictionary; otherwise the retention
ratio would compare two different instruments.

Output is cache/randnull/{corpus}_{model}.pt, the per-layer neighbour lists.
xmedlayer.py pairs them with the J-lens lists and produces
results/randdict_null_bylayerstat.json, the file Table 1 is built from.
"""
import hashlib
import os
import traceback

import torch

from xsurrogate import get_WU_and_w, neighbors, nnomp_batch, prep
from xsurrogate_all import M, NAMES, surr_acts_path

DEV = "cuda" if torch.cuda.is_available() else "cpu"
COMPS = ["J"]
VARIANT = "rand"                           # `real` comes from cache/surr55
OLD_DIR = "cache/surr55"
NB_DIR = "cache/randnull"


def log(msg):
    print(msg, flush=True)


def seed_of(variant, name, L):
    h = hashlib.md5(f"{variant}|{name}|{L}".encode()).hexdigest()[:8]
    return int(h, 16)


def make_dict(n_vocab, d_model, seed):
    """Gaussian dictionary shaped like the J-lens one, with unit-norm rows.

    Only the shape is shared with the real dictionary; the paper describes this
    arm as a Gaussian dictionary replacing the J-lens one, nothing more."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    D = torch.randn(n_vocab, d_model, generator=g, device=DEV)
    return D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)


def decompose(name, corpus):
    """{variant: {component: {layer: nbrs}}}, cached."""
    cache = f"{NB_DIR}/{corpus}_{name}.pt"
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)["nbrs"]
    hf, real_path, _, lens_path = M[name]
    acts = torch.load(real_path if corpus == "real" else surr_acts_path(name),
                      weights_only=False)
    lens = torch.load(lens_path, map_location="cpu", weights_only=False)
    WU, _ = get_WU_and_w(hf)
    n_vocab = WU.shape[0]
    del WU

    nbrs = {VARIANT: {c: {} for c in COMPS}}
    for L in sorted(acts.keys()):
        H0 = acts[L].to(DEV, torch.float32)
        q = torch.quantile(H0.abs().flatten(), 0.95)
        H0 = H0.clamp(-q, q)
        D = make_dict(n_vocab, lens["J"][L].shape[1],
                      seed_of(VARIANT, name, L))
        HJ = nnomp_batch(H0, D)
        nbrs[VARIANT]["J"][L] = neighbors(prep(HJ)).cpu()
        del D, HJ, H0
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    os.makedirs(NB_DIR, exist_ok=True)
    torch.save(dict(nbrs=nbrs), cache)
    return nbrs


def load_real(name, corpus):
    """The J-lens neighbour lists for the same model and corpus."""
    d = torch.load(f"{OLD_DIR}/{corpus}_{name}.pt", weights_only=False)
    return d["nbrs"]


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    for corpus in ["real", "surrogate"]:
        for name in NAMES:
            try:
                decompose(name, corpus)
                log(f"  [{corpus:>9}] {name:>10}  done")
            except Exception:
                log(f"  FAILED {corpus}/{name}\n{traceback.format_exc()}")
    log(f"\nwrote {NB_DIR}/; run xmedlayer.py to build Table 1's source file")


if __name__ == "__main__":
    main()
