"""Build the shared, metric-independent feature cache for the sweep.

For every (side, model/encoder, component, layer) this stores the two things
every metric in the sweep needs:
  *_gram.npy : the n x n inner-product kernel of the paper-preprocessed
               features (cosine similarity, since rows are unit norm)
  *_u.npy    : the top-10 SVCCA basis (centred/scaled left singular vectors)

Both are derived from exactly the paper's pipeline, so they are metric
independent by construction -- which is why the per-metric folders share them
rather than each re-running the k=25 decomposition.

Sides:
  text : ../acts_{model}.pt              (1,000 pile-10k docs, all band layers)
  cap  : ../cache/text_acts/*_pool.npy   (1,024 WIT captions, 5 band layers)
  img  : ../cache/vision/{enc}/eval_acts_L{L}.npy (1,024 images, 6 layers)

Idempotent: existing outputs are skipped, so it is safe to re-run or resume.
"""
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


def save(side, key, gram, U):
    np.save(f"{OUT}/{side}/{key}_gram.npy", gram.cpu().numpy().astype(np.float32))
    np.save(f"{OUT}/{side}/{key}_u.npy", U.cpu().numpy().astype(np.float32))


def done(side, key):
    return (os.path.exists(f"{OUT}/{side}/{key}_gram.npy")
            and os.path.exists(f"{OUT}/{side}/{key}_u.npy"))


def emit(side, key, feats):
    gram = feats @ feats.T
    U = C.svcca_basis(feats)
    save(side, key, gram, U)


def build_text():
    for name, cfg in C.MODELS.items():
        acts = torch.load(os.path.join(C.ROOT, cfg["acts"]), weights_only=False)
        layers = sorted(acts.keys())
        if all(done("text", f"{name}_{c}_L{L}") for L in layers for c in C.COMPS):
            print(f"text {name}: cached, skipping", flush=True)
            del acts
            continue
        lens = torch.load(os.path.join(C.ROOT, cfg["lens"]), map_location="cpu",
                          weights_only=False)
        WUeff = C.unembed_eff(cfg["hf"])
        for L in layers:
            if all(done("text", f"{name}_{c}_L{L}") for c in C.COMPS):
                continue
            H = acts[L].to(C.DEV, torch.float32)
            q = torch.quantile(H.abs().flatten(), 0.95)
            H = H.clamp(-q, q)
            D = C.dictionary(lens, L, WUeff)
            comps = C.decompose(H, D)
            for c, X in comps.items():
                emit("text", f"{name}_{c}_L{L}", X)
            del D, comps, H
            torch.cuda.empty_cache()
            print(f"  text {name} L{L}", flush=True)
        del WUeff, acts, lens
        torch.cuda.empty_cache()


def build_cap():
    band = C.band_layers()
    for name, cfg in C.MODELS.items():
        layers = band[name]["band_layers"]
        if all(done("cap", f"{name}_{c}_L{L}") for L in layers for c in C.COMPS):
            print(f"cap {name}: cached, skipping", flush=True)
            continue
        lens = torch.load(os.path.join(C.ROOT, cfg["lens"]), map_location="cpu",
                          weights_only=False)
        WUeff = C.unembed_eff(cfg["hf"])
        for L in layers:
            if all(done("cap", f"{name}_{c}_L{L}") for c in C.COMPS):
                continue
            arr = np.load(os.path.join(
                C.ROOT, f"cache/text_acts/{name}_L{L}_pool.npy"))
            H = torch.tensor(arr, device=C.DEV, dtype=torch.float32)
            q = torch.quantile(H.abs().flatten(), 0.95)
            H = H.clamp(-q, q)
            D = C.dictionary(lens, L, WUeff)
            comps = C.decompose(H, D)
            for c, X in comps.items():
                emit("cap", f"{name}_{c}_L{L}", X)
            del D, comps, H
            torch.cuda.empty_cache()
            print(f"  cap {name} L{L}", flush=True)
        del WUeff, lens
        torch.cuda.empty_cache()


def build_img():
    vl = C.vision_layers()
    for enc, cfg in vl.items():
        for L in cfg["layers"]:
            key = f"{enc}_L{L}"
            if done("img", key):
                continue
            arr = np.load(os.path.join(
                C.ROOT, f"cache/vision/{enc}/eval_acts_L{L}.npy"))
            X = C.prep_image(torch.tensor(arr, device=C.DEV, dtype=torch.float32))
            emit("img", key, X)
            print(f"  img {enc} L{L}", flush=True)


def main():
    for side in ("text", "cap", "img"):
        os.makedirs(f"{OUT}/{side}", exist_ok=True)
    t0 = time.time()
    build_img()
    print(f"[img done {time.time()-t0:.0f}s]", flush=True)
    build_cap()
    print(f"[cap done {time.time()-t0:.0f}s]", flush=True)
    build_text()
    print(f"[text done {time.time()-t0:.0f}s]", flush=True)

    def layers_on_disk(side, name):
        pre, suf = name + "_full_L", "_gram.npy"
        return sorted(int(f[len(pre):-len(suf)])
                      for f in os.listdir(f"{OUT}/{side}")
                      if f.startswith(pre) and f.endswith(suf))

    manifest = dict(
        text_layers={n: layers_on_disk("text", n) for n in C.MODELS},
        cap_layers={n: C.band_layers()[n]["band_layers"] for n in C.MODELS},
        img_layers={e: C.vision_layers()[e]["layers"] for e in C.ENCODERS},
        comps=C.COMPS, topk=C.TOPK, cca_dim=C.CCA_DIM)
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"feature cache complete ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
