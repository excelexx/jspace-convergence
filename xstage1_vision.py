"""Stage 1 (vision): cache the raw pooled image features for the 4 encoders.

These are the image kernels of measurement B (section 4.2): the caption side
carries the J-lens components, the image side enters raw. No Jacobian is
estimated on the vision side anywhere in this release.

Per model, 5 evenly spaced blocks + the final block (absolute hidden_states
indices; index 0 is the embeddings). All h^(L) are block outputs BEFORE any
post-layernorm / projection / pooling head. The 1,024 WIT eval images are
pooled over PATCH TOKENS ONLY (CLS/register tokens excluded; siglip has
none), fp32 -> cache/vision/{model}/eval_acts_L{L}.npy, plus the layer
sidecar cache/vision/layers.json. Both are read by xstage6_measB.py and
xstage7_controls.py.

Token layout per model is asserted against expected_T, so a wrong patch_start
cannot silently pool the CLS token in. TF32 off. Idempotent per model."""
import json
import os

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from xvision_config import VISION, forward_kwargs, load_model

BATCH = 16
_NAME = {}                                        # model object -> config name


def forward_hs(model, proc, pil_images, expected_T):
    inputs = proc(images=pil_images, return_tensors="pt").to("cuda")
    extra = forward_kwargs(_NAME[id(model)], VISION[_NAME[id(model)]],
                           len(pil_images))
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True, **extra).hidden_states
    assert hs is not None, "no hidden states returned"
    T = hs[0].shape[1]
    assert T == expected_T, f"sequence length {T} != expected {expected_T}"
    return hs


def images_from(folder, n):
    return [Image.open(f"{folder}/{i:04d}.png").convert("RGB") for i in range(n)]


def main():
    layer_choice = {}
    eval_imgs = images_from("cache/images/eval", 1024)

    for name, cfg in VISION.items():
        outdir = f"cache/vision/{name}"
        os.makedirs(outdir, exist_ok=True)
        proc, model = load_model(name)
        _NAME[id(model)] = name

        probe = forward_hs(model, proc, [eval_imgs[0]], cfg["expected_T"])
        n_states = len(probe)                     # embeddings + blocks
        final = n_states - 1
        blocks = np.round(np.linspace(2, final - 2, 5)).astype(int)
        layers = sorted(set(blocks.tolist()) | {final})
        layer_choice[name] = {"layers": layers, "final": final,
                              "expected_T": cfg["expected_T"],
                              "patch_start": cfg["patch_start"]}
        if os.path.exists(f"{outdir}/eval_acts_L{layers[0]}.npy"):
            print(f"{name}: cached, skipping")
            del model
            torch.cuda.empty_cache()
            continue
        print(f"{name}: layers {layers} (final {final}, T={cfg['expected_T']})")

        # pooled patch tokens, one row per eval image
        pooled = {L: [] for L in layers}
        for s in range(0, len(eval_imgs), BATCH):
            hs = forward_hs(model, proc, eval_imgs[s:s + BATCH], cfg["expected_T"])
            for L in layers:
                h = hs[L][:, cfg["patch_start"]:, :].float()
                assert torch.isfinite(h).all(), f"{name} L{L}: non-finite"
                pooled[L].append(h.mean(dim=1).cpu())
            if (s // BATCH) % 16 == 0:
                print(f"  eval {s}/{len(eval_imgs)}", flush=True)

        for L in layers:
            P = torch.cat(pooled[L]).numpy()
            assert P.shape[0] == 1024
            np.save(f"{outdir}/eval_acts_L{L}.npy", P)

        del model
        torch.cuda.empty_cache()

    with open("cache/vision/layers.json", "w", encoding="utf-8") as f:
        json.dump(layer_choice, f, indent=2)
    print("stage 1 vision caching done")


if __name__ == "__main__":
    main()
