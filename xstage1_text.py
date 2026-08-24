"""Stage 1 (text): cache pooled caption activations for all 11 pilot models.

These are the caption side of measurement B (section 4.2): xstage6_measB.py
decomposes them into J / non-J / full components, and xstage7_controls.py
builds the shuffled control's raw text kernels from them.

Per model: 5 evenly spaced lens layers inside the pilot band (all of them if
the band is smaller), plus the final block. Each layer is mean-pooled over
all token positions (pilot step6 convention: single sequence, no padding,
hs[L][0].mean(dim=0)). Dtypes match the pilot (gpt2 fp16, everything else
bf16), stored fp32. Outputs cache/text_acts/{model}_L{layer}_pool.npy,
(1024, d), plus the layer sidecar cache/text_acts/layers.json.
Idempotent per model."""
import json
import os

import numpy as np
import torch

from crossmodal_utils import load_pilot

pilot = load_pilot()
OUT = "cache/text_acts"
os.makedirs(OUT, exist_ok=True)
# pins the numerics of the cached activations behind results/measB.json
torch.backends.cuda.matmul.allow_tf32 = False

DTYPES = {"gpt2": torch.float16}                 # pilot: fp16 gpt2, bf16 rest


def choose_layers(band):
    if len(band) <= 5:
        return list(band)
    idx = np.round(np.linspace(0, len(band) - 1, 5)).astype(int)
    return sorted({band[i] for i in idx})


def forward_acts(model, tok, caption, layers, final_idx):
    ids = tok(caption, return_tensors="pt").to("cuda")
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    assert hs is not None and len(hs) == final_idx + 1, \
        f"expected {final_idx + 1} hidden states, got {hs and len(hs)}"
    out = {}
    for L in layers + [final_idx]:
        h = hs[L][0].float()
        assert torch.isfinite(h).all(), f"non-finite at L{L}: {caption[:40]!r}"
        out[L] = h.mean(dim=0).cpu()
    return out


def main():
    captions = [p["caption"] for p in
                json.load(open("eval_manifest.json", encoding="utf-8"))["pairs"]]
    n = len(captions)
    sidecar = f"{OUT}/layers.json"
    layer_choice = json.load(open(sidecar, encoding="utf-8")) if \
        os.path.exists(sidecar) else {}

    for name, cfg in pilot.MODELS.items():
        band = sorted(torch.load(cfg["acts"], weights_only=False).keys())
        layers = choose_layers(band)
        if os.path.exists(f"{OUT}/{name}_L{layers[0]}_pool.npy"):
            print(f"{name}: cached, skipping")    # sidecar entry already loaded
            continue

        print(f"{name}: layers {layers} + final")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["hf"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf"], dtype=DTYPES.get(name, torch.bfloat16))
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "vision_tower"):
            inner.vision_tower = None
            if hasattr(inner, "multi_modal_projector"):
                inner.multi_modal_projector = None
        model = model.to("cuda").eval()

        probe = model(**tok("probe", return_tensors="pt").to("cuda"),
                      output_hidden_states=True).hidden_states
        final_idx = len(probe) - 1

        acc = {L: [] for L in layers + [final_idx]}
        for i, cap in enumerate(captions):
            for L, pool in forward_acts(model, tok, cap, layers,
                                        final_idx).items():
                acc[L].append(pool)
            if (i + 1) % 256 == 0:
                print(f"  {i + 1}/{n}", flush=True)

        for L, pools in acc.items():
            np.save(f"{OUT}/{name}_L{L}_pool.npy", torch.stack(pools).numpy())
        layer_choice[name] = {"band_layers": layers, "final": final_idx}
        del model
        torch.cuda.empty_cache()

    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(layer_choice, f, indent=2)
    print("stage 1 text caching done")


if __name__ == "__main__":
    main()
