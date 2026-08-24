"""Vision-encoder config and loaders for the text-vision stages.

Used only by xstage1_vision.py, which caches the raw pooled patch features
that measurement B (section 4.2) scores the caption components against.

attn_implementation='eager' is pinned: the cached image activations behind
results/measB.json were produced with it, and switching to SDPA kernels would
change them."""
import torch

# patch_start: index of the first patch token (CLS excluded when 1)
VISION = {
    "dinov2": dict(hf="facebook/dinov2-base", patch_start=1, expected_T=257),
    "mae":    dict(hf="facebook/vit-mae-base", patch_start=1, expected_T=197),
    # local safetensors conversion of openai/clip-vit-base-patch16 (pickle-only
    # repo, refused by transformers on torch<2.6) — see xconvert_clip.py
    "clip":   dict(hf="models/clip-vit-base-patch16", patch_start=1, expected_T=197),
    "siglip": dict(hf="google/siglip-base-patch16-224", patch_start=0, expected_T=196),
}


def load_model(name):
    from transformers import (AutoImageProcessor, AutoModel, CLIPVisionModel,
                              SiglipVisionModel, ViTMAEModel)
    cfg = VISION[name]
    proc = AutoImageProcessor.from_pretrained(cfg["hf"])
    kw = {"attn_implementation": "eager"}
    if name == "clip":
        model = CLIPVisionModel.from_pretrained(cfg["hf"], **kw)
    elif name == "siglip":
        model = SiglipVisionModel.from_pretrained(cfg["hf"], **kw)
    elif name == "mae":
        model = ViTMAEModel.from_pretrained(cfg["hf"], mask_ratio=0.0, **kw)
    else:
        model = AutoModel.from_pretrained(cfg["hf"], **kw)
    return proc, model.to("cuda", torch.float32).eval()


def forward_kwargs(name, cfg, batch_size):
    """Per-model forward extras. MAE: its masking machinery shuffles patch
    order with random noise even at mask_ratio=0 — identity noise makes the
    shuffle a no-op, so forwards are deterministic and sequences canonical."""
    if name == "mae":
        n_patches = cfg["expected_T"] - 1         # minus CLS
        noise = torch.arange(n_patches, dtype=torch.float32, device="cuda")
        return {"noise": noise.unsqueeze(0).expand(batch_size, -1)}
    return {}
