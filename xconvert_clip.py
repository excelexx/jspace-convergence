"""One-time: convert openai/clip-vit-base-patch16 (pickle-only repo) to a
local safetensors copy at models/clip-vit-base-patch16, since transformers
refuses torch.load-format checkpoints on torch < 2.6. Weights are loaded with
weights_only=True (tensor data only, no pickle code execution).

Run before xstage1_vision.py: it is how the CLIP encoder of section 4.2
loads (xvision_config.VISION points at the converted copy)."""
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

DST = "models/clip-vit-base-patch16"

if os.path.exists(f"{DST}/model.safetensors"):
    print("already converted")
    raise SystemExit(0)

snapshot_download("openai/clip-vit-base-patch16", local_dir=DST,
                  allow_patterns=["config.json", "preprocessor_config.json",
                                  "pytorch_model.bin"])
sd = torch.load(f"{DST}/pytorch_model.bin", map_location="cpu",
                weights_only=True)
sd = {k: v.contiguous().clone() for k, v in sd.items()
      if not k.endswith("position_ids")}         # legacy buffers, not weights
save_file(sd, f"{DST}/model.safetensors")
os.remove(f"{DST}/pytorch_model.bin")
print(f"converted {len(sd)} tensors -> {DST}/model.safetensors")
