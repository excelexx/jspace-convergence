"""Competence axis for all three experiments: HellaSwag 0-shot (limit 1500,
acc_norm) for the 11 text models, through one consistent lm_eval harness.

`hellaswag_acc_norm` is the x-axis of Figure 2, Figure 4, tab:competence and
tab:crossaxes. Results go to results/lmeval/{model}.json, printed as each model
finishes; idempotent, so an interrupted sweep resumes.

Requires `lm_eval` importable; set PYTHONPATH to your lm-eval checkout if it is
installed outside site-packages."""
import json
import os

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM

# hf id, batch size (memory-scaled; 3080 10GB)
MODELS = {
    "pythia70m": ("EleutherAI/pythia-70m-deduped", 32),
    "gpt2":      ("gpt2", 32),
    "gemma270":  ("google/gemma-3-270m", 32),
    "qwen08b":   ("Qwen/Qwen3.5-0.8B", 16),
    "gemma":     ("google/gemma-3-1b-pt", 16),
    "qwen17b":   ("Qwen/Qwen3-1.7B", 16),
    "qwen2b":    ("Qwen/Qwen3.5-2B-Base", 8),
    "gemma2_2b": ("google/gemma-2-2b", 8),
    "qwen4b":    ("Qwen/Qwen3-4B", 4),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", 4),
    "gemma3_4b": ("google/gemma-3-4b-pt", 4),
}
LIMIT_HS = 1500
os.makedirs("results/lmeval", exist_ok=True)

for name, (hf, bs) in MODELS.items():
    out = f"results/lmeval/{name}.json"
    if os.path.exists(out):
        print(f"{name}: done, skipping")
        continue
    print(f"=== {name} ({hf}, bs {bs}) ===", flush=True)
    lm = HFLM(pretrained=hf, dtype="bfloat16", batch_size=bs)
    r_hs = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"],
                                   limit=LIMIT_HS)["results"]["hellaswag"]
    print(f"  hellaswag acc_norm = {r_hs['acc_norm,none']:.4f}", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": name, "hf": hf,
                   "hellaswag_acc_norm": r_hs["acc_norm,none"]}, f, indent=2)
    del lm
    torch.cuda.empty_cache()

print("benchmark sweep complete")
