"""Fetch the prefitted Neuronpedia J-lenses for the 11 text models.

One directory per model, each a dict with J[layer] a d x d fp16 matrix.
Model-to-lens-path mapping lives in step7_align.py::MODELS.
"""
from huggingface_hub import snapshot_download

LENSES = ["gpt2-small", "gemma-3-270m", "gemma-3-1b", "gemma-2-2b",
          "gemma-3-4b", "pythia-70m-deduped", "qwen3.5-0.8b", "qwen3-1.7b",
          "qwen3.5-2b-pt", "qwen3-4b", "qwen3.5-4b"]

path = snapshot_download(
    "neuronpedia/jacobian-lens",
    allow_patterns=[f"{d}/*" for d in LENSES],
    local_dir="lenses",
)
print("Downloaded to:", path)
