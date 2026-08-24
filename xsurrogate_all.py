"""Content ablation across all 11 models (section 4.1, Table 1).

Extracts pooled activations on the ablated corpus built by xsurrogate.py, then
decomposes both the real and the ablated activations against the J-lens
dictionary at every band layer of every model.

The per-model neighbour lists are cached to cache/surr55/. The run is
resumable; a model whose cache exists is not refit.

Writes results/surrogate_55.json, whose health block supplies the 18.3% / 13.5%
J variance shares quoted in the paper (limitations, and appendix C.2).
"""
import json
import os
import sys

import numpy as np
import torch

from xsurrogate import (MAX_TOKENS, N_DOCS, SURR_TEXT, get_WU_and_w, neighbors,
                        nnomp_batch, prep)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
COMPS = ["full", "J", "perp"]
NB_DIR = "cache/surr55"
FP16 = torch.float16
BF16 = torch.bfloat16

M = {
    "gpt2": ("gpt2", "acts_gpt2.pt", FP16,
             "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
    "gemma": ("google/gemma-3-1b-pt", "acts_gemma.pt", BF16,
              "lenses/gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt"),
    "gemma270": ("google/gemma-3-270m", "acts_gemma270.pt", BF16,
                 "lenses/gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt"),
    "pythia70m": ("EleutherAI/pythia-70m-deduped", "acts_pythia70m.pt", BF16,
                  "lenses/pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"),
    "qwen08b": ("Qwen/Qwen3.5-0.8B", "acts_qwen08b.pt", BF16,
                "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"),
    "qwen17b": ("Qwen/Qwen3-1.7B", "acts_qwen17b.pt", BF16,
                "lenses/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"),
    "qwen2b": ("Qwen/Qwen3.5-2B-Base", "acts_qwen2b.pt", BF16,
               "lenses/qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt"),
    "gemma2_2b": ("google/gemma-2-2b", "acts_gemma2_2b.pt", BF16,
                  "lenses/gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt"),
    "qwen4b": ("Qwen/Qwen3-4B", "acts_qwen4b.pt", BF16,
               "lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "acts_qwen35_4b.pt", BF16,
                  "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"),
    "gemma3_4b": ("google/gemma-3-4b-pt", "acts_gemma3_4b.pt", BF16,
                  "lenses/gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt"),
}
NAMES = list(M)


def log(*a):
    print(*a, flush=True)


def surr_acts_path(name):
    return f"acts_surr_{name}.pt"


def extract(name, docs):
    """Surrogate activations, mirroring step6_acts.py (incl. dropping the
    vision tower on multimodal checkpoints for the text-only run)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, _, dtype, lens_path = M[name]
    out = surr_acts_path(name)
    if os.path.exists(out):
        log(f"    {out} cached")
        return
    lens = torch.load(lens_path, map_location="cpu", weights_only=False)
    layers = sorted(lens["J"].keys())
    band = [L for L in layers if 0.35 <= L / max(layers) <= 0.90]
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "vision_tower"):
        inner.vision_tower = None
        if hasattr(inner, "multi_modal_projector"):
            inner.multi_modal_projector = None
    model = model.to(DEV)
    feats = {L: [] for L in band}
    with torch.no_grad():
        for i, d in enumerate(docs):
            ids = tok(d, return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to(DEV)
            hs = model(**ids, output_hidden_states=True).hidden_states
            assert hs is not None, f"{name}: no hidden states"
            for L in band:
                feats[L].append(hs[L][0].mean(dim=0).float().cpu())
            if (i + 1) % 500 == 0:
                log(f"      {i+1}/{len(docs)}")
    torch.save({L: torch.stack(v) for L, v in feats.items()}, out)
    del model
    torch.cuda.empty_cache()
    log(f"    saved {out}")


def decompose(name, corpus):
    """Neighbor lists per component per layer, cached to disk."""
    cache = f"{NB_DIR}/{corpus}_{name}.pt"
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        return d["nbrs"], d["health"]
    hf, real_path, _, lens_path = M[name]
    acts = torch.load(real_path if corpus == "real" else surr_acts_path(name),
                      weights_only=False)
    lens = torch.load(lens_path, map_location="cpu", weights_only=False)
    WU, w = get_WU_and_w(hf)
    WUeff = WU.to(DEV) * w.to(DEV)
    del WU
    WUeff -= WUeff.mean(dim=0, keepdim=True)

    nbrs = {c: {} for c in COMPS}
    health = {}
    for L in sorted(acts.keys()):
        H = acts[L].to(DEV, torch.float32)
        q = torch.quantile(H.abs().flatten(), 0.95)
        H = H.clamp(-q, q)
        J = lens["J"][L].to(DEV, torch.float32)
        D = WUeff @ J
        D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
        HJ = nnomp_batch(H, D)
        perp = H - HJ
        # share of the pooled activation's squared norm the J component carries
        health[L] = dict(
            var_share=float((HJ.norm(dim=1) ** 2 / H.norm(dim=1) ** 2).mean()))
        for c, X in dict(full=H, J=HJ, perp=perp).items():
            nbrs[c][L] = neighbors(prep(X)).cpu()
        del H, J, D, HJ, perp
        torch.cuda.empty_cache()
    del WUeff
    torch.cuda.empty_cache()
    torch.save(dict(nbrs=nbrs, health=health), cache)
    return nbrs, health


def main():
    from datasets import load_dataset
    os.makedirs(NB_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log("=== corpus ===")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs = [d["text"][:1500] for d in ds.select(range(N_DOCS))]
    assert os.path.exists(SURR_TEXT), f"{SURR_TEXT} missing; run xsurrogate.py"
    surr = json.load(open(SURR_TEXT, encoding="utf-8"))
    assert len(surr) == len(docs) == N_DOCS
    log(f"  reusing {SURR_TEXT}")

    log("\n=== extracting surrogate activations (11 models) ===")
    for i, name in enumerate(NAMES):
        log(f"  [{i+1}/11] {name}")
        extract(name, surr)

    log("\n=== decomposing ===")
    health = {}
    for corpus in ("real", "surrogate"):
        for name in NAMES:
            _, h = decompose(name, corpus)
            health[f"{corpus}/{name}"] = h
            vs = float(np.mean([v["var_share"] for v in h.values()]))
            log(f"  [{corpus:>9}] {name:>10}  var_share {vs:.4f}  "
                f"layers {len(h)}")

    json.dump(dict(models=NAMES, health=health),
              open("results/surrogate_55.json", "w", encoding="utf-8"), indent=1)
    log("\nwrote results/surrogate_55.json")


if __name__ == "__main__":
    sys.exit(main())
