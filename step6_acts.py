"""Cache the pooled activations experiments 1 and 2 are computed from.

For each of the 11 text models: the first 1,000 documents of pile-10k,
truncated to 1,500 characters then 300 tokens, mean-pooled over token positions
at every layer of the lens band (0.35 <= L/L_max <= 0.90) -> acts_{model}.pt.
The lens file is read only to learn which layers the band contains.

Resumable: a model whose acts_*.pt already exists is skipped.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

N_DOCS, MAX_TOKENS = 1000, 300

print("Loading corpus...")
ds = load_dataset("NeelNanda/pile-10k", split="train")
docs = [d["text"][:1500] for d in ds.select(range(N_DOCS))]

def cache(model_id, lens_path, out, dtype):
    print(f"\n=== {model_id} ===")
    if os.path.exists(out):
        print(f"{out} already cached, skipping")
        return
    lens = torch.load(lens_path, map_location="cpu", weights_only=False)
    layers = sorted(lens["J"].keys())
    band = [L for L in layers if 0.35 <= L / max(layers) <= 0.90]
    print("band layers:", band)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "vision_tower"):
        inner.vision_tower = None                # text-only run, save VRAM
        if hasattr(inner, "multi_modal_projector"):
            inner.multi_modal_projector = None
    model = model.to("cuda")

    feats = {L: [] for L in band}
    with torch.no_grad():
        for i, d in enumerate(docs):
            ids = tok(d, return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            hs = model(**ids, output_hidden_states=True).hidden_states
            assert hs is not None, f"{model_id}: model returned no hidden states"
            for L in band:
                feats[L].append(hs[L][0].mean(dim=0).float().cpu())
            if (i + 1) % 50 == 0: print(f"  {i+1}/{N_DOCS}")

    torch.save({L: torch.stack(v) for L, v in feats.items()}, out)
    print("saved", out)
    del model; torch.cuda.empty_cache()

cache("gpt2", "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
      "acts_gpt2.pt", torch.float16)
cache("google/gemma-3-1b-pt",
      "lenses/gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt",
      "acts_gemma.pt", torch.bfloat16)
cache("google/gemma-3-270m",
      "lenses/gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt",
      "acts_gemma270.pt", torch.bfloat16)
cache("EleutherAI/pythia-70m-deduped",
      "lenses/pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt",
      "acts_pythia70m.pt", torch.bfloat16)
cache("Qwen/Qwen3.5-0.8B",
      "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
      "acts_qwen08b.pt", torch.bfloat16)
cache("Qwen/Qwen3-1.7B",
      "lenses/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
      "acts_qwen17b.pt", torch.bfloat16)
cache("Qwen/Qwen3.5-2B-Base",
      "lenses/qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt",
      "acts_qwen2b.pt", torch.bfloat16)
cache("google/gemma-2-2b",
      "lenses/gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt",
      "acts_gemma2_2b.pt", torch.bfloat16)
cache("Qwen/Qwen3-4B",
      "lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
      "acts_qwen4b.pt", torch.bfloat16)
cache("Qwen/Qwen3.5-4B",
      "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
      "acts_qwen35_4b.pt", torch.bfloat16)
cache("google/gemma-3-4b-pt",
      "lenses/gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt",
      "acts_gemma3_4b.pt", torch.bfloat16)