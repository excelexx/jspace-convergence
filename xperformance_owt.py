"""Language-model performance as 1 - bits-per-byte over 4M tokens of
OpenWebText (Gokaslan & Cohen, 2019), following the convention used by the
Platonic Representation Hypothesis.

This is the third competence axis: the `1-bits-per-byte` column of
tab:crossaxes, and the within-language J correlation of rho = +0.35 reported
alongside it. Results go to results/performance_owt.json, read by xmeanmain.py
and verify_paper_numbers.py.

The document sample is FIXED (cache/owt/sample_4M.json, ~4M GPT-2 tokens,
17.7MB) so every model is scored on identical bytes.

Scoring: the sample is concatenated once, tokenised per model, and split into
windows of CTX tokens. Within each window the first token is context only; NLL
is summed over the remaining tokens and the denominator counts exactly the
bytes those scored tokens decode to. Idempotent per model.
"""
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results/performance_owt.json"
SAMPLE = "cache/owt/sample_4M.json"
CTX = 1024
# bf16 logit elements per forward pass; keeps CTX x vocab x batch bounded
LOGIT_BUDGET = 120_000_000
# positions scored per cross-entropy call (bounds the float32 upcast)
CE_CHUNK = 128

# hf id, dtype, windows per forward pass (3080, 10GB)
MODELS = {
    "pythia70m": ("EleutherAI/pythia-70m-deduped", torch.float16, 8),
    "gpt2":      ("gpt2", torch.float16, 8),
    "gemma270":  ("google/gemma-3-270m", torch.bfloat16, 8),
    "qwen08b":   ("Qwen/Qwen3.5-0.8B", torch.bfloat16, 4),
    "gemma":     ("google/gemma-3-1b-pt", torch.bfloat16, 4),
    "qwen17b":   ("Qwen/Qwen3-1.7B", torch.bfloat16, 2),
    "qwen2b":    ("Qwen/Qwen3.5-2B-Base", torch.bfloat16, 2),
    "gemma2_2b": ("google/gemma-2-2b", torch.bfloat16, 2),
    "qwen4b":    ("Qwen/Qwen3-4B", torch.bfloat16, 1),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", torch.bfloat16, 1),
    "gemma3_4b": ("google/gemma-3-4b-pt", torch.bfloat16, 1),
}


def load_text():
    files = json.load(open(SAMPLE, encoding="utf-8"))
    docs = [open(f, encoding="utf-8", errors="replace").read() for f in files]
    return "\n\n".join(docs), len(docs)


def score(name, hf, dtype, bs, text):
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "vision_tower"):
        inner.vision_tower = None          # text-only, as elsewhere in the repo
        if hasattr(inner, "multi_modal_projector"):
            inner.multi_modal_projector = None
    model = model.to(DEV).eval()

    ids = tok(text, return_tensors="pt", verbose=False,
              add_special_tokens=False).input_ids[0]
    n_full = ids.numel()
    # Each window carries its own BOS; chunking a single tokenisation would
    # give BOS to window 0 only.
    body = CTX - 1
    n_win = n_full // body
    wins = ids[:n_win * body].view(n_win, body)
    bos = tok.bos_token_id
    if bos is None:
        bos = tok.eos_token_id
    print(f"    bos token {bos}, {body} scored tokens/window", flush=True)

    # Logit memory scales with vocab, not parameters: Gemma/Qwen carry 248k-262k
    # vocabularies against GPT-2's 50k, so size the batch from CTX x vocab and
    # cap the requested batch accordingly.
    vocab = int(getattr(model.config, "vocab_size", 0)) or len(tok)
    bs = max(1, min(bs, LOGIT_BUDGET // (CTX * vocab)))
    print(f"    vocab {vocab:,} -> {bs} window(s)/forward", flush=True)

    nll_sum = 0.0
    n_bytes = 0
    with torch.no_grad():
        for i in range(0, n_win, bs):
            body_ids = wins[i:i + bs].to(DEV)
            pre = torch.full((body_ids.shape[0], 1), bos,
                             dtype=body_ids.dtype, device=DEV)
            batch = torch.cat([pre, body_ids], dim=1)   # [b, CTX]
            logits = model(batch).logits
            tgt = batch[:, 1:]                          # the real tokens
            # upcast and score in position slices so the full float32 logit
            # tensor is never materialised
            for a in range(0, CTX - 1, CE_CHUNK):
                b = min(a + CE_CHUNK, CTX - 1)
                nll_sum += torch.nn.functional.cross_entropy(
                    logits[:, a:b].float().reshape(-1, logits.shape[-1]),
                    tgt[:, a:b].reshape(-1), reduction="sum").item()
            del logits
            for row in body_ids:
                n_bytes += len(tok.decode(row, skip_special_tokens=True)
                               .encode("utf-8"))
            if (i // max(bs, 1)) % 200 == 0:
                print(f"      window {i}/{n_win}", flush=True)

    del model
    torch.cuda.empty_cache()
    bpb = nll_sum / (n_bytes * math.log(2))
    return {"model": name, "hf": hf, "performance": 1.0 - bpb}


def main():
    os.makedirs("results", exist_ok=True)
    rec = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    todo = [m for m in MODELS if m not in rec]
    if todo:
        text, ndoc = load_text()
        print(f"sample: {ndoc} OpenWebText docs, "
              f"{len(text.encode('utf-8')):,} bytes", flush=True)
        for name in todo:
            hf, dtype, bs = MODELS[name]
            print(f"=== {name} ({hf}, {bs} win/fwd) ===", flush=True)
            rec[name] = score(name, hf, dtype, bs, text)
            print(f"  performance {rec[name]['performance']:.4f}", flush=True)
            json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
