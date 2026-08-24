"""Tokenisation control for section 3.1: how much text does each model see?

The 11 models do not share a vocabulary, so the fixed 300-token cap covers a
different amount of each document per model. This measures how often the cap
actually binds and how much of the corpus each model ingests, supporting the
paper's statement that tokeniser efficiency ranges from 3.79 to 4.19 characters
per token and that the least efficient tokeniser still ingests 94.8% as many
characters as the most efficient.

CPU only; no model weights are loaded, only tokenisers.
"""
import json

import numpy as np
from transformers import AutoTokenizer

from xsurrogate import MAX_TOKENS, N_DOCS

OUT = "results/token_control.json"
HF = {
    "pythia70m": "EleutherAI/pythia-70m-deduped", "gpt2": "gpt2",
    "gemma270": "google/gemma-3-270m", "qwen08b": "Qwen/Qwen3.5-0.8B",
    "gemma": "google/gemma-3-1b-pt", "qwen17b": "Qwen/Qwen3-1.7B",
    "qwen2b": "Qwen/Qwen3.5-2B-Base", "gemma2_2b": "google/gemma-2-2b",
    "qwen4b": "Qwen/Qwen3-4B", "qwen35_4b": "Qwen/Qwen3.5-4B",
    "gemma3_4b": "google/gemma-3-4b-pt",
}
ORDER = list(HF)


def corpus():
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [d["text"][:1500] for d in ds.select(range(N_DOCS))]


def main():
    docs = corpus()
    doc_chars = np.array([len(d) for d in docs], dtype=float)
    print(f"corpus: {len(docs)} docs, mean {doc_chars.mean():.0f} chars "
          f"(already truncated to 1500)\n")

    rec = {}
    for name in ORDER:
        tok = AutoTokenizer.from_pretrained(HF[name])
        n_tok, ingested = [], []
        for d in docs:
            ids = tok(d, add_special_tokens=False).input_ids
            n_tok.append(len(ids))
            if len(ids) > MAX_TOKENS:
                ingested.append(len(tok.decode(ids[:MAX_TOKENS],
                                               skip_special_tokens=True)))
            else:
                ingested.append(len(d))
        n_tok = np.array(n_tok, dtype=float)
        ingested = np.array(ingested, dtype=float)
        rec[name] = {
            "chars_per_token": float(doc_chars.sum() / n_tok.sum()),
            "frac_corpus_ingested": float(ingested.sum() / doc_chars.sum()),
        }
        r = rec[name]
        print(f"  {name:>11}  chars/tok {r['chars_per_token']:.2f}  "
              f"ingested {100*r['frac_corpus_ingested']:5.1f}% of corpus",
              flush=True)

    cpt = [rec[m]["chars_per_token"] for m in ORDER]
    ing = [rec[m]["frac_corpus_ingested"] for m in ORDER]
    print(f"\nchars/token spread : {min(cpt):.2f} to {max(cpt):.2f} "
          f"({100*(max(cpt)/min(cpt)-1):.1f}% range)")
    print(f"corpus ingested    : {100*min(ing):.1f}% to {100*max(ing):.1f}% "
          f"(least efficient sees {100*min(ing)/max(ing):.1f}% of what the "
          f"most efficient sees)")

    json.dump({"per_model": rec}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
