"""Lens-fitting control, step 1: refit each in-scope lens on one corpus half.

For every (model, band layer, half) this accumulates the averaged Jacobian in
fp64 with the project's causal JVP estimator over that half of WikiText-103,
using documents of at least T=300 tokens, each truncated to T (shorter ones
are skipped). Accumulators are checkpointed, so a run resumes where it stopped.
Before anything accumulates, the suffix wiring gate (xlc_suffix.py) is
asserted for every layer in the band: replaying blocks L..end on the captured
hidden state must reproduce the model's own pre-final-norm state.

Scope is the five models Appendix C.1 covers -- Pythia-70M, GPT-2,
Gemma-3-270M, Qwen3.5-0.8B, Gemma-3-1B -- which give the 10 pairs the paper
reports.

Outputs: results/lenscontrol/jfit/{model}_L{L}_{half}.pt, consumed by
xlc_phase2.py (stability gate) and xlc_phase4.py (decomposition).
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crossmodal_jacobian import JAccumulator, estimate_J_item
from crossmodal_utils import LENSFIT_SCOPE, load_pilot, wikitext_docs
from xlc_suffix import TextSuffixFactory

T_FIT = 300
pilot = load_pilot(dev="cpu")
stage1 = json.load(open("cache/text_acts/layers.json", encoding="utf-8"))
prov = json.load(open("run_manifest_lenscontrol.json",
                      encoding="utf-8"))["provenance"]
half_man = json.load(open("results/lenscontrol/half_manifest.json",
                          encoding="utf-8"))
os.makedirs("results/lenscontrol/jfit", exist_ok=True)

docs = wikitext_docs()
assert len(docs) == half_man["n_docs"], "doc list drifted from phase-0 manifest"
halves = {"h1": [docs[i] for i in half_man["h1_indices"]],
          "h2": [docs[i] for i in half_man["h2_indices"]]}


def fit_model(name):
    cfg = pilot.MODELS[name]
    band = stage1[name]["band_layers"]
    n_half = prov[name]["n_prompts"] // 2
    done_marker = f"results/lenscontrol/jfit/{name}_DONE"
    if os.path.exists(done_marker):
        print(f"{name}: complete, skipping")
        return
    print(f"=== {name}: band {band}, {n_half} docs/half ===", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf"], dtype=torch.float32, attn_implementation="eager")
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    fac = None
    for half, pool in halves.items():
        accs = {L: JAccumulator(
            d, f"results/lenscontrol/jfit/{name}_L{L}_{half}.pt")
            for L in band}
        target = n_half
        done = min(a.count for a in accs.values())
        if done >= target:
            print(f"  {half}: complete ({done})", flush=True)
            continue
        used = -1
        for doc in pool:
            ids = tok(doc, return_tensors="pt", truncation=True,
                      max_length=T_FIT).to("cuda")
            if ids["input_ids"].shape[1] < T_FIT:
                continue
            used += 1
            if used < done:
                continue                          # resume: skip already-done
            if fac is None:
                fac = TextSuffixFactory(model, ids)
                for L in band:
                    rel = fac.wiring_gate(L)
                    assert rel < 1e-4, f"{name} L{L} wiring {rel}"
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True,
                           use_cache=False).hidden_states
            for L in band:
                if accs[L].count > done:          # uneven counts after a crash
                    continue
                J = estimate_J_item(fac.suffix(L), hs[L][0].float(), chunk=16)
                accs[L].add(J)
            done += 1
            if done % 16 == 0:
                print(f"  {half} {done}/{target}", flush=True)
            if done >= target:
                break
        assert done >= target, f"{name} {half}: only {done} qualifying docs"
        for a in accs.values():
            a.save()
        print(f"  {half}: done ({done} docs)", flush=True)
    open(done_marker, "w").close()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for name in LENSFIT_SCOPE:                    # smallest first
        fit_model(name)
    print("phase 1 complete")
