"""Lens-fitting control, step 0: lens provenance and the WikiText-103 split.

Feeds the control in Appendix C.1 ("Lens-fitting control", Figure
10_lens_fitting_control.pdf), which refits the J-lenses so that no two
compared models share a fitting corpus.

Two things happen here. Provenance: record how many prompts each in-scope
pilot lens was fitted on, which is what sets each half refit's document
budget in xlc_phase1.py. Corpus split: take the deduplicated WikiText-103
document list and split it in two by document with seed 0. xlc_phase1.py
refits every in-scope model on each half; those two halves are what make a
crossed-half pair possible.

Writes results/lenscontrol/half_manifest.json (read by xlc_phase1.py) and
run_manifest_lenscontrol.json.
"""
import json
import os

import torch

from crossmodal_utils import LENSFIT_SCOPE, load_pilot, wikitext_docs

pilot = load_pilot(dev="cpu")
os.makedirs("results/lenscontrol", exist_ok=True)

man = {"provenance": {}}
for name in LENSFIT_SCOPE:
    lens = torch.load(pilot.MODELS[name]["lens"], map_location="cpu",
                      weights_only=False)
    man["provenance"][name] = {"n_prompts": int(lens.get("n_prompts", -1))}

docs = wikitext_docs()
print(f"wikitext-103 train: {len(docs)} documents (heading-split, deduped)")

g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(docs), generator=g).tolist()
h1_idx, h2_idx = sorted(perm[: len(docs) // 2]), sorted(perm[len(docs) // 2:])

with open("results/lenscontrol/half_manifest.json", "w", encoding="utf-8") as f:
    json.dump({"seed": 0, "n_docs": len(docs),
               "h1_indices": h1_idx, "h2_indices": h2_idx}, f)
with open("run_manifest_lenscontrol.json", "w", encoding="utf-8") as f:
    json.dump(man, f, indent=2)
print("wrote run_manifest_lenscontrol.json")
