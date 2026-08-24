"""Lens-fitting control: the paper's +0.0002.

For each of the 10 model pairs, this takes the median over the layer-pair grid
of J-component m-NN alignment in the shared-half condition (both lenses fitted
on the same half of WikiText-103) and in the crossed-half condition (fitted on
disjoint halves, averaged over the two crossing directions), and reports the
per-pair difference Delta = shared-half - crossed-half.

The median of those ten differences is the +0.0002 quoted in Appendix C.1, and
the median shared-half level is the 0.42 it is quoted against.
verify_paper_numbers.py reads summary["pile"].

Everything is recomputed from the neighbour-set cache xlc_phase4.py wrote in
results/lenscontrol/sparse/.
"""
import json
from itertools import combinations

import numpy as np
import torch

from crossmodal_utils import LENSFIT_SCOPE
from xkernels import grid_median

CORPUS = "pile"                                   # the 1,000 Pile eval docs
OUT = "results/lenscontrol/median_delta.json"
S = {m: torch.load(f"results/lenscontrol/sparse/{m}.pt", weights_only=False)
     for m in LENSFIT_SCOPE}

out = {}
print(f"=== lens-fitting control, eval corpus = {CORPUS} ===")
for a, b in combinations(LENSFIT_SCOPE, 2):
    s1, s2 = (grid_median(S[a]["cond"][h], S[b]["cond"][h])
              for h in ("h1", "h2"))
    c1, c2 = (grid_median(S[a]["cond"][x], S[b]["cond"][y])
              for x, y in (("h1", "h2"), ("h2", "h1")))
    sh, ch = (s1 + s2) / 2, (c1 + c2) / 2
    out[f"{CORPUS}|{a}|{b}"] = {"J": {"shared_half_med": sh,
                                      "crossed_half_med": ch,
                                      "delta_med": sh - ch}}
    print(f"  {a:>10} x {b:<10} J: median {sh:.4f} vs {ch:.4f}  "
          f"d_med {sh - ch:+.4f}", flush=True)

rows = [v["J"] for v in out.values()]
dm = [r["delta_med"] for r in rows]
summary = {CORPUS: {
    "n_pairs": len(dm),
    "median_delta_med": float(np.median(dm)),
    "median_alignment_level_med": float(
        np.median([r["shared_half_med"] for r in rows]))}}
s = summary[CORPUS]
print(f"\nmedian Delta {s['median_delta_med']:+.6f} against an alignment "
      f"level of {s['median_alignment_level_med']:.4f}")

json.dump({"summary": summary, "pairs": out},
          open(OUT, "w", encoding="utf-8"), indent=1)
print(f"\nwrote {OUT}")
