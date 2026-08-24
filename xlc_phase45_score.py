"""Lens-fitting control: is J-space alignment still above the null when the
two lenses are fitted on disjoint corpora?

Scores the cached decompositions from xlc_phase4.py. For each of the 10 model
pairs, J-component m-NN alignment is taken as the maximum over the layer-pair
grid under three fitting conditions -- shared-full (both lenses the downloaded
full-corpus lens), shared-half (both on the same half), crossed-half (one on
each half, averaged over the two crossing directions) -- and compared against
the R=19 random-dictionary draws, which are lens-independent and so identical
across conditions. The per-pair p-value is (1 + #draws >= observed) / (R + 1),
so 0.05 means the observed value beat every draw.

This is what backs the appendix sentence "All three alignment conditions
remain far above the random-dictionary null for every pair" and the null
crosses in the middle panel of Figure 10_lens_fitting_control.pdf. The paper's
+0.0002 is the median-over-grid difference and comes from
xlc_median_delta.py, not from here.

Writes results/lenscontrol/phase4_sparse.json; verify_paper_numbers.py reads
summary["pile"]["J_beats_null_crossed"].
"""
import json
from itertools import combinations

import torch

from crossmodal_utils import LENSFIT_SCOPE
from xkernels import grid_max

CORPUS = "pile"                                   # the 1,000 Pile eval docs
R_NULL = 19
S = {m: torch.load(f"results/lenscontrol/sparse/{m}.pt", weights_only=False)
     for m in LENSFIT_SCOPE}

out = {}
print(f"=== lens-fitting control, eval corpus = {CORPUS} ===")
for a, b in combinations(LENSFIT_SCOPE, 2):
    sh = [grid_max(S[a]["cond"][h], S[b]["cond"][h]) for h in ("h1", "h2")]
    ch = [grid_max(S[a]["cond"][x], S[b]["cond"][y])
          for x, y in (("h1", "h2"), ("h2", "h1"))]
    row = {"shared_full": grid_max(S[a]["cond"]["full"], S[b]["cond"]["full"]),
           "shared_half": (sh[0] + sh[1]) / 2,
           "crossed_half": (ch[0] + ch[1]) / 2}
    draws = [grid_max(S[a]["null"][r], S[b]["null"][r]) for r in range(R_NULL)]
    row["null_draws"] = draws
    for cond in ("shared_full", "shared_half", "crossed_half"):
        row[f"p_J_{cond}"] = (1 + sum(v >= row[cond]
                                      for v in draws)) / (R_NULL + 1)
    out[f"{CORPUS}|{a}|{b}"] = row
    print(f"  {a} x {b}: J full {row['shared_full']:.4f} | shared-half "
          f"{row['shared_half']:.4f} | crossed {row['crossed_half']:.4f} | "
          f"null max {max(draws):.4f} | p(crossed) "
          f"{row['p_J_crossed_half']:.3f}", flush=True)

rows = list(out.values())
COUNT_KEYS = {"shared_full": "J_beats_null_shared_full",
              "shared_half": "J_beats_null_shared_half",
              "crossed_half": "J_beats_null_crossed"}
summary = {CORPUS: {"n_pairs": len(rows)}}
for cond, key in COUNT_KEYS.items():
    summary[CORPUS][key] = sum(r[f"p_J_{cond}"] <= 0.05 for r in rows)
s = summary[CORPUS]
print(f"\nJ beats the random-dictionary null in "
      f"{s['J_beats_null_shared_full']}/{s['n_pairs']} shared-full, "
      f"{s['J_beats_null_shared_half']}/{s['n_pairs']} shared-half and "
      f"{s['J_beats_null_crossed']}/{s['n_pairs']} crossed-half pairs")

with open("results/lenscontrol/phase4_sparse.json", "w", encoding="utf-8") as f:
    json.dump({"pairs": out, "summary": summary}, f, indent=2)
print("\nwrote results/lenscontrol/phase4_sparse.json")
