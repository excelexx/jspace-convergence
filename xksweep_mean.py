"""Lens-degradation (instrument-only) control: appendix C.2's numbers.

Scores the degraded-J neighbour sets from xksweep.py at the MEAN over the
layer-pair grid and the MEAN over the 55 pairs, the aggregation Table 1 uses,
so the degraded alignment is directly comparable to the content-ablated one.

For each model, k* is the sparsity whose real-text variance share is closest to
that model's variance share on the ablated corpus at k = 25; alignment at those
k* is the matched instrument-only null (0.2924 in the appendix, against 0.1099
for content-ablated text). k = 5 gives the 2.1x figure quoted alongside it.

Neighbour sets are reused from cache/ksweep/, the ablated variance shares from
results/surrogate_55.json and the content-ablated alignment from
results/randdict_null_bylayerstat.json; nothing is refit.
"""
import json

import numpy as np
import torch

from xmedlayer import grid
from xsurrogate_all import NAMES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [5, 10, 15, 20, 25]
NB_DIR = "cache/ksweep"

nb, health = {}, {}
for name in NAMES:
    d = torch.load(f"{NB_DIR}/{name}.pt", weights_only=False)
    nb[name], health[name] = d["nbrs"], d["health"]

pairs = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]]

per_k = {k: [grid(nb[a][k], nb[b][k], DEV) for a, b in pairs] for k in KS}

surr_mean = float(np.mean(
    [p["real"]["J"]["surr_mean"] for p in
     json.load(open("results/randdict_null_bylayerstat.json",
                    encoding="utf-8"))["pairs"]]))

surr = json.load(open("results/surrogate_55.json", encoding="utf-8"))
surr_vs = {m: float(np.mean([v["var_share"]
                             for v in surr["health"]["surrogate/" + m].values()]))
           for m in surr["models"]}
kstar = {}
for m in NAMES:
    vs = {k: float(np.mean([v["var_share"] for v in health[m][k].values()]))
          for k in KS}
    kstar[m] = min(KS, key=lambda k: abs(vs[k] - surr_vs[m]))
print("k* per model:", kstar)

matched = [grid(nb[a][kstar[a]], nb[b][kstar[b]], DEV) for a, b in pairs]
mm = float(np.mean(matched))
print("\ndegraded (matched var share) alignment : %.4f" % mm)
print("content-ablated alignment              : %.4f" % surr_mean)
print("k=5 alignment                          : %.4f" % np.mean(per_k[5]))
print("k=5 / content-ablated ratio            : %.2fx"
      % (np.mean(per_k[5]) / surr_mean))

json.dump({"per_k_mean": {str(k): float(np.mean(per_k[k])) for k in KS},
           "matched_mean": mm, "surr_mean": surr_mean, "kstar": kstar},
          open("results/ksweep_mean.json", "w", encoding="utf-8"), indent=1)
print("\nwrote results/ksweep_mean.json")
