"""Raw (uncorrected) mean depth grid for Figure 2(a): the 11x11 percentile grid
over all 55 text-model pairs, scored as the matched top-25 word overlap.

Companion to xw_meangrid.py, which scores the same grid as
    Delta = matched overlap - position-shuffled overlap.
Section 4.3 reports the matched overlap directly and confines the
position-shuffled floor to the controls subsection, so xfig_paper.py reads this
file rather than mean_grid.json.

Reads the top-25 word lists cached by xw_all.py.
Writes results/wordalign/mean_grid_raw.json.
"""
import json, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE, OUT = "cache/wordalign", "results/wordalign"
K = 25
PGRID = [i / 10 for i in range(11)]
NAMES = ["gpt2", "gemma", "gemma270", "pythia70m", "qwen08b", "qwen17b",
         "qwen2b", "gemma2_2b", "qwen4b", "qwen35_4b", "gemma3_4b"]
D = {n: torch.load(f"{CACHE}/{n}.pt", weights_only=False) for n in NAMES}
cache = {}
def g(t):
    if id(t) not in cache: cache[id(t)] = t[:, :K].to(DEV).long()
    return cache[id(t)]
def ov(Ta, Tb):
    a, b = g(Ta), g(Tb)
    return ((a.unsqueeze(2) == b.unsqueeze(1)).any(-1).sum(1).float() / K).mean().item()
def diag_layers(Ls):
    n = len(Ls)
    return [min(Ls, key=lambda L: (abs(L / (n - 1) - p), -L)) for p in PGRID]
pairs = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]]
G = []
for a, b in pairs:
    la, lb = diag_layers(D[a]["layers"]), diag_layers(D[b]["layers"])
    G.append([[ov(D[a]["J"][x], D[b]["J"][y]) for y in lb] for x in la])
    print(f"  {a} x {b}", flush=True)
    cache.clear()
json.dump({"pgrid": PGRID, "pairs": [f"{a}|{b}" for a, b in pairs], "J": G},
          open(f"{OUT}/mean_grid_raw.json", "w"))
print(f"wrote {OUT}/mean_grid_raw.json")
