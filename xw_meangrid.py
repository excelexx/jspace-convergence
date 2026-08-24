"""Mean depth grid for Figure 2(a): the 11x11 percentile grid over all 55
text-model pairs.

Per-pair grids are recomputed from the top-25 word lists cached by xw_all.py,
because xw_stats.py stores only each grid's diagonal.

Writes results/wordalign/mean_grid.json; xfig_paper.py symmetrises the per-pair
grids, averages them and renders the result at print size as
paper/06_wordalign_mean_heatmap.pdf.
"""
import json, os
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE, OUT = "cache/wordalign", "results/wordalign"
SEED_SHUF, K = 777, 25
PGRID = [i / 10 for i in range(11)]
NAMES = ["gpt2", "gemma", "gemma270", "pythia70m", "qwen08b", "qwen17b",
         "qwen2b", "gemma2_2b", "qwen4b", "qwen35_4b", "gemma3_4b"]
MF = f"{OUT}/mean_grid.json"

if os.path.exists(MF):
    print(f"{MF} already built")
else:
    D = {n: torch.load(f"{CACHE}/{n}.pt", weights_only=False) for n in NAMES}
    n_pos = D[NAMES[0]]["J"][D[NAMES[0]]["layers"][0]].shape[0]
    perm = torch.randperm(n_pos, generator=torch.Generator().manual_seed(SEED_SHUF))
    cache = {}

    def g(t, sh=False):
        key = (id(t), sh)
        if key not in cache:
            cache[key] = (t[perm] if sh else t)[:, :K].to(DEV).long()
        return cache[key]

    def delta(Ta, Tb):
        a = g(Ta)
        out = []
        for sh in (False, True):
            b = g(Tb, sh)
            out.append(((a.unsqueeze(2) == b.unsqueeze(1)).any(-1).sum(1)
                        .float() / K).mean().item())
        return out[0] - out[1]

    def diag_layers(Ls):
        n = len(Ls)
        return [min(Ls, key=lambda L: (abs(L / (n - 1) - p), -L)) for p in PGRID]

    pairs = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]]
    G = []
    for a, b in pairs:
        la, lb = diag_layers(D[a]["layers"]), diag_layers(D[b]["layers"])
        G.append([[delta(D[a]["J"][x], D[b]["J"][y]) for y in lb] for x in la])
        print(f"  {a} x {b}", flush=True)
        cache.clear()
    json.dump({"pgrid": PGRID, "J": G}, open(MF, "w"))
    print(f"wrote {MF}")
