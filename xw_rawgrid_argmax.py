"""Best-matching-counterpart-layer offset on the RAW top-25 overlap grid.

Mirrors xw_stats.py's rowargmax_dp exactly, except each grid cell is the
matched overlap ov() rather than the shuffle-corrected delta(). Section 4.3
quotes the mean over the 38 cross-family pairs (11.1%); xw_stats.py stores only
the delta version (11.0%), and only the grid diagonal in raw form, so the full
raw grid has to be rebuilt here.

Reads the top-25 word lists cached by xw_all.py.
Writes results/wordalign/rawgrid_argmax.json.
"""
import json, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE, OUT = "cache/wordalign", "results/wordalign"
PGRID = [i / 10 for i in range(11)]
NAMES = ["gpt2", "gemma", "gemma270", "pythia70m", "qwen08b", "qwen17b",
         "qwen2b", "gemma2_2b", "qwen4b", "qwen35_4b", "gemma3_4b"]
FAMILY = {"gemma270": "Gemma", "gemma": "Gemma", "gemma2_2b": "Gemma",
          "gemma3_4b": "Gemma", "qwen08b": "Qwen", "qwen17b": "Qwen",
          "qwen2b": "Qwen", "qwen4b": "Qwen", "qwen35_4b": "Qwen",
          "gpt2": "GPT-style", "pythia70m": "GPT-style"}
D = {n: torch.load(f"{CACHE}/{n}.pt", weights_only=False) for n in NAMES}
n_pos = D[NAMES[0]]["J"][D[NAMES[0]]["layers"][0]].shape[0]
print(f"n_pos={n_pos} dev={DEV}", flush=True)
_c = {}
def gpu(t, k):
    key = (id(t), k)
    if key not in _c: _c[key] = t[:, :k].to(DEV).long()
    return _c[key]
def ov(Ta, Tb, k):
    a, b = gpu(Ta, k), gpu(Tb, k)
    return ((a.unsqueeze(2) == b.unsqueeze(1)).any(-1).sum(1).float() / k).mean().item()
out = {}
pairs = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]]
for a, b in pairs:
    la = [min(D[a]["layers"], key=lambda L: (abs(L/(len(D[a]["layers"])-1)-p), -L)) for p in PGRID]
    lb = [min(D[b]["layers"], key=lambda L: (abs(L/(len(D[b]["layers"])-1)-p), -L)) for p in PGRID]
    P = [[ov(D[a]["J"][x], D[b]["J"][y], 25) for y in lb] for x in la]
    rows = [max(range(11), key=lambda j: P[i][j]) for i in range(11)]
    out[f"{a}|{b}"] = {
        "rowargmax_dp_raw": sum(abs(PGRID[i]-PGRID[j]) for i, j in enumerate(rows))/11,
        "same_family": FAMILY[a] == FAMILY[b],
    }
    print(f"  {a:>10} x {b:<10} dp_raw {out[f'{a}|{b}']['rowargmax_dp_raw']:.4f}", flush=True)
    _c.clear()
json.dump(out, open(f"{OUT}/rawgrid_argmax.json", "w"), indent=1)
print("wrote results/wordalign/rawgrid_argmax.json")
