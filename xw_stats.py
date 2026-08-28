"""Experiment 3 (paper section 4.3), statistics stage: per-pair agreement over
the top-25 word lists cached by xw_all.py.

Headline object per pair = the matched-depth diagonal of the 11x11 PERCENTILE
grid (each model's layer nearest p = L/(N-1) for p in 0.0..1.0), scored as the
raw overlap
    mean_x |top-25_a(x) ^ top-25_b(x)| / 25
for the J-lens and for the plain logit lens, plus the same J-lens quantity with
positions shuffled, which is the floor the controls paragraph reports.

Section 4.3 reports the raw overlap directly and confines the position-shuffled
floor to the controls subsection, so no shuffle correction is applied here.

Writes results/wordalign/stats.json, read by verify_paper_numbers.py,
xfig_paper.py (Figure 3(b)) and xplot_wordalign_pairs.py (Figure 4).
"""
import json

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE, OUT = "cache/wordalign", "results/wordalign"
SEED_SHUF = 777
PGRID = [i / 10 for i in range(11)]
NAMES = ["gpt2", "gemma", "gemma270", "pythia70m", "qwen08b", "qwen17b",
         "qwen2b", "gemma2_2b", "qwen4b", "qwen35_4b", "gemma3_4b"]
FAMILY = {"gemma270": "Gemma", "gemma": "Gemma", "gemma2_2b": "Gemma",
          "gemma3_4b": "Gemma", "qwen08b": "Qwen", "qwen17b": "Qwen",
          "qwen2b": "Qwen", "qwen4b": "Qwen", "qwen35_4b": "Qwen",
          "gpt2": "GPT-style", "pythia70m": "GPT-style"}

D = {n: torch.load(f"{CACHE}/{n}.pt", weights_only=False) for n in NAMES}
nA = len(json.load(open(f"{OUT}/anchors.json"))["strings"])
n_pos = D[NAMES[0]]["J"][D[NAMES[0]]["layers"][0]].shape[0]
perm = torch.randperm(n_pos, generator=torch.Generator().manual_seed(SEED_SHUF))
print(f"{len(NAMES)} models, n_pos={n_pos}, n_A={nA}")

_cache = {}


def gpu(t, k, sh=False):
    key = (id(t), k, sh)
    if key not in _cache:
        x = t[perm] if sh else t
        _cache[key] = x[:, :k].to(DEV).long()
    return _cache[key]


def ov(Ta, Tb, k, sh=False):
    a, b = gpu(Ta, k), gpu(Tb, k, sh)
    return ((a.unsqueeze(2) == b.unsqueeze(1)).any(-1).sum(1).float() / k).mean().item()


pairs = [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1:]]
R = {}
for a, b in pairs:
    la = [min(D[a]["layers"], key=lambda L: (abs(L / (len(D[a]["layers"]) - 1) - p), -L))
          for p in PGRID]
    lb = [min(D[b]["layers"], key=lambda L: (abs(L / (len(D[b]["layers"]) - 1) - p), -L))
          for p in PGRID]
    e = {"la": la, "lb": lb, "same_family": FAMILY[a] == FAMILY[b]}
    # both arms are read only at matched depth
    for comp in ("J", "base"):
        e[f"raw_{comp}_k25"] = [ov(D[a][comp][la[i]], D[b][comp][lb[i]], 25)
                                for i in range(11)]
    e["shuf_J_k25"] = [ov(D[a]["J"][la[i]], D[b]["J"][lb[i]], 25, True)
                       for i in range(11)]
    R[(a, b)] = e
    print(f"  {a:>10} x {b:<10} median raw J {sorted(e['raw_J_k25'])[5]:.3f}"
          f"  base {sorted(e['raw_base_k25'])[5]:.3f}", flush=True)
_cache.clear()

res = {"n_pos": n_pos, "n_A": nA, "pgrid": PGRID, "pairs": {}}
for (a, b), e in R.items():
    res["pairs"][f"{a}|{b}"] = {
        **{k: e[k] for k in e if k.startswith(("raw_", "shuf_"))},
        "same_family": e["same_family"],
    }

# Carried so the section 4.3 competence figure (xplot_wordalign_pairs.py) can
# read model order and HellaSwag from one file. The correlation itself is
# computed there, at the pair level with a model-label permutation p-value.
res["per_model"] = {
    n: {"hellaswag": json.load(open(f"results/lmeval/{n}.json"))
        ["hellaswag_acc_norm"]} for n in NAMES}

json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1)
print(f"wrote {OUT}/stats.json")
