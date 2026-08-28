"""Targeted rerun of the random-unembedding-row null (section 4.4) under the
paper's mean-over-grid aggregation, writing results/randunembed_null.json.

The J-space side is NOT recomputed: per-pair mean-over-grid J alignment is
read from results/randdict_null_bylayerstat.json (pairs[*].real.J.real_mean),
which was produced from the same acts_*.pt activations and the same pursuit.
Only the R_DRAWS = 5 seeded null decompositions are recomputed here, with the
identical seeds (1000 + r), row subsets (min(V, 20000)), unembedding
preprocessing, clamp, pursuit, and neighbour extraction as step7_align.py
(whose helpers are reused verbatim via crossmodal_utils.load_pilot()).
"""
import json
import time

import torch

from crossmodal_utils import load_pilot

P = load_pilot()
R_DRAWS = 5

t0 = time.time()
nbrs = {}
n_docs = None
for name, cfg in P.MODELS.items():
    print(f"\n=== null decompositions: {name} ===", flush=True)
    acts = torch.load(cfg["acts"], weights_only=False)
    n0 = next(iter(acts.values())).shape[0]
    assert n_docs in (None, n0), f"doc-count mismatch for {name}"
    n_docs = n0
    WU, w = P.get_WU_and_w(cfg["hf"])
    WUeff = WU.to(P.DEV) * w.to(P.DEV)
    del WU
    WUeff -= WUeff.mean(dim=0, keepdim=True)
    Vsz = WUeff.shape[0]
    Drs = []
    for r in range(R_DRAWS):
        g = torch.Generator().manual_seed(1000 + r)
        rand_rows = torch.randperm(Vsz, generator=g)[: min(Vsz, 20000)]
        Dr = WUeff[rand_rows]
        Drs.append(Dr / Dr.norm(dim=1, keepdim=True).clamp(min=1e-8))
    del WUeff
    torch.cuda.empty_cache()

    nbrs[name] = [{} for _ in range(R_DRAWS)]
    for L in sorted(acts.keys()):
        H = acts[L].to(P.DEV, torch.float32)
        q = torch.quantile(H.abs().flatten(), 0.95)
        H = H.clamp(-q, q)
        for r, Dr in enumerate(Drs):
            HJr = P.nnomp_batch(H, Dr)
            nbrs[name][r][L] = P.neighbors(P.prep(HJr)).cpu()
        print(f"  L{L:>2} done ({time.time() - t0:.0f}s elapsed)", flush=True)
    del Drs
    torch.cuda.empty_cache()

CHANCE = P.K_NN / (n_docs - 1)

def mnn_grid_mean(na, nb):
    vals = [P.mnn(na[a], nb[b]) for a in sorted(na) for b in sorted(nb)]
    return sum(vals) / len(vals)

rd = json.load(open("results/randdict_null_bylayerstat.json"))
j_mean = {frozenset((p["a"], p["b"])): p["real"]["J"]["real_mean"]
          for p in rd["pairs"]}

names = list(P.MODELS)
pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
records = []
for a, b in pairs:
    draws = [mnn_grid_mean(nbrs[a][r], nbrs[b][r]) for r in range(R_DRAWS)]
    if not (max(draws) < 1.0 and min(draws) > CHANCE / 2):
        print(f"  WARNING: null draws look degenerate for {a} vs {b}: {draws}")
    jm = j_mean[frozenset((a, b))]
    n_beat = sum(jm > v for v in draws)
    records.append(dict(
        a=a, b=b, J_mean=round(jm, 6),
        null_means=[round(v, 6) for v in draws], n_beat=n_beat,
        margin_over_strongest=round(jm - max(draws), 6)))
    print(f"{a} vs {b}: J mean {jm:.4f}, null means "
          + " ".join(f"{v:.4f}" for v in draws)
          + f", beats {n_beat}/{R_DRAWS}, margin {jm - max(draws):+.4f}",
          flush=True)

agg = dict(
    n_pairs=len(records),
    pairs_beating_all_draws=sum(r["n_beat"] == R_DRAWS for r in records),
    mean_margin_over_strongest=round(
        sum(r["margin_over_strongest"] for r in records) / len(records), 6),
    min_margin_over_strongest=round(
        min(r["margin_over_strongest"] for r in records), 6))
with open("results/randunembed_null.json", "w") as f:
    json.dump(dict(
        note="random-unembedding-row null (section 4.4), mean-over-grid "
             "aggregation; R_DRAWS seeded draws of min(V, 20000) rows. "
             "J side from randdict_null_bylayerstat.json (same acts, same "
             "pursuit); null side recomputed by xrandunembed_rerun.py",
        chance=CHANCE, r_draws=R_DRAWS, aggregate=agg, pairs=records,
    ), f, indent=1)
print(f"\nsummary: {agg['pairs_beating_all_draws']}/{agg['n_pairs']} pairs "
      f"beat all {R_DRAWS} draws, mean margin "
      f"{agg['mean_margin_over_strongest']:+.4f}, min margin "
      f"{agg['min_margin_over_strongest']:+.4f} "
      f"({time.time() - t0:.0f}s total)")
