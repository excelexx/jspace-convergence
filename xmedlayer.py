"""Builds results/randdict_null_bylayerstat.json, the source file for Table 1.

Pairs the J-lens neighbour lists (cache/surr55, from xsurrogate_all.py) with the
Gaussian-dictionary ones (cache/randnull, from xrandnull.py) and scores every
one of the 55 model pairs on the real and the content-ablated corpus.

Each pair's alignment is reduced over its layer-pair grid by the MEAN, the
aggregation Table 1 and every competence correlation in the paper use.
Retention is the ablated value over the real value, per pair. Neighbour lists
are reused from cache; nothing is refit.
"""
import json
import numpy as np
import torch
from xsurrogate import K_NN, N_DOCS, mnn
from xrandnull import VARIANT, load_real
from xsurrogate_all import NAMES

DEV = "cuda"
OUT = "results/randdict_null_bylayerstat.json"

# the J-lens arm carries all three components; the Gaussian arm carries only J,
# which is the row Table 1 prints
COMPS = {"real": ["full", "J", "perp"], VARIANT: ["J"]}


def grid(nA, nB, dev=DEV):
    """Mean m-NN over the band x band layer-pair grid."""
    return float(np.mean([mnn(a.to(dev), b.to(dev))
                          for a in nA.values() for b in nB.values()]))


def main():
    nb = {"real": {}, "surrogate": {}}
    for corpus in ["real", "surrogate"]:
        for n in NAMES:
            try:
                d = torch.load(f"cache/randnull/{corpus}_{n}.pt",
                               weights_only=False)
                nb[corpus][n] = {"real": load_real(n, corpus), **d["nbrs"]}
            except Exception:
                pass
    names = sorted(set(nb["real"]) & set(nb["surrogate"]))
    assert names, ("no neighbour lists found; run xsurrogate_all.py and "
                   f"xrandnull.py first (refusing to overwrite {OUT})")
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rec = {"a": a, "b": b}
            for v, comps in COMPS.items():
                rec[v] = {}
                for c in comps:
                    rmn = grid(nb["real"][a][v][c], nb["real"][b][v][c])
                    smn = grid(nb["surrogate"][a][v][c],
                               nb["surrogate"][b][v][c])
                    rec[v][c] = dict(real_mean=rmn, surr_mean=smn,
                                     ret_mean=smn / rmn)
            rows.append(rec)
        print(f"  {a} done", flush=True)

    print(f"\n=== {len(rows)} model pairs (chance {K_NN/(N_DOCS-1):.4f}) ===")
    print("mean over grid, mean over pairs (Table 1)")
    print(f"{'dict':>5} {'comp':>6}" + " ".join(f"{h:>9}" for h in
                                                ("real", "surr", "ret")))
    for v, comps in COMPS.items():
        for c in comps:
            mn = [float(np.mean([x[v][c][f] for x in rows]))
                  for f in ("real_mean", "surr_mean", "ret_mean")]
            print(f"{v:>5} {c:>6}" + " ".join(f"{x:>9.4f}" for x in mn))
        print()
    json.dump(dict(n_pairs=len(rows), pairs=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
