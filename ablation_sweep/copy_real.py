"""Copy the real-corpus alignment numbers in from the completed metric sweep.

The 8-metric sweep already scored every metric on the real corpus with the
J-lens dictionary, over the same 55 pairs, the same band x band grid and the
same preprocessing -- those per-pair values ARE the "real alignment" column of
the paper's Table 1. Rather than recompute them we copy them once into
real_reference.json, after which this folder reads nothing outside itself.

Read-only with respect to everything outside ablation_sweep/.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
SRC = os.path.join("..", "metric_sweep")

METRICS = ["mutual_knn", "cknna", "cka", "unbiased_cka",
           "cycle_knn", "edit_knn", "lcs_knn", "svcca"]
COMPS = ["full", "J", "perp"]


def main():
    out = {}
    missing = []
    for m in METRICS:
        p = os.path.join(SRC, m, "results.json")
        if not os.path.exists(p):
            missing.append(m)
            continue
        with open(p) as f:
            d = json.load(f)
        if not d.get("complete") or "text_pairs" not in d:
            missing.append(m)
            continue
        out[m] = {f"{r['a']}|{r['b']}": {c: r[c] for c in COMPS}
                  for r in d["text_pairs"]}
    if missing:
        print("MISSING real-corpus results for: " + ", ".join(missing),
              file=sys.stderr)
        return 1
    with open("real_reference.json", "w") as f:
        json.dump(dict(
            note="real-corpus, J-lens-dictionary alignment per pair, copied "
                 "verbatim from ../metric_sweep/<metric>/results.json "
                 "(text_pairs). Same 55 pairs, same band grid, same "
                 "preprocessing as this folder recomputes for the ablated "
                 "corpus and the Gaussian dictionary.",
            metrics=METRICS, comps=COMPS, pairs=out), f, indent=1)
    npairs = len(next(iter(out.values())))
    print(f"wrote real_reference.json: {len(out)} metrics x {npairs} pairs")
    # echo the paper's Table 1 real column as a sanity check
    mk = out["mutual_knn"]
    for c in COMPS:
        mean = sum(v[c] for v in mk.values()) / len(mk)
        print(f"  mutual_knn real {c:5s} = {mean:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
