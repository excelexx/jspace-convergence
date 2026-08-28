"""Score one CKNNA neighbourhood size over both of the paper's experiments.

Same contract as scorer.py -- identical decomposition, preprocessing, layer
band and grid-mean aggregation -- but the neighbourhood size comes from a
config dict, so run_sweep2.py can sweep it through one code path.

The per-layer `prep` (the cached Gram matrix) is computed once per layer, since
a layer appears in many pairs; scoring a pair is then cheap.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C

FEAT = C.FEATURES


# ------------------------------------------------------------- artifacts ---
def load_gram(side, key, dtype=torch.float64):
    a = np.load(os.path.join(FEAT, side, key + "_gram.npy"))
    return torch.tensor(a, device=C.DEV, dtype=dtype)


def prep_artifact(cfg, side, key):
    if cfg["kind"] != "cknna":
        raise ValueError("unknown kind " + cfg["kind"])
    return load_gram(side, key)


def score_pair(cfg, a, b):
    if cfg["kind"] != "cknna":
        raise ValueError("unknown kind " + cfg["kind"])
    # topk = n-1 is the largest admissible neighbourhood (the whole kernel)
    return C.cknna_from_gram(a, b, min(cfg["topk"], a.shape[0] - 1))


# ------------------------------------------------------------ experiments --
def run_text(cfg, manifest, log):
    layers = manifest["text_layers"]
    pairs = C.text_pairs()
    records = []
    t0 = time.time()
    cur_a, cache_a = None, None
    for a, b in pairs:
        if a != cur_a:
            cache_a = {(c, L): prep_artifact(cfg, "text", f"{a}_{c}_L{L}")
                       for c in C.COMPS for L in layers[a]}
            cur_a = a
            torch.cuda.empty_cache()
        cache_b = {(c, L): prep_artifact(cfg, "text", f"{b}_{c}_L{L}")
                   for c in C.COMPS for L in layers[b]}
        rec = dict(a=a, b=b)
        for c in C.COMPS:
            vals = [score_pair(cfg, cache_a[(c, La)], cache_b[(c, Lb)])
                    for La in layers[a] for Lb in layers[b]]
            rec[c] = float(np.mean(vals))
        records.append(rec)
        del cache_b
        torch.cuda.empty_cache()
        log(f"  text {a} vs {b}: " + " ".join(f"{c} {rec[c]:.4f}"
                                              for c in C.COMPS)
            + f"  [{time.time()-t0:.0f}s]")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True, help="JSON config dict")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_perm", type=int, default=200_000)
    args = ap.parse_args()
    cfg = json.loads(args.config)

    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            if json.load(f).get("complete"):
                print(f"{args.name}: already complete, skipping")
                return

    logf = open(os.path.join(out, "run.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    log(f"=== {args.name} | cfg {json.dumps(cfg)} ===")
    with open(os.path.join(FEAT, "manifest.json")) as f:
        manifest = json.load(f)

    import scorer as S
    tr = run_text(cfg, manifest, log)
    payload = dict(metric=args.name, config=cfg, n_perm=args.n_perm,
                   text_pairs=tr, text_stats=S.stats_text(tr, args.n_perm),
                   complete=True, seconds=time.time() - t0,
                   note="within-language alignment = mean of the metric over "
                        "the band x band layer-pair grid; paper-identical "
                        "decomposition and preprocessing, CKNNA "
                        "neighbourhood size varied")
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=1)
    log("text stats: " + json.dumps(
        {c: round(payload["text_stats"][c]["rho"], 3) for c in C.COMPS}))
    log(f"=== {args.name} done in {time.time()-t0:.0f}s ===")
    logf.close()


if __name__ == "__main__":
    main()
