"""Score one metric for the content-ablation and random-dictionary controls.

Reproduces the paper's Table 1 under an arbitrary similarity metric:

  full   real vs content-ablated, J-lens dictionary
  J      real vs content-ablated, J-lens dictionary
  non-J  real vs content-ablated, J-lens dictionary
  Gauss  real vs content-ablated, Gaussian dictionary (J component only)

Real-corpus J-lens alignment is read from real_reference.json (copied verbatim
from the completed 8-metric sweep and validated against a fresh recomputation);
everything else is scored from this folder's own Gram cache.

Retention is the mean of per-pair ratios ablated/real, matching the paper's
footnote, not the ratio of means.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import common as C

FEAT = "_features"
KNN_METRICS = {"mutual_knn", "cycle_knn", "lcs_knn", "edit_knn"}
GRAM_METRICS = {"cka", "unbiased_cka", "cknna"}
SVCCA_METRICS = {"svcca"}
ALL_METRICS = sorted(KNN_METRICS | GRAM_METRICS | SVCCA_METRICS)

# (row label, corpus->component) for the four rows of the table
ROWS = [("full", "full"), ("non-J", "perp"), ("J", "J"), ("Gaussian", "gaussJ")]
REAL_FROM_COPY = {"full", "J", "perp"}      # rows whose real column is copied


def kind(metric):
    if metric in KNN_METRICS:
        return "knn"
    if metric in GRAM_METRICS:
        return "gram"
    return "svcca"


def load_artifact(metric, corpus, key):
    k = kind(metric)
    if k == "svcca":
        return np.load(os.path.join(FEAT, corpus, key + "_u.npy"))
    g = np.load(os.path.join(FEAT, corpus, key + "_gram.npy"))
    K = torch.tensor(g, device=C.DEV,
                     dtype=torch.float32 if k == "knn" else torch.float64)
    if k == "knn":
        return C.knn_from_gram(K, C.TOPK)
    return K


class Scorer:
    def __init__(self, metric):
        self.metric = metric
        self._self = {}

    def self_term(self, tag, K):
        if tag in self._self:
            return self._self[tag]
        if self.metric == "cka":
            v = C.hsic_biased(K, K)
        elif self.metric == "unbiased_cka":
            v = C.hsic_unbiased(K, K)
        else:                                            # cknna
            n = K.shape[0]
            Kh = K.clone().fill_diagonal_(float("-inf"))
            ia = torch.topk(Kh, C.TOPK, dim=1).indices
            m = torch.zeros(n, n, device=K.device, dtype=K.dtype).scatter_(1, ia, 1)
            v = C.hsic_unbiased(m * K, m * K)
        self._self[tag] = v
        return v

    def score(self, a, b, ta=None, tb=None):
        m = self.metric
        if m == "mutual_knn":
            return C.mutual_knn(a, b)
        if m == "cycle_knn":
            return C.cycle_knn(a, b)
        if m == "lcs_knn":
            return C.lcs_knn(a, b)
        if m == "edit_knn":
            return C.edit_knn(a, b)
        if m == "svcca":
            return C.svcca_from_basis(a, b)
        sa, sb = self.self_term(ta, a), self.self_term(tb, b)
        if m == "cka":
            num = C.hsic_biased(a, b)
        elif m == "unbiased_cka":
            num = C.hsic_unbiased(a, b)
        else:
            n = a.shape[0]
            ah = a.clone().fill_diagonal_(float("-inf"))
            bh = b.clone().fill_diagonal_(float("-inf"))
            ia = torch.topk(ah, C.TOPK, dim=1).indices
            ib = torch.topk(bh, C.TOPK, dim=1).indices
            ma = torch.zeros(n, n, device=a.device, dtype=a.dtype).scatter_(1, ia, 1)
            mb = torch.zeros(n, n, device=a.device, dtype=a.dtype).scatter_(1, ib, 1)
            mk = ma * mb
            num = C.hsic_unbiased(mk * a, mk * b)
        return (num / (torch.sqrt(sa * sb) + 1e-6)).item()


def run(metric, need, layers, log):
    """need: list of (corpus, component) to compute. Returns {(c,comp): {pair: v}}."""
    sc = Scorer(metric)
    out = {k: {} for k in need}
    pairs = C.text_pairs()
    t0 = time.time()
    cur_a, cache_a = None, None
    for a, b in pairs:
        if a != cur_a:
            cache_a = {(c, comp, L): load_artifact(metric, c, f"{a}_{comp}_L{L}")
                       for c, comp in need for L in layers[a]}
            cur_a = a
            if kind(metric) == "gram":
                torch.cuda.empty_cache()
        cache_b = {(c, comp, L): load_artifact(metric, c, f"{b}_{comp}_L{L}")
                   for c, comp in need for L in layers[b]}
        for c, comp in need:
            vals = [sc.score(cache_a[(c, comp, La)], cache_b[(c, comp, Lb)],
                             f"{c}|{a}|{comp}|{La}", f"{c}|{b}|{comp}|{Lb}")
                    for La in layers[a] for Lb in layers[b]]
            out[(c, comp)][f"{a}|{b}"] = float(np.mean(vals))
        del cache_b
        if kind(metric) == "gram":
            torch.cuda.empty_cache()
        log(f"  {a} vs {b}  " + "  ".join(
            f"{c}/{comp} {out[(c, comp)][f'{a}|{b}']:.4f}" for c, comp in need)
            + f"  [{time.time()-t0:.0f}s]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=ALL_METRICS)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)
    res = os.path.join(out, "results.json")
    if os.path.exists(res):
        with open(res) as f:
            if json.load(f).get("complete"):
                print(f"{args.metric}: already complete, skipping")
                return

    logf = open(os.path.join(out, "run.log"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    with open(os.path.join(FEAT, "manifest.json")) as f:
        layers = json.load(f)["layers"]
    with open("real_reference.json") as f:
        ref = json.load(f)["pairs"][args.metric]

    t0 = time.time()
    log(f"=== ablation sweep: {args.metric} ===")
    need = [("abl", "full"), ("abl", "perp"), ("abl", "J"),
            ("real", "gaussJ"), ("abl", "gaussJ")]
    got = run(args.metric, need, layers, log)

    # assemble real/ablated/retention per row
    table = {}
    for label, comp in ROWS:
        if comp in REAL_FROM_COPY:
            real = {p: v[comp] for p, v in ref.items()}
        else:
            real = got[("real", comp)]
        abl = got[("abl", comp)]
        pairs = sorted(real)
        ratios = []
        for p in pairs:
            r = real[p]
            ratios.append(abl[p] / r if abs(r) > 1e-12 else float("nan"))
        finite = [x for x in ratios if np.isfinite(x)]
        table[label] = dict(
            real=float(np.mean([real[p] for p in pairs])),
            ablated=float(np.mean([abl[p] for p in pairs])),
            retention=float(np.mean(finite)) if finite else float("nan"),
            retention_n=len(finite),
            per_pair={p: dict(real=real[p], abl=abl[p]) for p in pairs})

    payload = dict(metric=args.metric, complete=True,
                   seconds=time.time() - t0, table=table,
                   note="real column for full/J/non-J copied from the "
                        "8-metric sweep (validated); ablated and Gaussian "
                        "columns computed here. Retention is the mean of "
                        "per-pair ratios, as in the paper.")
    with open(res, "w") as f:
        json.dump(payload, f, indent=1)
    for label, _ in ROWS:
        t = table[label]
        log(f"  {label:9s} real {t['real']:.4f}  abl {t['ablated']:.4f}  "
            f"retention {100*t['retention']:.1f}%")
    log(f"=== {args.metric} done in {time.time()-t0:.0f}s ===")
    logf.close()


if __name__ == "__main__":
    main()
