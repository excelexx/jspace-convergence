"""Score one alignment metric over the paper's two experiments.

Text-text  : all 55 model pairs, three components, alignment = mean of the
             metric over the band x band layer-pair grid, then Spearman
             against the pair's mean HellaSwag with a model-label permutation
             null (the paper's Fig. 1a / Table 3 setup).
Text-vision: all 44 (text model, encoder) pairs, caption components against
             raw pooled image features, alignment = mean over the 5 x 6 grid.
             Reported both per encoder (the paper's Table 2) and averaged over
             the four encoders to give one point per text model (Table 4).

Only the metric changes; every upstream step is the paper's. Results are
written to <outdir>/results.json and the run is idempotent.
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

# which cached artifact each metric consumes
KNN_METRICS = {"mutual_knn", "cycle_knn", "lcs_knn", "edit_knn"}
GRAM_METRICS = {"cka", "unbiased_cka", "cknna"}
SVCCA_METRICS = {"svcca"}
ALL_METRICS = sorted(KNN_METRICS | GRAM_METRICS | SVCCA_METRICS)


def gram_path(side, key):
    return os.path.join(C.FEATURES, side, key + "_gram.npy")


def u_path(side, key):
    return os.path.join(C.FEATURES, side, key + "_u.npy")


def load_gram(side, key, dtype=torch.float64):
    a = np.load(gram_path(side, key))
    return torch.tensor(a, device=C.DEV, dtype=dtype)


def load_knn(side, key):
    K = load_gram(side, key, torch.float32)
    return C.knn_from_gram(K, C.TOPK)


def load_u(side, key):
    return np.load(u_path(side, key))


class Scorer:
    """Wraps one metric, caching whatever per-layer terms it can reuse."""

    def __init__(self, metric):
        self.metric = metric
        self._self_cache = {}

    def kind(self):
        if self.metric in KNN_METRICS:
            return "knn"
        if self.metric in GRAM_METRICS:
            return "gram"
        return "svcca"

    def self_term(self, tag, K):
        """f(K,K) for CKA, or sim(K,K) for CKNNA -- depends only on K."""
        if tag in self._self_cache:
            return self._self_cache[tag]
        if self.metric == "cka":
            v = C.hsic_biased(K, K)
        elif self.metric == "unbiased_cka":
            v = C.hsic_unbiased(K, K)
        else:                                       # cknna
            n = K.shape[0]
            Kh = K.clone().fill_diagonal_(float("-inf"))
            ia = torch.topk(Kh, C.TOPK, dim=1).indices
            m = torch.zeros(n, n, device=K.device, dtype=K.dtype).scatter_(1, ia, 1)
            v = C.hsic_unbiased(m * K, m * K)
        self._self_cache[tag] = v
        return v

    def score(self, a, b, ta=None, tb=None):
        """a, b are the cached artifacts for one layer pair."""
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
        # gram metrics, reusing the cached self terms
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


def load_side(scorer, side, key):
    k = scorer.kind()
    if k == "knn":
        return load_knn(side, key)
    if k == "svcca":
        return load_u(side, key)
    return load_gram(side, key)


def run_text(scorer, manifest, log):
    layers = manifest["text_layers"]
    pairs = C.text_pairs()
    records = []
    t0 = time.time()
    cur_a, cache_a = None, None
    for a, b in pairs:
        if a != cur_a:                        # pairs are grouped by first model
            cache_a = {(c, L): load_side(scorer, "text", f"{a}_{c}_L{L}")
                       for c in C.COMPS for L in layers[a]}
            cur_a = a
            if scorer.kind() == "gram":
                torch.cuda.empty_cache()
        cache_b = {(c, L): load_side(scorer, "text", f"{b}_{c}_L{L}")
                   for c in C.COMPS for L in layers[b]}
        rec = dict(a=a, b=b)
        for c in C.COMPS:
            vals = []
            for La in layers[a]:
                for Lb in layers[b]:
                    vals.append(scorer.score(
                        cache_a[(c, La)], cache_b[(c, Lb)],
                        f"text|{a}|{c}|{La}", f"text|{b}|{c}|{Lb}"))
            rec[c] = float(np.mean(vals))
        records.append(rec)
        del cache_b
        if scorer.kind() == "gram":
            torch.cuda.empty_cache()
        log(f"  text {a} vs {b}: " + " ".join(
            f"{c} {rec[c]:.4f}" for c in C.COMPS)
            + f"  [{time.time()-t0:.0f}s]")
    return records


def run_vision(scorer, manifest, log):
    cap_layers = manifest["cap_layers"]
    img_layers = manifest["img_layers"]
    img_cache = {}
    for enc in C.ENCODERS:
        for L in img_layers[enc]:
            img_cache[(enc, L)] = load_side(scorer, "img", f"{enc}_L{L}")
    records = []
    t0 = time.time()
    for tname in C.MODELS:
        cap = {(c, L): load_side(scorer, "cap", f"{tname}_{c}_L{L}")
               for c in C.COMPS for L in cap_layers[tname]}
        for enc in C.ENCODERS:
            rec = dict(text=tname, enc=enc)
            for c in C.COMPS:
                vals = []
                for Lt in cap_layers[tname]:
                    for Li in img_layers[enc]:
                        vals.append(scorer.score(
                            cap[(c, Lt)], img_cache[(enc, Li)],
                            f"cap|{tname}|{c}|{Lt}", f"img|{enc}|{Li}"))
                rec[c] = float(np.mean(vals))
            records.append(rec)
            log(f"  vision {tname} x {enc}: " + " ".join(
                f"{c} {rec[c]:.4f}" for c in C.COMPS)
                + f"  [{time.time()-t0:.0f}s]")
        del cap
        if scorer.kind() == "gram":
            torch.cuda.empty_cache()
    return records


def stats_text(records, n_perm):
    hs = C.hellaswag(list(C.MODELS))
    names = [(r["a"], r["b"]) for r in records]
    y = [(hs[a] + hs[b]) / 2 for a, b in names]
    out = {}
    for c in C.COMPS:
        al = [r[c] for r in records]
        rho = C.spearman(y, al)
        slope, se = C.ols(y, al)
        out[c] = dict(mean=float(np.mean(al)), min=float(np.min(al)),
                      max=float(np.max(al)), rho=rho,
                      perm_p=C.perm_p_pairs(al, hs, names, rho, n_perm=n_perm),
                      slope=slope, slope_se=se)
    return out


def stats_vision(records, n_perm):
    hs = C.hellaswag(list(C.MODELS))
    models = list(C.MODELS)
    by = {(r["text"], r["enc"]): r for r in records}
    per_encoder, averaged = {}, {}
    for enc in C.ENCODERS:
        per_encoder[enc] = {}
        for c in C.COMPS:
            al = [by[(m, enc)][c] for m in models]
            rho = C.spearman([hs[m] for m in models], al)
            per_encoder[enc][c] = dict(
                mean=float(np.mean(al)), rho=rho,
                perm_p=C.perm_p_models(al, [hs[m] for m in models], rho,
                                       n_perm=n_perm))
    for c in C.COMPS:
        al = [float(np.mean([by[(m, e)][c] for e in C.ENCODERS])) for m in models]
        rho = C.spearman([hs[m] for m in models], al)
        slope, se = C.ols([hs[m] for m in models], al)
        averaged[c] = dict(mean=float(np.mean(al)), min=float(np.min(al)),
                           max=float(np.max(al)), rho=rho,
                           perm_p=C.perm_p_models(al, [hs[m] for m in models],
                                                  rho, n_perm=n_perm),
                           slope=slope, slope_se=se,
                           per_model={m: v for m, v in zip(models, al)})
    return dict(per_encoder=per_encoder, averaged=averaged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=ALL_METRICS)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n_perm", type=int, default=200_000)
    ap.add_argument("--skip_text", action="store_true")
    ap.add_argument("--skip_vision", action="store_true")
    args = ap.parse_args()

    out = os.path.abspath(args.outdir)
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            existing = json.load(f)
        if existing.get("complete"):
            print(f"{args.metric}: already complete, skipping")
            return

    logf = open(os.path.join(out, "run.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    log(f"=== metric {args.metric} | topk={C.TOPK} cca_dim={C.CCA_DIM} "
        f"| perm draws {args.n_perm} ===")
    with open(os.path.join(C.FEATURES, "manifest.json")) as f:
        manifest = json.load(f)

    scorer = Scorer(args.metric)
    payload = dict(metric=args.metric, topk=C.TOPK, cca_dim=C.CCA_DIM,
                   n_perm=args.n_perm, complete=False,
                   note="alignment = mean of the metric over the band x band "
                        "layer-pair grid; identical decomposition and "
                        "preprocessing to the paper, metric varied")

    if not args.skip_text:
        tr = run_text(scorer, manifest, log)
        payload["text_pairs"] = tr
        payload["text_stats"] = stats_text(tr, args.n_perm)
        log("text stats: " + json.dumps(
            {c: {k: round(v, 4) for k, v in payload["text_stats"][c].items()}
             for c in C.COMPS}))
        with open(res_path, "w") as f:
            json.dump(payload, f, indent=1)

    scorer._self_cache.clear()
    if not args.skip_vision:
        vr = run_vision(scorer, manifest, log)
        payload["vision_pairs"] = vr
        payload["vision_stats"] = stats_vision(vr, args.n_perm)
        log("vision stats (encoder-averaged): " + json.dumps(
            {c: {k: round(v, 4) for k, v in
                 payload["vision_stats"]["averaged"][c].items()
                 if k != "per_model"} for c in C.COMPS}))

    payload["complete"] = not (args.skip_text or args.skip_vision)
    payload["seconds"] = time.time() - t0
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=1)
    log(f"=== {args.metric} done in {time.time()-t0:.0f}s -> {res_path} ===")
    logf.close()


if __name__ == "__main__":
    main()
