"""Content ablation: frequency-matched content-word substitution.

Builds the ablated corpus described in section 3.1, on which Table 1 and the
retention statistics of section 4.1 are computed. Every content-word
occurrence is replaced by a random word drawn from the same corpus
log-frequency bin. Function words, punctuation, numbers, whitespace and
capitalisation are untouched, so document length, syntactic skeleton, register
and unigram frequency profile all survive while word identity and
co-occurrence do not.

Substitution is per occurrence, not per type.

The corpus is written once to cache/surrogate_docs.json and read by
xsurrogate_all.py, which caches the activations the Gaussian-dictionary arm
then reuses, so both arms of Table 1 are computed against the same ablated
text.

This module is also the numerical core for experiment 1, imported by those
scripts. The numerics themselves are the pilot's, lifted out of
step7_align.py by crossmodal_utils.load_pilot() and re-exported here under
their original names: non-negative OMP against the J-lens dictionary, the
winsorise-and-normalise preprocessing, k-NN neighbour lists and the m-NN
alignment statistic (kappa = 10, k = 25 atoms, band 0.35-0.90).
"""
import json
import os
import re

import numpy as np
import torch

from crossmodal_utils import load_pilot

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS, MAX_TOKENS = 1000, 300
SEED = 0
N_FREQ_BINS = 20

_pilot = load_pilot(dev=DEV)
K_NN = _pilot.K_NN
get_WU_and_w = _pilot.get_WU_and_w
nnls_refit = _pilot.nnls_refit
best_atom = _pilot.best_atom
nnomp_batch = _pilot.nnomp_batch
prep = _pilot.prep
neighbors = _pilot.neighbors
mnn = _pilot.mnn


SURR_TEXT = "cache/surrogate_docs.json"

STOP = set("""a about above after again against all am an and any are aren't as at be
because been before being below between both but by can cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves will shall may might must can't
one two three first second new also just like get got make made see said says""".split())

WORD_RE = re.compile(r"[A-Za-z]+")


def match_case(src, repl):
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper():
        return repl.capitalize()
    return repl


def build_surrogate(docs):
    """Per-occurrence content-word substitution within log-frequency bins."""
    freq = {}
    for d in docs:
        for w in WORD_RE.findall(d):
            lw = w.lower()
            if lw not in STOP:
                freq[lw] = freq.get(lw, 0) + 1

    types = sorted(freq)                          # sorted: deterministic binning
    logf = np.log10(np.array([freq[t] for t in types], dtype=np.float64))
    edges = np.quantile(logf, np.linspace(0, 1, N_FREQ_BINS + 1))
    edges[-1] += 1e-9
    bin_of = np.clip(np.searchsorted(edges, logf, side="right") - 1,
                     0, N_FREQ_BINS - 1)
    bins = [[] for _ in range(N_FREQ_BINS)]
    for t, b in zip(types, bin_of):
        bins[b].append(t)
    tbin = dict(zip(types, bin_of))

    rng = np.random.default_rng(SEED)
    out = []
    for d in docs:
        def sub(m):
            w = m.group(0)
            lw = w.lower()
            if lw in STOP or lw not in tbin:
                return w
            pool = bins[tbin[lw]]
            return match_case(w, pool[rng.integers(len(pool))])
        out.append(WORD_RE.sub(sub, d))
    return out


def main():
    """Write cache/surrogate_docs.json, the ablated corpus for section 4.1."""
    from datasets import load_dataset
    os.makedirs("cache", exist_ok=True)
    print("=== corpus ===")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs = [d["text"][:1500] for d in ds.select(range(N_DOCS))]
    if os.path.exists(SURR_TEXT):
        surr = json.load(open(SURR_TEXT, encoding="utf-8"))
        print(f"  {SURR_TEXT} cached")
    else:
        surr = build_surrogate(docs)
        json.dump(surr, open(SURR_TEXT, "w", encoding="utf-8"))
    print("\n  sample real:      ", repr(docs[1][:180]))
    print("  sample surrogate: ", repr(surr[1][:180]))
    print("\nrun xsurrogate_all.py next to decompose all 11 models.")


if __name__ == "__main__":
    main()
