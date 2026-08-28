"""Print every table the paper contains, with the MEAN at both aggregation
steps: each pair is summarised by the mean over its layer-pair grid, and pairs
(or models) are then averaged.

    Table 1  tab:surrogate   content ablation over the 55 text pairs,
                             including the Gaussian-dictionary row
    Table 2  tab:perencoder  cross-modal correlation within each encoder
    Table 3  tab:competence  within-language alignment vs competence
    Table 4  tab:crossaxes   cross-modal alignment vs three competence axes

Rows are printed as LaTeX. Figure 2's legend values are the HellaSwag columns
of Table 3 (panel a) and Table 4 (panel b); the mean alignment levels quoted in
prose in section 4.1 are Table 1's first column.

Within-language per-pair mean-over-grid comes from
results/randdict_null_bylayerstat.json (the `real` variant, and `rand` for the
Gaussian-dictionary row); cross-modal mean-over-grid is computed from the
stored grids in results/measB.json. Bootstrap (2,000 resamples) and permutation
settings match what the paper quotes.
"""
import json

import numpy as np
from scipy.stats import spearmanr

NB, SEED, NPERM = 2000, 9, 50000
P = {"pythia70m": 70, "gpt2": 124, "gemma270": 270, "qwen08b": 800,
     "gemma": 1000, "qwen17b": 1700, "qwen2b": 2000, "gemma2_2b": 2600,
     "qwen4b": 4000, "qwen35_4b": 4000, "gemma3_4b": 4300}
O = list(P)
IDX = {m: i for i, m in enumerate(O)}
C = ("full", "perp", "J")
VN = [("dinov2", "DINOv2"), ("mae", "MAE"), ("clip", "CLIP"),
      ("siglip", "SigLIP")]

BL = json.load(open("results/randdict_null_bylayerstat.json",
                    encoding="utf-8"))["pairs"]
measB = json.load(open("results/measB.json", encoding="utf-8"))
hs = np.array([json.load(open(f"results/lmeval/{m}.json", encoding="utf-8"))
               ["hellaswag_acc_norm"] for m in O])
lp = np.array([np.log10(P[m]) for m in O])
bp = np.array([json.load(open("results/performance_owt.json",
                              encoding="utf-8"))[m]["performance"] for m in O])


def boot(x, scale=1.0):
    """95% percentile bootstrap CI of the mean, NB resamples over the pairs."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(NB, len(x)))
    return np.percentile(scale * np.mean(x[idx], axis=1), [2.5, 97.5])


print("=" * 78)
print("TABLE 1 (tab:surrogate)  content ablation, mean over 55 pairs")
print("=" * 78)


def table1_row(label, variant, comp):
    """One Table 1 row: real, content-ablated, retention, each with its CI."""
    r = np.array([p[variant][comp]["real_mean"] for p in BL])
    s = np.array([p[variant][comp]["surr_mean"] for p in BL])
    t = np.array([p[variant][comp]["ret_mean"] for p in BL])
    out = []
    for arr, sc, pct in ((r, 1, 0), (s, 1, 0), (t, 100, 1)):
        lo, hi = boot(arr, sc)
        out.append(("%.1f\\%%\\,{\\tiny[%.2f, %.2f]}" if pct
                    else "%.4f\\,{\\tiny[%.3f, %.3f]}") % (sc * arr.mean(), lo, hi))
    print("    %-9s& %s \\\\" % (label, " & ".join(out)))


print("    \\multicolumn{4}{l}{\\emph{Components, J-lens dictionary}} \\\\")
for c, label in zip(C, ("full", "non-J", "J")):
    table1_row(label, "real", c)
print("    \\multicolumn{4}{l}{\\emph{J-space under a substituted dictionary}} \\\\")
table1_row("Gaussian", "rand", "J")

ai = np.array([IDX[p["a"]] for p in BL])
bi = np.array([IDX[p["b"]] for p in BL])
Ypair = {c: np.array([p["real"][c]["real_mean"] for p in BL]) for c in C}

def gmean(m, v, c):
    return float(np.mean(list(map(float, measB[m][v][c]["grid"].values()))))

Xmodel = {c: np.array([np.mean([gmean(m, v, c) for v, _ in VN]) for m in O])
          for c in C}

rng = np.random.default_rng(0)
n = 11
a_ = np.arange(n); am = a_.mean(); den = ((a_ - am) ** 2).sum()
null11 = ((rng.permuted(np.tile(a_, (1_000_000, 1)), axis=1) - am)
          * (a_ - am)).sum(1) / den
ex = lambda r: float((np.abs(null11) >= abs(r) - 1e-12).mean())


def labperm(y, vals):
    r = spearmanr((vals[ai] + vals[bi]) / 2, y)[0]
    rg = np.random.default_rng(SEED)
    hit = 0
    for _ in range(NPERM):
        pv = rg.permutation(vals)
        if abs(spearmanr((pv[ai] + pv[bi]) / 2, y)[0]) >= abs(r) - 1e-12:
            hit += 1
    return r, hit / NPERM


def fmt(r, p):
    ps = "p < 0.0001" if p < 1e-4 else "p = %.4f" % p
    return r"$\rho = %+.2f$\,{\tiny$%s$}" % (r, ps)


print()
print("=" * 78)
print("TABLE 2 (tab:perencoder)  cross-modal correlation within each encoder")
print("=" * 78)
for v, disp in VN:
    lvl = np.mean([gmean(m, v, "J") for m in O])
    cells = ["%.4f" % lvl]
    for c in C:
        r = spearmanr(hs, [gmean(m, v, c) for m in O])[0]
        cells.append(fmt(r, ex(r)))
    print("    %-8s& %s \\\\" % (disp, " & ".join(cells)))

print()
print("=" * 78)
print("TABLE 3 (tab:competence)  within-language competence, 55 pairs")
print("=" * 78)
for c in C:
    cells = [fmt(*labperm(Ypair[c], v)) for v in (lp, hs)]
    print("    %-5s& %s \\\\" % (c, " & ".join(cells)))

print()
print("=" * 78)
print("TABLE 4 (tab:crossaxes)  cross-modal competence, 11 models")
print("=" * 78)
for c in C:
    cells = []
    for v in (lp, hs, bp):
        r = spearmanr(v, Xmodel[c])[0]
        cells.append(fmt(r, ex(r)))
    print("    %-5s& %s \\\\" % (c, " & ".join(cells)))
