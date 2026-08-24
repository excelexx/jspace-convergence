"""Appendix figure for the lens-fitting control -> paper/10_lens_fitting_control.pdf.

Three panels over the 10 pairs among the 5 in-scope models, all on the median
over each pair's layer-pair grid, on the 1,000 Pile evaluation documents:

  left    crossed-half against shared-half J-component alignment, with y=x and
          the +-0.02 band;
  middle  the three fitting conditions per pair -- shared-full, shared-half,
          crossed-half -- against the per-pair random-dictionary null (max of
          the R=19 draws);
  right   the per-pair difference Delta = shared-half - crossed-half, whose
          median is the +0.0002 the appendix quotes.

Everything is recomputed from the neighbour-set cache xlc_phase4.py wrote in
results/lenscontrol/sparse/, including the null: the draws stored in
phase4_sparse.json are grid maxima and this figure is on grid medians.
"""
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crossmodal_utils import LENSFIT_SCOPE
from xkernels import grid_median

R_NULL = 19
BAND = 0.02
OUT_PDF = "paper/10_lens_fitting_control.pdf"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_SHF, C_SHH, C_CRH = "#2a78d6", "#eb6834", "#1baf7a"

S = {m: torch.load(f"results/lenscontrol/sparse/{m}.pt", weights_only=False)
     for m in LENSFIT_SCOPE}

labels, sf, sh, ch, nulls = [], [], [], [], []
for a, b in combinations(LENSFIT_SCOPE, 2):
    labels.append(f"{a}×{b}")
    sf.append(grid_median(S[a]["cond"]["full"], S[b]["cond"]["full"]))
    s1, s2 = (grid_median(S[a]["cond"][h], S[b]["cond"][h])
              for h in ("h1", "h2"))
    c1, c2 = (grid_median(S[a]["cond"][x], S[b]["cond"][y])
              for x, y in (("h1", "h2"), ("h2", "h1")))
    sh.append((s1 + s2) / 2)
    ch.append((c1 + c2) / 2)
    nulls.append(max(grid_median(S[a]["null"][r], S[b]["null"][r])
                     for r in range(R_NULL)))

sf, sh, ch, nulls = map(np.array, (sf, sh, ch, nulls))
delta = sh - ch

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0), facecolor=SURFACE)
for a_ in ax:
    a_.set_facecolor(SURFACE)
    a_.grid(True, color=GRID, lw=0.7, zorder=0)
    a_.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a_.spines[s].set_color(GRID)

# --- (a) paired scatter
lo = min(sh.min(), ch.min()) - 0.02
hi = max(sh.max(), ch.max()) + 0.02
ax[0].fill_between([lo, hi], [lo - BAND, hi - BAND], [lo + BAND, hi + BAND],
                   color=GRID, alpha=0.55, zorder=1,
                   label=f"$\\pm${BAND:.2f} materiality band")
ax[0].plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=INK2, zorder=2,
           label="$y=x$ (no effect of sharing)")
ax[0].scatter(sh, ch, s=44, c=C_CRH, alpha=0.9, edgecolors=SURFACE,
              linewidths=1.2, zorder=3)
ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
ax[0].set_xlabel("shared-half: both lenses on the same half", color=INK2, fontsize=9.5)
ax[0].set_ylabel("crossed-half: lenses on disjoint halves", color=INK2, fontsize=9.5)
ax[0].set_title("points on the diagonal mean sharing does nothing",
                color=INK, fontsize=10, pad=7)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

# --- (b) three conditions vs null
x = np.arange(len(labels)); w = 0.26
ax[1].bar(x - w, sf, w, color=C_SHF, label="shared-full corpus", zorder=3)
ax[1].bar(x, sh, w, color=C_SHH, label="shared-half", zorder=3)
ax[1].bar(x + w, ch, w, color=C_CRH, label="crossed-half", zorder=3)
ax[1].scatter(x, nulls, marker="x", s=46, c=INK, zorder=4,
              label=f"random-dictionary null (max of $R$={R_NULL})")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax[1].set_ylabel("J-component alignment (m-NN)", color=INK2, fontsize=9.5)
ax[1].set_ylim(0, 0.70)
ax[1].set_title("all three conditions sit far above the null",
                color=INK, fontsize=10, pad=7)
ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center",
             ncol=2, columnspacing=1.2, handlelength=1.4)

# --- (c) per-pair delta
ax[2].axhspan(-BAND, BAND, color=GRID, alpha=0.55, zorder=1,
              label=f"$\\pm${BAND:.2f} materiality band")
ax[2].axhline(0, color=INK2, lw=1.0, zorder=2)
ax[2].bar(x, delta, 0.6, color=C_SHH, zorder=3)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax[2].set_ylim(-0.03, 0.03)
ax[2].set_ylabel("$\\Delta$ = shared-half $-$ crossed-half", color=INK2, fontsize=9.5)
ax[2].set_title(f"every $\\Delta$ a hairline (median {np.median(delta):+.4f})",
                color=INK, fontsize=10, pad=7)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

fig.tight_layout()
fig.savefig(OUT_PDF, facecolor=SURFACE)
plt.close(fig)
print(f"wrote {OUT_PDF}")
