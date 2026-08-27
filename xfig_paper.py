"""Figure 2 of the paper (section 4.3), drawn at print size.

  (a) paper/06_wordalign_mean_heatmap.pdf -- mean J-lens word agreement over
      the 11x11 relative-depth grid, from results/wordalign/mean_grid_raw.json.
  (b) paper/08_j_minus_logit_vs_depth.pdf -- J-lens minus plain-logit-lens
      agreement against relative depth, mean +- sd within same-family and
      cross-family pairs, from the per-pair grid diagonals in
      results/wordalign/stats.json (xw_stats.py).

Both panels plot the matched top-25 overlap directly. The position-shuffled
floor (0.1 of 25 words) is reported in the paper's controls section only, so it
is not subtracted here; the raw_*_k25 fields are used, not the diag_* deltas.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "paper"
SURFACE, INK2, MUTED = "#fcfcfb", "#52514e", "#898781"
GRID = "#e3e2df"


def sym(A):
    A = np.array(A)
    return (A + A.transpose(0, 2, 1)) / 2


M = json.load(open("results/wordalign/mean_grid_raw.json", encoding="utf-8"))
PG = M["pgrid"]
mean_J = sym(M["J"]).mean(0)

fig, ax = plt.subplots(figsize=(3.5, 3.15), facecolor=SURFACE)
im = ax.imshow(mean_J, origin="lower", cmap="magma", vmin=0,
               extent=(-0.05, 1.05, -0.05, 1.05))
ax.plot([-0.05, 1.05], [-0.05, 1.05], color="#7dd3fc", lw=1.4, ls="--", zorder=3)
ax.text(0.20, 0.27, "matched depth", rotation=45, rotation_mode="anchor",
        ha="center", va="center", color="#7dd3fc", fontsize=8, zorder=4)
ax.set_xlabel("relative depth of model B,  $\\lambda$", color=INK2, fontsize=9)
ax.set_ylabel("relative depth of model A,  $\\lambda$", color=INK2, fontsize=9)
ax.set_xticks(PG[::2]); ax.set_yticks(PG[::2])
ax.set_xticklabels([f"{p:.1f}" for p in PG[::2]])
ax.set_yticklabels([f"{p:.1f}" for p in PG[::2]])
ax.tick_params(colors=MUTED, labelsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("shared words / 25", color=INK2, fontsize=8.5)
cb.ax.tick_params(colors=MUTED, labelsize=7.5)
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/06_wordalign_mean_heatmap.pdf", facecolor=SURFACE)
plt.close(fig)
print(f"wrote {OUT}/06_wordalign_mean_heatmap.pdf")

S = json.load(open("results/wordalign/stats.json", encoding="utf-8"))
P = S["pairs"]
PGb = S["pgrid"]
D = {k: [j - b for j, b in zip(x["raw_J_k25"], x["raw_base_k25"])]
     for k, x in P.items()}

# p = 0 is dropped from this panel only: there the plain logit lens reads the
# embedding matrix straight back, which is not a comparison of lenses.
i0 = 1
fig, ax = plt.subplots(figsize=(3.7, 3.15), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
lo, hi = [], []
for lab, sel, col in [
        ("same family", [k for k in P if P[k]["same_family"]], "#2a78d6"),
        ("cross family", [k for k in P if not P[k]["same_family"]], "#eb6834")]:
    Ms = np.array([D[k] for k in sel])[:, i0:]
    m, sd = Ms.mean(0), Ms.std(0)
    ax.plot(PGb[i0:], m, color=col, lw=2.2, label=f"{lab} (n={len(sel)})", zorder=3)
    ax.fill_between(PGb[i0:], m - sd, m + sd, color=col, alpha=0.18, zorder=2)
    lo.append((m - sd).min()); hi.append((m + sd).max())
ax.axhline(0, color=MUTED, lw=1.1, ls="--", zorder=2)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
ax.set_xlabel(r"relative depth  $\lambda = \ell/(L_A-1)$", color=INK2, fontsize=9)
ax.set_ylabel("J-lens $-$ logit-lens overlap", color=INK2, fontsize=9)
pad = 0.08 * (max(hi) - min(lo))
ax.set_ylim(min(min(lo) - pad, -0.02), max(hi) + pad)
ax.set_xlim(PGb[i0] - 0.03, 1.03)
ax.grid(True, color=GRID, lw=0.7, zorder=0)
ax.tick_params(colors=MUTED, labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT}/08_j_minus_logit_vs_depth.pdf", facecolor=SURFACE)
plt.close(fig)
print(f"wrote {OUT}/08_j_minus_logit_vs_depth.pdf")
