"""Figure 2 of the paper (paper/07_lobf_components.pdf): alignment against
competence for the full activation, the non-J remainder and the J-space.

(a) within-language: one point per MODEL PAIR (55), x = mean HellaSwag of the
    two models, y = that pair's mean over its layer-pair grid. The legend
    p-values are the model-label permutation values printed by xmeanmain.py --
    not scipy's, which would treat the 55 interdependent pairs as independent
    and report 0.0000.

(b) cross-modal: one point per TEXT MODEL (11), alignment averaged over the
    four vision encoders. The 44 cross-modal pairs are not pooled; per-encoder
    correlations are in Appendix tab:perencoder.

Panel (b) error bars are the standard error across the four encoders of the
same mean-over-grid statistic the points plot, computed from measB.json.

Reads results/randdict_null_bylayerstat.json, results/measB.json, and
results/lmeval/*.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr

M_ORDER = ["pythia70m", "gpt2", "gemma270", "qwen08b", "gemma", "qwen17b",
           "qwen2b", "gemma2_2b", "qwen4b", "qwen35_4b", "gemma3_4b"]
COMPS = [("full", "full activation", "#2a78d6"),
         ("perp", "non-J remainder", "#eb6834"),
         ("J", "J-space", "#1baf7a")]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FS_LABEL, FS_TITLE, FS_TICK, FS_LEG, FS_NOTE = 11, 12, 9.5, 9.5, 8.5

hs = {m: json.load(open(f"results/lmeval/{m}.json", encoding="utf-8"))
      ["hellaswag_acc_norm"] for m in M_ORDER}
# Legend p-values, printed by xmeanmain.py: model-label permutation values
# within-language (55 pairs, 200k permutations: 164, 17 and 3219 hits) and
# exact permutation values cross-modally (n=11).
# Store the UNROUNDED values: fmt() renders "< 0.0001" below 1e-4, so a
# pre-rounded 0.0001 would print as an equality and disagree with the tables --
# perp in particular MUST stay unrounded.
# J is held at the 0.0162 printed in tab:competence.
P_PAIR = {"full": 0.000820, "perp": 0.000085, "J": 0.0162}
P_MODEL = {"full": 0.0001, "perp": 0.0002, "J": 0.00005}

# ---- (a) within-language, per pair
s55 = json.load(open("results/randdict_null_bylayerstat.json",
                   encoding="utf-8"))["pairs"]
x_pair = np.array([(hs[p["a"]] + hs[p["b"]]) / 2 for p in s55])
y_pair = {c: np.array([p["real"][c]["real_mean"] for p in s55]) for c, _, _ in COMPS}

# ---- (b) cross-modal, per text model (encoders averaged)
x_model = np.array([hs[m] for m in M_ORDER])
measB = json.load(open("results/measB.json", encoding="utf-8"))
VN = ["dinov2", "mae", "clip", "siglip"]
gmean = lambda m, v, c: float(np.mean(list(map(float, measB[m][v][c]["grid"].values()))))
y_model = {c: np.array([np.mean([gmean(m, v, c) for v in VN]) for m in M_ORDER])
           for c, _, _ in COMPS}
sem_model = {c: {m: float(np.std([gmean(m, v, c) for v in VN], ddof=1)
                          / np.sqrt(len(VN)))
                 for m in M_ORDER}
             for c, _, _ in COMPS}


def frame(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=FS_TICK)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def panel(ax, xs, ys, plabel, ylim, chance, xlabel, ylabel, title, sems=None):
    frame(ax)
    grid = np.linspace(xs.min(), xs.max(), 100)
    for key, label, colour in COMPS:
        y = ys[key]
        fit = linregress(xs, y)
        rho = spearmanr(xs, y)[0]
        p = plabel(key)
        if sems is not None:
            ax.errorbar(xs, y, yerr=np.array([sems[key][m] for m in M_ORDER]),
                        fmt="none", ecolor=colour, elinewidth=1.1, capsize=2.5,
                        alpha=0.55, zorder=2)
        ax.scatter(xs, y, s=26 if sems is None else 40, c=colour,
                   alpha=0.62 if sems is None else 0.9, edgecolors=SURFACE,
                   linewidths=0.8 if sems is None else 1.2, zorder=3)
        ax.plot(grid, fit.intercept + fit.slope * grid, color=colour, lw=2.2,
                zorder=4,
                label=f"{label}:  {fit.slope:+.2f},  $\\rho$ {rho:.2f},  p {p}")
    ax.text(0.985, 0.015, f"chance = {chance:.3f}", transform=ax.transAxes,
            fontsize=FS_NOTE, color=MUTED, va="bottom", ha="right")
    ax.set_xlabel(xlabel, color=INK2, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, color=INK2, fontsize=FS_LABEL)
    ax.set_title(title, color=INK, fontsize=FS_TITLE, pad=8)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=False, fontsize=FS_LEG,
              labelcolor=INK2, labelspacing=0.4, handlelength=1.5,
              handletextpad=0.5, borderpad=0.2)


def fmt(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), facecolor=SURFACE)
panel(axes[0], x_pair, y_pair,
      lambda k: fmt(P_PAIR[k]),
      (0.30, 0.78), 10 / 999,
      "mean HellaSwag accuracy of the pair", "within-language alignment",
      "(a) between language models, per pair")
panel(axes[1], x_model, y_model,
      lambda k: fmt(P_MODEL[k]),
      (0.03, 0.14), 10 / 1023,
      "HellaSwag accuracy", "cross-modal alignment",
      "(b) to vision encoders, per text model", sems=sem_model)
fig.tight_layout()
out = "paper/07_lobf_components.pdf"
fig.savefig(out, facecolor=SURFACE)
plt.close(fig)
print("wrote", out)
for c, _, _ in COMPS:
    print(f"  (a) {c:>5}: n=55 rho {spearmanr(x_pair, y_pair[c])[0]:+.2f} "
          f"label-perm p {P_PAIR[c]:.4f}")
