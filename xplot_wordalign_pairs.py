"""Figure 4 of the paper (paper/09_wordalign_pairs_vs_competence.pdf): per-pair
J-lens WORD overlap against the pair's competence.

This is the Experiment 3 statistic -- vocabulary read straight off the J-lens,
not the activation-kernel m-NN of Figure 2(a). For each of the 55 pairs,
y = median over the 11 matched-relative-depth grid points of the mean matched
top-25 overlap (results/wordalign/stats.json, raw_J_k25), x = mean HellaSwag
acc_norm of the two models.

The position-shuffled floor (0.1 of 25 words) is NOT subtracted: it is reported
in the paper's controls section only. Subtracting it moves rho from +0.74 to
+0.73, so the choice is immaterial to the trend.
The right axis restates the overlap as words out of 25, the paper's framing.

The 55 pairs are not independent (each model appears in 10), so the quoted p
is a model-label permutation value -- competence permuted across the 11
models with the overlap values held fixed -- matching tab:competence's
convention. Colour marks family composition.

Prints the two section 4.3 correlations: rho = +0.74 for the J-lens and
rho = +0.41 for the plain logit lens.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr

NPERM, SEED = 200000, 0
FAMILY = {"gemma270": "Gemma", "gemma": "Gemma", "gemma2_2b": "Gemma",
          "gemma3_4b": "Gemma", "qwen08b": "Qwen", "qwen17b": "Qwen",
          "qwen2b": "Qwen", "qwen4b": "Qwen", "qwen35_4b": "Qwen",
          "gpt2": "GPT-style", "pythia70m": "GPT-style"}
CAT_COLOR = {"Gemma": "#2a78d6", "Qwen": "#eb6834", "GPT-style": "#1baf7a",
             "cross-family": "#9b9992"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FS_LABEL, FS_TICK, FS_LEG = 10.5, 9.0, 9.0

S = json.load(open("results/wordalign/stats.json", encoding="utf-8"))
NAMES = list(S["per_model"])
hs = {m: S["per_model"][m]["hellaswag"] for m in NAMES}
P = S["pairs"]
keys = list(P)
assert len(keys) == 55

ab = [k.split("|") for k in keys]
x = np.array([(hs[a] + hs[b]) / 2 for a, b in ab])
y = np.array([float(np.median(P[k]["raw_J_k25"])) for k in keys])
y_base = np.array([float(np.median(P[k]["raw_base_k25"])) for k in keys])
cat = [FAMILY[a] if FAMILY[a] == FAMILY[b] else "cross-family" for a, b in ab]

rho = spearmanr(x, y)[0]
fit = linregress(x, y)

# ---- model-label permutation null (pairs share models, so naive p is invalid)
IDX = {m: i for i, m in enumerate(NAMES)}
ai = np.array([IDX[a] for a, _ in ab])
bi = np.array([IDX[b] for _, b in ab])
vals = np.array([hs[m] for m in NAMES])
rng = np.random.default_rng(SEED)
perm_x = (lambda o: (vals[o][ai] + vals[o][bi]) / 2)
hits = sum(abs(spearmanr(perm_x(rng.permutation(len(NAMES))), y)[0])
           >= abs(rho) - 1e-12 for _ in range(NPERM))
p_perm = hits / NPERM

rho_b = spearmanr(x, y_base)[0]
rng_b = np.random.default_rng(SEED)
hits_b = sum(abs(spearmanr(perm_x(rng_b.permutation(len(NAMES))), y_base)[0])
             >= abs(rho_b) - 1e-12 for _ in range(NPERM))
p_perm_b = hits_b / NPERM

print(f"  J-lens : rho {rho:+.4f}  label-perm p {p_perm:.5f}  "
      f"slope {fit.slope:+.3f} +/- {fit.stderr:.3f}")
print(f"  logit  : rho {rho_b:+.4f}  label-perm p {p_perm_b:.5f}")
print(f"  median overlap over 55 pairs {np.median(y):.3f} "
      f"({25 * np.median(y):.1f} of 25 words)")

fig, ax = plt.subplots(figsize=(6.6, 3.0), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, lw=0.6, zorder=0)
ax.spines["top"].set_visible(False)
for s in ("left", "bottom", "right"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=FS_TICK)
ax.set_xlim(x.min() - 0.018, x.max() + 0.018)
ax.set_ylim(0.03, 0.43)

grid = np.linspace(x.min(), x.max(), 100)
ax.plot(grid, fit.intercept + fit.slope * grid, color=INK2, lw=1.6, ls="--",
        zorder=2)
for c in ("cross-family", "Gemma", "Qwen", "GPT-style"):
    sel = np.array([k == c for k in cat])
    ax.scatter(x[sel], y[sel], s=30, c=CAT_COLOR[c], alpha=0.9,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)

ax.text(0.025, 0.965,
        f"ρ = {rho:+.2f},  label-permutation p = {p_perm:.4f},  "
        f"slope {fit.slope:+.2f} ± {fit.stderr:.2f}",
        transform=ax.transAxes, va="top", fontsize=FS_LEG, color=INK)
handles = [plt.Line2D([], [], marker="o", ls="", color=CAT_COLOR[c], ms=6,
                      label=l)
           for c, l in [("Gemma", "both Gemma"), ("Qwen", "both Qwen"),
                        ("GPT-style", "both GPT-style"),
                        ("cross-family", "cross-family")]]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS_LEG,
          labelcolor=INK2, labelspacing=0.25, ncols=2, columnspacing=1.2,
          handletextpad=0.4, borderpad=0.2)
ax.set_xlabel("mean HellaSwag accuracy of the pair", color=INK2,
              fontsize=FS_LABEL)
ax.set_ylabel("J-lens word agreement", color=INK2, fontsize=FS_LABEL)

rax = ax.twinx()                       # same data, restated as words out of 25
rax.set_ylim(0.03 * 25, 0.43 * 25)
rax.set_ylabel("words out of 25", color=INK2, fontsize=FS_LABEL)
rax.tick_params(colors=MUTED, labelsize=FS_TICK)
rax.spines["top"].set_visible(False)
for s in ("left", "bottom", "right"):
    rax.spines[s].set_color(GRID)

fig.tight_layout()
fig.savefig("paper/09_wordalign_pairs_vs_competence.pdf", facecolor=SURFACE)
plt.close(fig)
print("wrote paper/09_wordalign_pairs_vs_competence.pdf")
