"""Teaser figure (paper/01_teaser.pdf): what the J-space is, that it is the
semantically loaded part of the activation, and what that does to the
competence-convergence trend.

Schematic. The only numbers are Table 1's retentions and the slopes already
reported in section 4.1; the document text and word lists are illustrative.

Styled to match 07_lobf_components.pdf: same palette/type scale, matplotlib's
default sans throughout (no custom font family), no bordered card panels --
colour-coded text and thin accent rules instead, the same idiom the rest of
the paper's figures use.

Sized for \\linewidth (5.5in) at the NeurIPS text width: the canvas is 7.4in
wide, so everything renders at 0.74x.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
MONO = "DejaVu Sans Mono"
# same scale as 07_lobf_components.pdf (FS_LABEL..FS_NOTE = 11,12,9.5,9.5,8.5)
# at this canvas's 0.74x relative size
FS_TITLE, FS_LABEL, FS_NOTE, FS_MONO = 9.2, 8.4, 6.5, 7.0

RET = [("full activation", 54.1, BLUE, False),
       ("J-space", 26.3, GREEN, False),
       ("Gaussian dict.", 34.7, MUTED, True)]
SLOPES = [("within language", {"full": 0.76, "J": 0.23}, "J converges ~3x slower"),
          ("across modalities", {"full": 0.11, "J": 0.12}, "J and full converge equally")]

fig = plt.figure(figsize=(7.4, 4.55), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], width_ratios=[1.16, 1.0],
                      left=0.012, right=0.988, top=0.86, bottom=0.055,
                      hspace=0.46, wspace=0.10)
REND = fig.canvas.get_renderer()


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(SURFACE)
    return ax


def title(ax, s, dy=0.0):
    ax.text(0.0, 1.10 + dy, s, transform=ax.transAxes, fontsize=FS_TITLE,
            color=INK, fontweight="bold", va="baseline")


def arrow(ax, xy0, xy1, color=MUTED, lw=1.0):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 zorder=3))


def rule(ax, x, y0, y1, color, lw=2.2):
    """Thin coloured accent rule, the box-panels' replacement."""
    ax.plot([x, x], [y0, y1], transform=ax.transAxes, color=color, lw=lw,
            solid_capstyle="butt", zorder=2)


def run(ax, x, y, chunks, fs, family=None):
    inv = ax.transAxes.inverted()
    for text, colour, strike in chunks:
        t = ax.text(x, y, text, transform=ax.transAxes, fontsize=fs,
                    color=colour, family=family, va="center", zorder=4)
        bb = inv.transform(t.get_window_extent(REND))
        if strike:
            ax.plot([bb[0][0], bb[1][0]], [y, y], transform=ax.transAxes,
                    color=colour, lw=0.7, zorder=5)
        x = bb[1][0]
    return x


# ---------------------------------------------------------------- panel (a)
ax = blank(fig.add_subplot(gs[0, :]))
title(ax, "(a)  The J-lens splits a pooled activation", dy=0.02)

ax.text(0.078, 0.755, "a document", ha="center", fontsize=FS_NOTE, color=MUTED)
ax.plot([0.008, 0.148], [0.71, 0.71], transform=ax.transAxes, color=GRID, lw=0.8)
ax.text(0.078, 0.505, "Remember that\nchildren each\nprogress at\ntheir own rate.",
        ha="center", va="center", fontsize=FS_MONO, color=INK, family=MONO,
        linespacing=1.35, style="italic")
ax.plot([0.008, 0.148], [0.30, 0.30], transform=ax.transAxes, color=GRID, lw=0.8)

arrow(ax, (0.158, 0.51), (0.196, 0.51))
ax.plot([0.206, 0.246, 0.246, 0.206, 0.206], [0.30, 0.30, 0.74, 0.74, 0.30],
        transform=ax.transAxes, color=INK2, lw=0.9)
ax.text(0.226, 0.51, "$h_\\ell$", ha="center", va="center", fontsize=11, color=INK)
ax.text(0.226, 0.20, "pooled residual", ha="center", va="top", fontsize=FS_NOTE,
        color=MUTED)

arrow(ax, (0.256, 0.60), (0.300, 0.80), GREEN, 1.0)
arrow(ax, (0.256, 0.42), (0.300, 0.24), ORANGE, 1.0)

rule(ax, 0.316, 0.60, 0.985, GREEN)
ax.text(0.332, 0.905, "J-space", fontsize=FS_LABEL, color=GREEN, fontweight="bold")
ax.text(0.986, 0.905, "18% of the activation", fontsize=FS_NOTE, color=MUTED,
        ha="right")
x = run(ax, 0.332, 0.735, [("reads out as:  ", INK2, False)], FS_NOTE)
run(ax, x + 0.006, 0.735,
    [("children  progress  memory  growth  pace  rate", INK, False)], FS_MONO, MONO)

rule(ax, 0.316, 0.055, 0.415, ORANGE)
ax.text(0.332, 0.375, "non-J remainder", fontsize=FS_LABEL, color=ORANGE,
        fontweight="bold")
ax.text(0.986, 0.375, "82% of the activation", fontsize=FS_NOTE, color=MUTED,
        ha="right")
x = run(ax, 0.332, 0.205, [("everything else:  ", INK2, False)], FS_NOTE)
run(ax, x + 0.006, 0.205,
    [("length  syntax  register  punctuation  formatting", INK, False)],
    FS_MONO, MONO)

# ---------------------------------------------------------------- panel (b)
ax = blank(fig.add_subplot(gs[1, 0]))
title(ax, "(b)  Retention after randomising content")

run(ax, 0.008, 0.90,
    [("Remember", MUTED, True), (" that ", INK, False), ("children", MUTED, True),
     (" each ", INK, False), ("progress", MUTED, True), (" at their own ", INK, False),
     ("rate", MUTED, True), (".", INK, False)], FS_MONO, MONO)
run(ax, 0.008, 0.755,
    [("Reduce", ORANGE, False), (" that ", INK, False), ("self", ORANGE, False),
     (" each ", INK, False), ("rates", ORANGE, False), (" at their own ", INK, False),
     ("special", ORANGE, False), (".", INK, False)], FS_MONO, MONO)
ax.plot([0.008, 0.992], [0.635, 0.635], transform=ax.transAxes, color=GRID, lw=0.8)

X0, W = 0.31, 0.57
for i, (name, val, col, hatched) in enumerate(RET):
    y = 0.475 - i * 0.185
    ax.text(X0 - 0.016, y, name, ha="right", va="center", fontsize=FS_LABEL,
            color=MUTED if hatched else INK)
    ax.add_patch(plt.Rectangle((X0, y - 0.05), W, 0.10, fc="none", ec=GRID, lw=0.8))
    ax.add_patch(plt.Rectangle((X0, y - 0.05), W * val / 100, 0.10,
                               fc="none" if hatched else col,
                               ec=MUTED if hatched else "none",
                               lw=0.8 if hatched else 0,
                               hatch="////" if hatched else None, alpha=1.0))
    ax.text(X0 + W * val / 100 + 0.014, y, f"{val:.1f}%", va="center",
            fontsize=FS_LABEL, color=MUTED if hatched else col,
            fontweight="normal" if hatched else "bold")

# ---------------------------------------------------------------- panel (c)
axo = blank(fig.add_subplot(gs[1, 1]))
title(axo, "(c)  Slope of alignment vs. competence")

lx = 0.0
axo.plot([lx, lx + 0.05], [0.886, 0.886], transform=axo.transAxes, color=BLUE, lw=2.2)
axo.text(lx + 0.065, 0.886, "full activation", transform=axo.transAxes,
         fontsize=FS_NOTE, color=INK2, va="center")
lx = 0.34
axo.plot([lx, lx + 0.05], [0.886, 0.886], transform=axo.transAxes, color=GREEN, lw=2.2)
axo.text(lx + 0.065, 0.886, "J-space", transform=axo.transAxes,
         fontsize=FS_NOTE, color=INK2, va="center")

pos = axo.get_position()
w = (pos.width - 0.028) / 2
for j, (lab, sl, note) in enumerate(SLOPES):
    a = fig.add_axes([pos.x0 + j * (w + 0.028), pos.y0 - 0.01, w,
                      pos.height - 0.1396])
    a.set_facecolor(SURFACE)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)
        a.spines[s].set_linewidth(0.8)
    a.set_xticks([]); a.set_yticks([])
    g = np.linspace(0, 1, 50)
    m = max(sl.values())
    for key, col in (("full", BLUE), ("J", GREEN)):
        a.plot(g, 0.08 + 0.56 * (sl[key] / m) * g, color=col, lw=1.8,
               solid_capstyle="round", zorder=3)
    a.set_ylim(0, 1.14); a.set_xlim(0, 1.28)
    a.text(0.5, 1.245, lab, transform=a.transAxes, ha="center", va="bottom",
           fontsize=FS_LABEL, color=INK)
    a.text(0.5, -0.052, "competence  " + r"$\rightarrow$", transform=a.transAxes,
           ha="center", va="top", fontsize=FS_NOTE, color=MUTED)
    a.text(0.0, 1.077, note, transform=a.transAxes, fontsize=FS_NOTE,
           color=INK, fontweight="bold", va="bottom")
    dy = 0.11 if j else 0.0
    a.text(1.03, 0.08 + 0.56 + dy, f"{sl['full']:+.2f}", color=BLUE,
           fontsize=FS_NOTE, va="center", fontweight="bold")
    a.text(1.03, 0.08 + 0.56 * (sl["J"] / m) - dy, f"{sl['J']:+.2f}",
           color=GREEN, fontsize=FS_NOTE, va="center", fontweight="bold")

fig.savefig("01_teaser.pdf", facecolor=SURFACE)
print("wrote 01_teaser.pdf")
