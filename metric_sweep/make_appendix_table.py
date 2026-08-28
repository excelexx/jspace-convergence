"""Emit the LaTeX appendix table for the metric-robustness sweep.

Prints a self-contained table body to stdout; paste it into the paper (the
paper deliberately does not \\input from this folder, so it stays standalone).
Re-run after more configs finish to regenerate with updated numbers.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

NEIGHBOUR = [
    ("mutual_knn", r"mutual $\kappa$-NN (ours)"),
    ("cknna", "CKNNA"),
    ("cycle_knn", r"cycle $\kappa$-NN"),
    ("edit_knn", r"edit-distance $\kappa$-NN"),
    ("lcs_knn", r"LCS $\kappa$-NN"),
]
GLOBAL = [
    ("cka", "CKA"),
    ("unbiased_cka", "unbiased CKA"),
    ("svcca", "SVCCA"),
]


def fmt_p(p):
    if p is None:
        return ""
    if p < 1e-4:
        return r"\,{\tiny$p<0.0001$}"
    if p < 0.01:                        # keep small p from rounding to 0.000
        return r"\,{\tiny$p=%.4f$}" % p
    return r"\,{\tiny$p=%.3f$}" % p


def row(key, label):
    with open(os.path.join(key, "results.json")) as f:
        d = json.load(f)
    t = d["text_stats"]
    v = d["vision_stats"]["averaged"]
    ratio = t["J"]["slope"] / t["full"]["slope"]
    return (f"    {label} "
            f"& ${t['full']['mean']:.4f}$ & ${t['J']['mean']:.4f}$ "
            f"& ${t['J']['rho']:+.2f}${fmt_p(t['J'].get('perm_p'))} "
            f"& ${ratio:+.2f}$ "
            f"& ${v['full']['mean']:.4f}$ & ${v['J']['mean']:.4f}$ "
            f"& ${v['J']['rho']:+.2f}${fmt_p(v['J'].get('perm_p'))} \\\\")


def main():
    have = lambda k: os.path.exists(os.path.join(k, "results.json"))
    nb = [(k, l) for k, l in NEIGHBOUR if have(k)]
    gl = [(k, l) for k, l in GLOBAL if have(k)]
    if not nb and not gl:
        print("no completed metrics", file=sys.stderr)
        return

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"  \centering")
    out.append(r"  \small")
    # eight columns overflow the text block at the default 6pt column padding
    out.append(r"  \setlength{\tabcolsep}{4pt}")
    out.append(r"  \caption{Alignment-metric robustness. Both experiments are "
               r"repeated with the decomposition, preprocessing and layer-band "
               r"aggregation held fixed, varying only the metric that scores a "
               r"layer pair. $\rho$(J) is the Spearman correlation between "
               r"J-space alignment and mean HellaSwag accuracy; J/full is the "
               r"ratio of OLS slopes of alignment against competence, which the "
               r"main text reports as $0.30$ for the mutual $\kappa$-NN "
               r"measure. $p$-values are model-label permutation values.}")
    out.append(r"  \label{tab:metricsweep}")
    out.append(r"  \begin{tabular}{lccccccc}")
    out.append(r"    \toprule")
    out.append(r"    & \multicolumn{4}{c}{Within language ($n=55$)} "
               r"& \multicolumn{3}{c}{Cross-modal ($n=11$)} \\")
    out.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-8}")
    out.append(r"    Metric & full & J & $\rho$(J) & J/full slope "
               r"& full & J & $\rho$(J) \\")
    out.append(r"    \midrule")
    if nb:
        out.append(r"    \multicolumn{8}{l}{\emph{Neighbourhood-based}} \\")
        out += [row(k, l) for k, l in nb]
    if gl:
        out.append(r"    \midrule")
        out.append(r"    \multicolumn{8}{l}{\emph{Global subspace}} \\")
        out += [row(k, l) for k, l in gl]
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
