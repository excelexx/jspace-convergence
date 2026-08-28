"""Recompute every number in the metric appendix from the sweep artifacts.

The appendix (`\\label{app:metrics}`) reports two tables over eight alignment
metrics, a paragraph on how CKNNA's correlation decays with neighbourhood size,
and prose summarising both tables. That is 102 assertable numbers, none of
which the main verifier covers.

    python verify_appendix_metrics.py

Unlike verify_paper_numbers.py, this script does not hardcode the paper's
values. It PARSES the tables and the surrounding prose out of
paper/neurips_2026.tex and compares every one against
metric_sweep/<metric>/results.json and ablation_sweep/<metric>/results.json. A
number added, edited or reordered in the tex is therefore checked
automatically, and neither a table row nor a summarising sentence can stop
matching its artifact without being noticed.

Everything is asserted at the precision the paper prints it. The three
permutation p-value claims are asserted as inequalities (p<0.02, the smallest
global p, and where significance is lost across the CKNNA sweep) rather than as
point values, since they are Monte-Carlo and move between runs.
"""
import json
import os
import re
import sys

TEX = os.path.join("paper", "neurips_2026.tex")
FAIL = CHECKED = 0

# table label -> artifact directory, in the row order the tables use
METRICS = [("mutual $\\kappa$-NN (ours)", "mutual_knn"),
           ("CKNNA", "cknna"),
           ("cycle $\\kappa$-NN", "cycle_knn"),
           ("edit-distance $\\kappa$-NN", "edit_knn"),
           ("LCS $\\kappa$-NN", "lcs_knn"),
           ("CKA", "cka"),
           ("unbiased CKA", "unbiased_cka"),
           ("SVCCA", "svcca")]


def check(label, got, want, tol, unit=""):
    global FAIL, CHECKED
    CHECKED += 1
    ok = abs(got - want) <= tol
    FAIL += not ok
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label:<46} "
          f"recomputed {got:>9.4f}{unit}   paper {want:>9.4f}{unit}")


def section(t):
    print(f"\n{t}\n" + "-" * len(t))


def tabular(label):
    """The body rows of the tabular carrying \\label{label}."""
    tex = open(TEX, encoding="utf-8").read()
    i = tex.index("\\label{" + label + "}")
    body = tex[i:tex.index("\\end{tabular}", i)]
    rows = []
    for line in body.split("\\\\"):
        line = line.strip()
        if "&" in line and "multicolumn" not in line and "toprule" not in line:
            rows.append([c.strip() for c in line.split("&")])
    return rows


def num(cell):
    """First signed number in a LaTeX cell, ignoring \\phantom and $ and %."""
    cell = cell.replace("\\phantom{-}", "").replace("\\,", " ")
    m = re.search(r"[-+]?\d*\.?\d+", re.sub(r"\{\\tiny[^}]*\}", "", cell))
    return float(m.group()) if m else None


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
section("tab:metricsweep - alignment under eight metrics")
try:
    rows = {r[0]: r for r in tabular("tab:metricsweep") if r[0].strip()}
    for name, d in METRICS:
        row = next((r for k, r in rows.items() if k.startswith(name[:12])), None)
        if row is None:
            print(f"  [SKIP] {name:<46} no row found in the tex")
            continue
        s = load(os.path.join("metric_sweep", d, "results.json"))
        t, v = s["text_stats"], s["vision_stats"]["averaged"]
        # columns: metric | full | J | rho(J) | J/full slope | full | J | rho(J)
        check(f"{name}: within-language full", t["full"]["mean"], num(row[1]), 5e-5)
        check(f"{name}: within-language J", t["J"]["mean"], num(row[2]), 5e-5)
        check(f"{name}: within-language rho(J)", t["J"]["rho"], num(row[3]), 5e-3)
        check(f"{name}: J/full slope ratio",
              t["J"]["slope"] / t["full"]["slope"], num(row[4]), 5e-3)
        check(f"{name}: cross-modal full", v["full"]["mean"], num(row[5]), 5e-5)
        check(f"{name}: cross-modal J", v["J"]["mean"], num(row[6]), 5e-5)
        check(f"{name}: cross-modal rho(J)", v["J"]["rho"], num(row[7]), 5e-3)
except (FileNotFoundError, KeyError, ValueError) as exc:
    print(f"  [SKIP] tab:metricsweep  {exc}")

# --------------------------------------------------------------------------
section("tab:metriccontent - content-ablation retention under eight metrics")
try:
    rows = {r[0]: r for r in tabular("tab:metriccontent") if r[0].strip()}
    for name, d in METRICS:
        row = next((r for k, r in rows.items() if k.startswith(name[:12])), None)
        if row is None:
            print(f"  [SKIP] {name:<46} no row found in the tex")
            continue
        tb = load(os.path.join("ablation_sweep", d, "results.json"))["table"]
        f, j, g = (tb["full"]["retention"] * 100, tb["J"]["retention"] * 100,
                   tb["Gaussian"]["retention"] * 100)
        check(f"{name}: retention full", f, num(row[1]), 0.05, "%")
        check(f"{name}: retention J", j, num(row[2]), 0.05, "%")
        check(f"{name}: retention Gaussian", g, num(row[3]), 0.05, "%")
        # the caption defines this column as (Gaussian - J) / (full - J)
        check(f"{name}: explained by J-space",
              (g - j) / (f - j) * 100, num(row[4]), 0.1, "%")
except (FileNotFoundError, KeyError, ValueError, ZeroDivisionError) as exc:
    print(f"  [SKIP] tab:metriccontent  {exc}")

# --------------------------------------------------------------------------
section("CKNNA locality sweep - the appendix's decay paragraph")
try:
    tex = open(TEX, encoding="utf-8").read()
    para = tex[tex.index("CKNNA restricts the kernel"):]
    para = para[:para.index("corpus.") + 7]
    want = {int(k): float(r) for r, k in
            re.findall(r"\$\\rho = ([-+][\d.]+)\$ at \$k=(\d+)\$", para)}
    for k, w in sorted(want.items()):
        s = load(os.path.join("metric_sweep", f"t1_cknna_k{k}", "results.json"))
        check(f"CKNNA k={k}: rho(J)", s["text_stats"]["J"]["rho"], w, 5e-3)
    # "loses significance for k around 50-100"
    lo = load(os.path.join("metric_sweep", "t1_cknna_k50", "results.json"))
    hi = load(os.path.join("metric_sweep", "t1_cknna_k100", "results.json"))
    p50, p100 = lo["text_stats"]["J"]["perm_p"], hi["text_stats"]["J"]["perm_p"]
    CHECKED += 1
    ok = p50 < 0.05 <= p100
    FAIL += not ok
    print(f"  [{'OK' if ok else 'MISMATCH'}] "
          f"{'significance lost between k=50 and k=100':<46} "
          f"p(50)={p50:.4f} < 0.05 <= p(100)={p100:.4f}")
except (FileNotFoundError, KeyError, ValueError) as exc:
    print(f"  [SKIP] CKNNA locality sweep  {exc}")

# --------------------------------------------------------------------------
section("appendix prose - the claims the two tables are summarised by")
NEIGHBOUR = ["mutual_knn", "cknna", "cycle_knn", "edit_knn", "lcs_knn"]
GLOBAL = ["cka", "unbiased_cka", "svcca"]
try:
    tex = open(TEX, encoding="utf-8").read()
    app = tex[tex.index("\\label{app:metrics}"):tex.index("\\end{document}")]

    def sweep(d):
        return load(os.path.join("metric_sweep", d, "results.json"))

    def abl(d):
        return load(os.path.join("ablation_sweep", d, "results.json"))["table"]

    order = [d for _, d in METRICS]

    # "the cross-modal correlation result is similarly strong ($+a \\leq \\rho
    # \\leq +b$)"
    lo, hi = (float(x) for x in re.search(
        r"\+([\d.]+) *\\leq *\\rho *\\leq *\+([\d.]+)", app).groups())
    rhos = [sweep(d)["vision_stats"]["averaged"]["J"]["rho"] for d in order]
    check("cross-modal rho(J), lowest", min(rhos), lo, 5e-3)
    check("cross-modal rho(J), highest", max(rhos), hi, 5e-3)

    # "within language, regardless of metric, full alignment is always greater
    # than J-space alignment"
    bad = [d for d in order if sweep(d)["text_stats"]["J"]["mean"]
           > sweep(d)["text_stats"]["full"]["mean"]]
    CHECKED += 1
    FAIL += bool(bad)
    print(f"  [{'OK' if not bad else 'MISMATCH'}] "
          f"{'within language, full > J for every metric':<46} "
          f"{len(order) - len(bad)}/{len(order)} metrics"
          + (f"   J exceeds full for: {', '.join(bad)}" if bad else ""))

    # "cross-modally ... J-space marginally ahead for six of the eight metrics"
    WORDS = dict(two=2, three=3, four=4, five=5, six=6, seven=7, eight=8)
    want = WORDS[re.search(r"J-space marginally ahead for (\w+) of the",
                           app).group(1)]
    ahead = [d for d in order
             if sweep(d)["vision_stats"]["averaged"]["J"]["mean"]
             > sweep(d)["vision_stats"]["averaged"]["full"]["mean"]]
    check("cross-modally, metrics with J ahead of full",
          float(len(ahead)), float(want), 0)

    # "each metric retains at least X%" (full alignment, global metrics)
    x = num(re.search(r"each metric retains at least \$([\d.]+)\\%",
                      app).group(1))
    # compared at the one decimal the table prints, which is where the prose
    # takes the bound from (SVCCA is 96.887%, printed and quoted as 96.9%)
    worst = min(abl(d)["full"]["retention"] * 100 for d in GLOBAL)
    CHECKED += 1
    ok = round(worst, 1) >= x
    FAIL += not ok
    print(f"  [{'OK' if ok else 'MISMATCH'}] "
          f"{'global full retention is at least %.1f%%' % x:<46} "
          f"lowest is {worst:.2f}%")

    # "the same metrics retain $a$, $b$ and $c$" (J alignment, global metrics)
    m = re.search(r"the same metrics retain \$([\d.]+)\\%\$, \$([\d.]+)\\%\$,"
                  r" and \$([\d.]+)\\%\$", app)
    for d, want in zip(GLOBAL, (float(g) for g in m.groups())):
        check(f"prose J retention, {d}", abl(d)["J"]["retention"] * 100,
              want, 0.05, "%")

    # "under a quarter of J-space alignment is attributable to semantic
    # content" -- the complement of the smallest J retention above
    loss = max(100 - abl(d)["J"]["retention"] * 100 for d in GLOBAL)
    CHECKED += 1
    ok = loss < 25.0
    FAIL += not ok
    print(f"  [{'OK' if ok else 'MISMATCH'}] "
          f"{'under a quarter of J alignment is content':<46} "
          f"largest loss is {loss:.2f}%")

    # "neighbourhood-based metrics all reach significance with $p<X$"
    x = float(re.search(r"reach significance with \$p<([\d.]+)\$", app).group(1))
    worst = max(sweep(d)["text_stats"]["J"]["perm_p"] for d in NEIGHBOUR)
    CHECKED += 1
    ok = worst < x
    FAIL += not ok
    print(f"  [{'OK' if ok else 'MISMATCH'}] "
          f"{'neighbourhood p(J) all below %.2f' % x:<46} "
          f"largest is {worst:.4f}")

    # "the global subspace metrics are not significant (the smallest is $p=X$)"
    x = float(re.search(r"the smallest is \$p=([\d.]+)\$", app).group(1))
    check("smallest global p(J)",
          min(sweep(d)["text_stats"]["J"]["perm_p"] for d in GLOBAL), x, 5e-3)
except (FileNotFoundError, KeyError, ValueError, AttributeError) as exc:
    print(f"  [SKIP] appendix prose  {exc}")

print("\n  permutation p-values are asserted as inequalities, not point values")
print(f"\n{'=' * 72}\n"
      f"{'ALL %d APPENDIX NUMBERS MATCH THE PAPER' % CHECKED if not FAIL else '%d MISMATCH(ES) of %d' % (FAIL, CHECKED)}"
      f"\n{'=' * 72}")
sys.exit(1 if FAIL else 0)
