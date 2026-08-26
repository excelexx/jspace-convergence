"""Build SUMMARY.md: the paper's Table 1 recomputed under all eight metrics."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

ORDER = ["mutual_knn", "cknna", "cka", "unbiased_cka",
         "cycle_knn", "edit_knn", "lcs_knn", "svcca"]
LABEL = {"mutual_knn": "mutual kNN (paper)", "cknna": "CKNNA", "cka": "CKA",
         "unbiased_cka": "unbiased CKA", "cycle_knn": "cycle kNN",
         "edit_knn": "edit-distance kNN", "lcs_knn": "LCS kNN",
         "svcca": "SVCCA"}
ROWS = ["full", "non-J", "J", "Gaussian"]
PAPER = {"full": (0.5577, 0.3009, 54.1), "non-J": (0.5057, 0.2609, 52.0),
         "J": (0.4202, 0.1099, 26.3), "Gaussian": (0.2103, 0.0719, 34.7)}


def load(m):
    p = os.path.join(m, "results.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d if d.get("complete") else None


def main():
    got = {m: load(m) for m in ORDER}
    got = {m: d for m, d in got.items() if d}
    if not got:
        print("no completed metrics yet")
        return

    L = []
    L.append("# Content ablation and random-dictionary null, across eight metrics")
    L.append("")
    L.append("The paper's Table 1 recomputed under every metric in the sweep. "
             "Everything upstream of the metric is the paper's: same documents, "
             "same content-ablated corpus (`../acts_surr_*.pt`), same k=25 "
             "decomposition, same preprocessing, same band x band grid mean, "
             "same 55 pairs. Retention is the mean of per-pair "
             "ablated/real ratios.")
    L.append("")
    L.append("The real-corpus J-lens columns are copied from the completed "
             "8-metric sweep and were validated against a fresh recomputation "
             "in this folder (exact match). The ablated columns and both "
             "Gaussian-dictionary columns are computed here. The Gaussian "
             "dictionary is seeded per (model, layer) so the real and ablated "
             "corpora share it, following `../xrandnull.py`.")
    L.append("")

    mk = got.get("mutual_knn")
    if mk:
        L.append("## Validation")
        L.append("")
        L.append("`mutual_knn` is the paper's own metric and must reproduce "
                 "Table 1:")
        L.append("")
        L.append("| row | real (here / paper) | ablated (here / paper) "
                 "| retention (here / paper) |")
        L.append("|---|---|---|---|")
        for r in ROWS:
            t = mk["table"][r]
            pr, pa, pret = PAPER[r]
            L.append(f"| {r} | {t['real']:.4f} / {pr:.4f} "
                     f"| {t['ablated']:.4f} / {pa:.4f} "
                     f"| {100*t['retention']:.1f}% / {pret:.1f}% |")
        L.append("")

    L.append("## Retention by metric")
    L.append("")
    L.append("Percentage of alignment surviving content ablation. The paper's "
             "claim is that J-space retention (26.3%) is far below full "
             "(54.1%), and that the Gaussian control (34.7%) accounts for only "
             "part of that gap.")
    L.append("")
    L.append("| metric | full | non-J | J | Gaussian | J vs full | "
             "J attributable to J-space |")
    L.append("|---|---|---|---|---|---|---|")
    for m in ORDER:
        d = got.get(m)
        if not d:
            continue
        t = d["table"]
        ret = {r: 100 * t[r]["retention"] for r in ROWS}
        gap = ret["full"] - ret["J"]
        # share of the full->J retention drop NOT explained by sparse coding
        share = ((ret["Gaussian"] - ret["J"]) / gap * 100) if abs(gap) > 1e-9 \
            else float("nan")
        L.append(f"| {LABEL[m]} | {ret['full']:.1f}% | {ret['non-J']:.1f}% "
                 f"| {ret['J']:.1f}% | {ret['Gaussian']:.1f}% "
                 f"| {ret['full'] - ret['J']:+.1f} pp | {share:.0f}% |")
    L.append("")
    L.append("The last column is the fraction of the full-to-J retention drop "
             "that survives the Gaussian control, i.e. the part attributable "
             "to the J-space rather than to sparse coding. The paper reports "
             "about one third.")
    L.append("")

    L.append("## Alignment levels")
    L.append("")
    for m in ORDER:
        d = got.get(m)
        if not d:
            continue
        L.append(f"### {LABEL[m]}")
        L.append("")
        L.append("| row | real | ablated | retention |")
        L.append("|---|---|---|---|")
        for r in ROWS:
            t = d["table"][r]
            L.append(f"| {r} | {t['real']:.4f} | {t['ablated']:.4f} "
                     f"| {100*t['retention']:.1f}% |")
        L.append("")

    L.append("Absolute alignment values are not comparable across metrics "
             "(LCS kNN is a count out of 10, CKA a normalised ratio, cycle kNN "
             "an accuracy). Retention percentages are comparable, since they "
             "are ratios within a metric.")
    L.append("")

    open("SUMMARY.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote SUMMARY.md ({len(got)} metrics)")


if __name__ == "__main__":
    main()
