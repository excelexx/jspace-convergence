"""Recompute every headline number in the paper from the shipped result JSONs.

No GPU, no model downloads, no lens files: this reads only `results/` as
shipped in this folder and recomputes each statistic the paper quotes, then
compares against the value printed in the paper. Runs in a few seconds and is
the fastest way for a reader to check that the artifacts and the text agree.

    python verify_paper_numbers.py

Every line prints  OK  (recomputed value matches the paper within tolerance)
or  MISMATCH  (it does not). A `SKIP` line means the number is not
machine-checkable from this folder; see README.md, "Known gaps".

The paper aggregates each model pair by the MEAN over its layer-pair grid and
then takes the mean over pairs; that is the only aggregation the paper reports,
and the only one checked below. `xmeanmain.py` is the script that produces
these numbers for the paper; this file re-derives them independently from the
same artifacts.
"""
import json
import os
import statistics as st
import sys

import numpy as np
from scipy.stats import spearmanr

R = "results"
FAIL = 0

MODELS = ["pythia70m", "gpt2", "gemma270", "qwen08b", "gemma", "qwen17b",
          "qwen2b", "gemma2_2b", "qwen4b", "qwen35_4b", "gemma3_4b"]
ENCODERS = ["dinov2", "mae", "clip", "siglip"]
PARAMS_M = {"pythia70m": 70, "gpt2": 124, "gemma270": 270, "qwen08b": 800,
            "gemma": 1000, "qwen17b": 1700, "qwen2b": 2000, "gemma2_2b": 2600,
            "qwen4b": 4000, "qwen35_4b": 4000, "gemma3_4b": 4300}
COMPS = ["full", "perp", "J"]


def load(path):
    with open(os.path.join(R, path), encoding="utf-8") as fh:
        return json.load(fh)


def check(label, got, want, tol, unit=""):
    global FAIL
    ok = abs(got - want) <= tol
    FAIL += not ok
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label:<50} "
          f"recomputed {got:.4f}{unit}   paper {want:.4f}{unit}")


def skip(label, why):
    print(f"  [SKIP] {label:<50} {why}")


def section(title):
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------
# Table 1 - content ablation, MEAN over grid then MEAN over 55 pairs
# --------------------------------------------------------------------------
section("Table 1 and section 4.1 - content ablation, mean over 55 pairs")
try:
    P = load("randdict_null_bylayerstat.json")["pairs"]
    assert len(P) == 55, f"expected 55 pairs, found {len(P)}"
    # variant "real" = the J-lens dictionary; "rand" = the Gaussian control
    paper = {("real", "full"): (0.5577, 0.3009, 54.1),
             ("real", "perp"): (0.5057, 0.2609, 52.0),
             ("real", "J"):    (0.4202, 0.1099, 26.3),
             ("rand", "J"):    (0.2103, 0.0719, 34.7)}
    for (variant, comp), (r_p, s_p, ret_p) in paper.items():
        name = comp if variant == "real" else "Gaussian"
        r = st.mean(p[variant][comp]["real_mean"] for p in P)
        s = st.mean(p[variant][comp]["surr_mean"] for p in P)
        ret = st.mean(p[variant][comp]["ret_mean"] for p in P) * 100
        check(f"{name}: real alignment", r, r_p, 5e-4)
        check(f"{name}: content-ablated alignment", s, s_p, 5e-4)
        check(f"{name}: retention", ret, ret_p, 0.05, "%")
    # the paper's 69.7% is the ratio of this table's own means
    ret_j = st.mean(p["real"]["J"]["ret_mean"] for p in P) * 100
    ret_g = st.mean(p["rand"]["J"]["ret_mean"] for p in P) * 100
    ret_f = st.mean(p["real"]["full"]["ret_mean"] for p in P) * 100
    check("share of the drop explained by sparse coding",
          (ret_f - ret_g) / (ret_f - ret_j) * 100, 69.7, 0.1, "%")
    check("ablation removes this much of J-space", 100 - ret_j, 74.0, 0.5, "%")
    check("ablation removes this much of full", 100 - ret_f, 46.0, 0.5, "%")
    # chance multiples quoted in 4.1 (chance = 10/999)
    chance = 10 / 999
    check("J alignment as a multiple of chance",
          st.mean(p["real"]["J"]["real_mean"] for p in P) / chance, 42.0, 0.1)
    check("full alignment as a multiple of chance",
          st.mean(p["real"]["full"]["real_mean"] for p in P) / chance, 55.7, 0.1)
except FileNotFoundError as exc:
    skip("Table 1 (all rows)", f"missing {exc.filename}")

# --------------------------------------------------------------------------
# Competence: within-language at the pair level, cross-modal at model level
# --------------------------------------------------------------------------
section("Sections 4.1-4.2 - alignment vs HellaSwag competence")
try:
    hs = {m: load(f"lmeval/{m}.json")["hellaswag_acc_norm"] for m in MODELS}
    P = load("randdict_null_bylayerstat.json")["pairs"]
    # (a) within-language: one point per model PAIR, x = mean of the two
    #     models' HellaSwag accuracy
    y_pair = [(hs[p["a"]] + hs[p["b"]]) / 2 for p in P]
    for comp, rho_p in (("full", 0.77), ("perp", 0.87), ("J", 0.56)):
        x = [p["real"][comp]["real_mean"] for p in P]
        check(f"within-language: rho({comp}, HellaSwag), 55 pairs",
              spearmanr(x, y_pair).statistic, rho_p, 0.005)

    # (b) cross-modal: average the mean-over-grid alignment within each of the
    #     4 encoders, then over encoders -> one point per text model
    B = load("measB.json")

    def grid_mean(model, enc, comp):
        return float(np.mean([float(v) for v in
                              B[model][enc][comp]["grid"].values()]))

    for comp, rho_p in (("full", 0.93), ("perp", 0.92), ("J", 0.97)):
        x = [np.mean([grid_mean(m, e, comp) for e in ENCODERS]) for m in MODELS]
        check(f"cross-modal: rho({comp}, HellaSwag), 11 models",
              spearmanr(x, [hs[m] for m in MODELS]).statistic, rho_p, 0.005)

    # per-encoder mean J alignment; section 4.2 quotes the range as 0.053 (MAE)
    # to 0.087 (SigLIP), and tab:perencoder lists all four
    per_enc = {e: float(np.mean([grid_mean(m, e, "J") for m in MODELS]))
               for e in ENCODERS}
    check("per-encoder mean J alignment, MAE", per_enc["mae"], 0.0535, 5e-4)
    check("per-encoder mean J alignment, SigLIP", per_enc["siglip"], 0.0865, 5e-4)
    check("per-encoder mean J alignment, DINOv2", per_enc["dinov2"], 0.0742, 5e-4)
    check("per-encoder mean J alignment, CLIP", per_enc["clip"], 0.0764, 5e-4)
    chance_x = 10 / 1023
    lo = min(np.mean([grid_mean(m, e, c) for e in ENCODERS for m in MODELS])
             for c in COMPS) / chance_x
    hi = max(np.mean([grid_mean(m, e, c) for e in ENCODERS for m in MODELS])
             for c in COMPS) / chance_x
    check("cross-modal alignment, lowest multiple of chance", lo, 6.8, 0.05)
    check("cross-modal alignment, highest multiple of chance", hi, 7.4, 0.05)
except (FileNotFoundError, KeyError) as exc:
    skip("competence correlations", f"missing {exc}")

skip("p-values on those correlations",
     "model-label permutation, stochastic; rerun xmeanmain.py to reproduce")
skip("convergence-rate slopes (+0.76 vs +0.23 per HellaSwag unit)",
     "fitted by xplot_pairlevel_fig1.py into the Figure 1 legend, not stored")

# --------------------------------------------------------------------------
# Section 4.3 - shared J-lens vocabulary
# --------------------------------------------------------------------------
section("Section 4.3 - J-lens top-25 word agreement, 55 text pairs")
try:
    W = load("wordalign/stats.json")
    T, pairs = W["tests"], W["pairs"]
    dj = st.median(p["median_J_k25"] for p in pairs.values())
    db = st.median(p["median_base_k25"] for p in pairs.values())
    # section 4.3 now reports the matched overlap directly; the shuffled floor
    # is quoted only in the controls subsection
    check("shuffle-corrected J-lens overlap (controls only)", dj * 25, 5.0, 0.05)
    check("shuffle-corrected logit-lens overlap (controls only)", db * 25, 1.9, 0.06)
    check("raw J-lens overlap, of 25",
          st.median(st.median(p["raw_J_k25"]) for p in pairs.values()) * 25,
          5.1, 0.05)
    check("raw logit-lens overlap, of 25",
          st.median(st.median(p["raw_base_k25"]) for p in pairs.values()) * 25,
          2.0, 0.05)
    check("position-shuffled floor, of 25",
          st.median(st.median(p["shuf_J_k25"]) for p in pairs.values()) * 25,
          0.1, 0.02)
    check("J-lens beats the logit lens (of 55)", T["H3_J_gt_base"], 50, 0)
    check("median J-lens minus logit-lens overlap", T["H3_median_diff"],
          0.117, 5e-4)
    # the counterpart-layer offset is measured on the RAW overlap grid, matching
    # section 4.3's switch away from the shuffle-corrected delta
    G = load("wordalign/rawgrid_argmax.json")
    cross = [v for v in G.values() if not v["same_family"]]
    check("cross-family pairs", len(cross), 38, 0)
    check("best-matching counterpart layer, % of depth away",
          st.mean(v["rowargmax_dp_raw"] for v in cross) * 100, 11.1, 0.05, "%")
    check("diagonal > off-diagonal (of 55 pairs)",
          sum(v["diag_mean_raw"] > v["offdiag_mean_raw"] for v in G.values()), 55, 0)
except (FileNotFoundError, KeyError) as exc:
    skip("word-alignment headline", f"missing {exc}")

# --------------------------------------------------------------------------
# Section 4.4 - controls
# --------------------------------------------------------------------------
section("Section 4.4 - controls")
try:
    S = load("control_shuffle.json")
    # The paper quotes the mean over the raw cells, which is what
    # xstage7_controls.py --shuffle now produces. The shipped artifact also
    # carries 40 projected cells from the withdrawn dense-projection pipeline;
    # they are ignored here so the check matches both the paper and a rerun.
    means = [c["raw"]["shuffled_mean"] for c in S.values() if c.get("raw")]
    check("shuffled image-caption pairs, mean alignment",
          sum(means) / len(means), 0.0096, 5e-5)
    check("  cells averaged", len(means), 44, 0)
except FileNotFoundError as exc:
    skip("shuffled-pair null", f"missing {exc.filename}")

try:
    M = load("lenscontrol/median_delta.json")["summary"]["pile"]
    check("lens-fitting control, median-over-grid delta",
          M["median_delta_med"], 0.0002, 1e-4)
    check("  pairs covered", M["n_pairs"], 10, 0)
    check("  alignment level", M["median_alignment_level_med"], 0.42, 5e-3)
    L = load("lenscontrol/phase4_sparse.json")["summary"]["pile"]
    check("  J beats the null in every crossed pair",
          L["J_beats_null_crossed"], 10, 0)
except (FileNotFoundError, KeyError) as exc:
    skip("lens-fitting control", f"missing {exc}")

try:
    K = load("ksweep_mean.json")
    check("degraded-J alignment at matched share", K["matched_mean"], 0.2924, 5e-4)
    check("content-ablated alignment", K["surr_mean"], 0.1099, 5e-4)
    check("five-direction J alignment", K["per_k_mean"]["5"], 0.2326, 5e-4)
    check("five-direction advantage over ablated",
          K["per_k_mean"]["5"] / K["surr_mean"], 2.1, 0.05)
except (FileNotFoundError, KeyError) as exc:
    skip("lens-degradation control", f"missing {exc}")

try:
    H = load("surrogate_55.json")["health"]
    # health is {corpus/model: {layer: {...}}}; the paper's 18.3% is the mean
    # over the 11 models of each model's mean var_share across its band layers
    def share(prefix):
        per_model = [st.mean(l["var_share"] for l in layers.values())
                     for k, layers in H.items() if k.startswith(prefix)]
        return st.mean(per_model) * 100
    check("J share of the squared norm, real text", share("real/"), 18.3, 0.05, "%")
    check("J share of the squared norm, content-ablated",
          share("surrogate/"), 13.5, 0.05, "%")
except (FileNotFoundError, KeyError) as exc:
    skip("J share of squared norm", f"missing {exc}")

try:
    T = load("token_control.json")["per_model"]
    cpt = [v["chars_per_token"] for v in T.values()]
    check("chars per token, most efficient", max(cpt), 4.19, 0.005)
    check("chars per token, least efficient", min(cpt), 3.79, 0.005)
except (FileNotFoundError, KeyError) as exc:
    skip("tokenisation control", f"missing {exc}")

# --------------------------------------------------------------------------
# Appendices - competence tables and per-encoder correlations
# --------------------------------------------------------------------------
section("Appendices - tab:competence, tab:crossaxes, tab:perencoder")
try:
    hs = {m: load(f"lmeval/{m}.json")["hellaswag_acc_norm"] for m in MODELS}
    owt = load("performance_owt.json")
    bpb = {m: owt[m]["performance"] for m in MODELS}
    logp = {m: np.log10(PARAMS_M[m] / 1000) for m in MODELS}
    P = load("randdict_null_bylayerstat.json")["pairs"]
    B = load("measB.json")

    def grid_mean(model, enc, comp):
        return float(np.mean([float(v) for v in
                              B[model][enc][comp]["grid"].values()]))

    # tab:competence - within-language, pair level, vs log parameters
    for comp, rho_p in (("full", 0.79), ("perp", 0.87), ("J", 0.60)):
        x = [p["real"][comp]["real_mean"] for p in P]
        y = [(logp[p["a"]] + logp[p["b"]]) / 2 for p in P]
        check(f"tab:competence log-params, {comp}",
              spearmanr(x, y).statistic, rho_p, 0.005)

    # tab:crossaxes - cross-modal, model level, vs log params and 1-bpb
    for comp, rl, rb in (("full", 0.84, 0.96), ("perp", 0.83, 0.98),
                         ("J", 0.89, 0.94)):
        x = [np.mean([grid_mean(m, e, comp) for e in ENCODERS]) for m in MODELS]
        check(f"tab:crossaxes log-params, {comp}",
              spearmanr(x, [logp[m] for m in MODELS]).statistic, rl, 0.005)
        check(f"tab:crossaxes 1-bpb, {comp}",
              spearmanr(x, [bpb[m] for m in MODELS]).statistic, rb, 0.005)

    # tab:perencoder - one correlation per encoder per component
    want = {"dinov2": (0.93, 0.92, 0.97), "mae": (0.91, 0.92, 0.95),
            "clip": (0.93, 0.92, 0.97), "siglip": (0.93, 0.92, 0.97)}
    for e, (rf, rp, rj) in want.items():
        for comp, r_p in (("full", rf), ("perp", rp), ("J", rj)):
            x = [grid_mean(m, e, comp) for m in MODELS]
            check(f"tab:perencoder {e}, {comp}",
                  spearmanr(x, [hs[m] for m in MODELS]).statistic, r_p, 0.005)

    # appendix B: within-language J vs 1-bpb (no other script computes this)
    x = [p["real"]["J"]["real_mean"] for p in P]
    y = [(bpb[p["a"]] + bpb[p["b"]]) / 2 for p in P]
    check("within-language J vs 1-bpb", spearmanr(x, y).statistic, 0.35, 0.005)
except (FileNotFoundError, KeyError) as exc:
    skip("appendix competence tables", f"missing {exc}")

# --------------------------------------------------------------------------
# Remaining prose numbers
# --------------------------------------------------------------------------
section("Prose numbers in sections 3, 4.3 and the limitations")
try:
    W = load("wordalign/stats.json")
    pairs, hs = W["pairs"], {m: load(f"lmeval/{m}.json")["hellaswag_acc_norm"]
                             for m in MODELS}
    check("J-lens advantage in words, of 25",
          W["tests"]["H3_median_diff"] * 25, 2.9, 0.05)
    # section 4.3: pair-level correlation of word agreement with competence
    y = [(hs[k.split("|")[0]] + hs[k.split("|")[1]]) / 2 for k in pairs]
    check("word agreement vs competence, J-lens",
          spearmanr([v["median_J_k25"] for v in pairs.values()], y).statistic,
          0.73, 0.005)
    # section 4.3 now reports the matched overlap directly, so the middle-depth
    # (p in [0.2, 0.5]) J/logit ratio is taken on the raw lists. The paper
    # states this band as "4--5x".
    idx = [i for i, d in enumerate(W["pgrid"]) if 0.2 <= d <= 0.5]
    # median over pairs, matching how 5.1 and 2.0 are aggregated above
    ratios = [st.median(v["raw_J_k25"][i] for v in pairs.values())
              / st.median(v["raw_base_k25"][i] for v in pairs.values())
              for i in idx]
    lo_r, hi_r = min(ratios), max(ratios)
    check(f"middle-layer J/logit ratios ({lo_r:.2f}-{hi_r:.2f}) inside the 4-5x band",
          float(round(lo_r) >= 4 and round(hi_r) <= 5), 1.0, 0)
    check("word agreement vs competence, logit lens",
          spearmanr([v["median_base_k25"] for v in pairs.values()], y).statistic,
          0.44, 0.005)
except (FileNotFoundError, KeyError) as exc:
    skip("section 4.3 correlations", f"missing {exc}")

try:
    T = load("token_control.json")["per_model"]
    ing = [v["frac_corpus_ingested"] for v in T.values()]
    check("least efficient tokenizer ingests this much of the corpus",
          min(ing) / max(ing) * 100, 94.8, 0.1, "%")
    check("Pythia-70M HellaSwag",
          load("lmeval/pythia70m.json")["hellaswag_acc_norm"] * 100, 30.8, 0.05, "%")
    check("GPT-2 HellaSwag",
          load("lmeval/gpt2.json")["hellaswag_acc_norm"] * 100, 38.5, 0.05, "%")
except (FileNotFoundError, KeyError) as exc:
    skip("prose constants", f"missing {exc}")

try:
    N = load("randunembed_null.json")["aggregate"]
    check("random-unembedding null: pairs beating all 5 draws",
          N["pairs_beating_all_draws"], 55, 0)
    check("random-unembedding null: mean margin",
          N["mean_margin_over_strongest"], 0.167, 5e-4)
except (FileNotFoundError, KeyError) as exc:
    skip("random-unembedding-row null", f"missing {exc}")
skip("~40 GPU hours",
     "not recorded; only the 27.5 h lens-refit chain is measured anywhere")

print(f"\n{'=' * 72}\n"
      f"{'ALL CHECKED NUMBERS MATCH THE PAPER' if not FAIL else str(FAIL) + ' MISMATCH(ES)'}"
      f"\n{'=' * 72}")
sys.exit(1 if FAIL else 0)
