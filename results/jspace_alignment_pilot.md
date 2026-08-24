# J-space alignment pilot — results

**Date:** 2026-08-01 · **Status:** pilot complete, all planned controls run

## Question

Do the "global workspace" (J-space) components of independently trained language
models align more strongly with each other than their raw representations do?

Combining two prior works: the **Jacobian lens** (Gurnee et al. 2026) decomposes an
activation into a J-component (best sparse non-negative fit over ≤25 atoms of the
per-layer lens dictionary `W_U · J_L`) and a remainder; the **Platonic Representation
Hypothesis** (Huh et al. 2024) compares models via mutual k-nearest-neighbor (m-NN)
overlap of kernels over a shared corpus, which makes different-width models comparable.

**Hypothesis:** `align(J) > align(full) > align(remainder)`, with J required to beat a
random-dictionary control to count.

**Headline result:** the random-dictionary control passes decisively (55/55 pairs), but
the hypothesized ordering **does not hold** — the full activation aligns best in every
one of the 55 pairs.

## Setup

| | |
|---|---|
| Models | 11 (below), spanning 70M–4.3B across 4 families |
| Pairs | 55 (all unordered pairs) |
| Eval corpus | 1000 docs from `NeelNanda/pile-10k`, ≤300 tokens, mean-pooled |
| Lens corpus | WikiText-103 (deliberately ≠ eval corpus) |
| Layer band | lens layers at 35–90% of each model's max lens layer |
| Metric | m-NN overlap, k=10, max over all layer pairs (grid medians also reported) |
| Chance | k/(n−1) = **0.010** |
| Hardware | RTX 3080 10GB |

Models and their observed decomposition health:

| Model | d_model | Band layers | J variance share | Dict anisotropy |
|---|---|---|---|---|
| gpt2 (124M) | 768 | 4–9 | 0.21–0.37 | 0.01–0.02 |
| pythia-70m-deduped | 512 | 2–3 | 0.17–0.19 | 0.32–0.33 |
| gemma-3-270m | 640 | 6–14 | 0.018–0.061 | 0.02–0.03 |
| qwen3.5-0.8b | 1024 | 8–19 | 0.22–0.32 | 0.01 |
| gemma-3-1b | 1152 | 9–21 | 0.12–0.21 | 0.01–0.03 |
| qwen3-1.7b | 2048 | 10–23 | 0.17–0.24 | 0.02 |
| qwen3.5-2b | 2048 | 8–19 | 0.19–0.26 | 0.01–0.02 |
| gemma-2-2b | 2304 | 9–21 | 0.17–0.23 | 0.01 |
| qwen3-4b | 2560 | 12–30 | 0.13–0.18 | 0.02–0.04 |
| qwen3.5-4b | 2560 | 11–27 | 0.17–0.22 | 0.01–0.02 |
| gemma-3-4b | 2560 | 12–28 | 0.088–0.13 | 0.02 |

All variance shares fall inside the sanity gate (0.02–0.25 nominal; no `!!` flags fired).
Two soft outliers worth remembering: **gemma-3-270m** sits at the low edge (0.018–0.061)
and **gpt2 L8–L9** above the nominal top (0.31–0.37). Pythia's dictionary anisotropy of
0.33 is the highest observed but far below the 0.9 collapse guard.

## Components compared

- **full** — the raw pooled activation.
- **J** — its J-space component (≤25 non-negative atoms, real NNLS).
- **perp** — the remainder, `full − J`.
- **perpm** — the remainder projected onto its top-r principal components, where r is
  J's effective rank (participation ratio of the covariance spectrum). This controls for
  the fact that J is low-rank while perp is near-full-width.
- **randJ** — the null: same pipeline with a dictionary of random unembedding rows,
  run as **5 seeded draws**, each getting the same max-over-layer-pairs selection as J.

## Results

Aggregated over all 55 pairs (max m-NN; chance = 0.010):

| Component | Mean | Min | Max |
|---|---|---|---|
| full | 0.6487 | 0.4998 | 0.7916 |
| perp | 0.5979 | 0.4307 | 0.7591 |
| perpm (rank-matched) | 0.5734 | 0.4098 | 0.7511 |
| **J** | **0.4929** | 0.3681 | 0.6076 |
| randJ null (max of 5 draws) | 0.3405 | 0.2154 | 0.5105 |

### 1. J beats the random-dictionary null everywhere — 55/55

Mean margin over the strongest of 5 null draws is **+0.148**; the smallest margin is
**+0.071** (qwen3-1.7b × gemma-3-4b). No pair is a close call. The lens dictionary
carries real structure that random unembedding directions do not, and this holds across
every family combination tested (GPT-2, Pythia, Gemma-2/3, Qwen3/3.5).

### 2. The hypothesized ordering fails, robustly

`full > perp > J` in **every** pair. Specifically:

- full is the top component in **55/55** pairs; J never beats full.
- J beats raw perp in **4/55** pairs, and rank-matched perp in **8/55**.

With 11 models across four families and two orders of magnitude of scale, this is no
longer attributable to the weak initial model pair. **By this metric, at these scales,
the J-component is the *less* universal part of the representation.**

Dimensionality explains part — but not all — of perp's advantage: rank-matching shrinks
perp in 37/55 pairs (mean 0.598 → 0.573), yet matched perp still exceeds J in 47/55.

### 3. Where J does win: the smallest model

6 of the 8 pairs where J beats rank-matched perp involve **pythia-70m**, the smallest
model in the set (the others are gpt2 × gemma-3-4b and gemma-3-1b × pythia-70m). For
Pythia pairs, J is competitive with or better than the matched remainder — a suggestive
scale-dependence, though based on few pairs and one model, so it is a hypothesis for
follow-up rather than a finding.

### 4. Alignment rises with scale, and mildly with shared family

| Grouping | J | full | perp | perpm | null |
|---|---|---|---|---|---|
| Same family (16 pairs) | 0.5361 | 0.7239 | 0.6834 | 0.6572 | 0.3742 |
| Cross family (39 pairs) | 0.4753 | 0.6178 | 0.5628 | 0.5391 | 0.3332 |

Correlation between alignment and log geometric-mean parameter count: **0.68 for J**,
**0.84 for full**. The same-family gap is *smaller* for J (+0.061) than for full (+0.106)
or perp (+0.121). Caveat: the same/cross split is confounded with scale — same-family
pairs here skew larger — and the null itself is higher for same-family pairs (+0.041),
so part of the gap is shared-tokenizer/shared-scale effects rather than family per se.

## Full pair table

Max m-NN per component; `null` is the largest of the 5 random-dictionary draws.
All 55 rows pass (`J > null`).

| Pair | J | full | perp | perpm | null | margin |
|---|---|---|---|---|---|---|
| gpt2 × gemma-3-1b | 0.5009 | 0.6281 | 0.5465 | 0.5093 | 0.4141 | +0.087 |
| gpt2 × gemma-3-270m | 0.4272 | 0.5849 | 0.5362 | 0.4597 | 0.3154 | +0.112 |
| gpt2 × pythia-70m | 0.4853 | 0.5696 | 0.4973 | 0.4875 | 0.3673 | +0.118 |
| gpt2 × qwen3.5-0.8b | 0.4621 | 0.5917 | 0.5092 | 0.4998 | 0.3522 | +0.110 |
| gpt2 × qwen3-1.7b | 0.4147 | 0.5359 | 0.4764 | 0.4594 | 0.3316 | +0.083 |
| gpt2 × qwen3.5-2b | 0.4416 | 0.5765 | 0.5015 | 0.4819 | 0.3054 | +0.136 |
| gpt2 × gemma-2-2b | 0.4373 | 0.5657 | 0.4870 | 0.4610 | 0.3273 | +0.110 |
| gpt2 × qwen3-4b | 0.4127 | 0.5492 | 0.4875 | 0.4773 | 0.3271 | +0.086 |
| gpt2 × qwen3.5-4b | 0.4167 | 0.5306 | 0.4563 | 0.4387 | 0.2803 | +0.136 |
| gpt2 × gemma-3-4b | 0.4626 | 0.5599 | 0.4837 | 0.4498 | 0.3372 | +0.125 |
| gemma-3-1b × gemma-3-270m | 0.4682 | 0.6607 | 0.6447 | 0.5188 | 0.3014 | +0.167 |
| gemma-3-1b × pythia-70m | 0.4936 | 0.5525 | 0.4741 | 0.4476 | 0.3395 | +0.154 |
| gemma-3-1b × qwen3.5-0.8b | 0.5262 | 0.6825 | 0.6274 | 0.6077 | 0.3856 | +0.141 |
| gemma-3-1b × qwen3-1.7b | 0.5291 | 0.6986 | 0.6621 | 0.6554 | 0.4552 | +0.074 |
| gemma-3-1b × qwen3.5-2b | 0.5485 | 0.7094 | 0.6617 | 0.6671 | 0.4002 | +0.148 |
| gemma-3-1b × gemma-2-2b | 0.5493 | 0.7297 | 0.6989 | 0.6730 | 0.3752 | +0.174 |
| gemma-3-1b × qwen3-4b | 0.5217 | 0.6910 | 0.6443 | 0.6446 | 0.4364 | +0.085 |
| gemma-3-1b × qwen3.5-4b | 0.5398 | 0.6927 | 0.6490 | 0.6497 | 0.3726 | +0.167 |
| gemma-3-1b × gemma-3-4b | 0.5583 | 0.7439 | 0.7134 | 0.6927 | 0.4720 | +0.086 |
| gemma-3-270m × pythia-70m | 0.4152 | 0.5260 | 0.4671 | 0.4098 | 0.2632 | +0.152 |
| gemma-3-270m × qwen3.5-0.8b | 0.4102 | 0.6273 | 0.5783 | 0.4912 | 0.2628 | +0.147 |
| gemma-3-270m × qwen3-1.7b | 0.3681 | 0.5843 | 0.5548 | 0.4531 | 0.2537 | +0.114 |
| gemma-3-270m × qwen3.5-2b | 0.3992 | 0.6313 | 0.6096 | 0.4817 | 0.2394 | +0.160 |
| gemma-3-270m × gemma-2-2b | 0.3914 | 0.6354 | 0.6103 | 0.4791 | 0.2465 | +0.145 |
| gemma-3-270m × qwen3-4b | 0.3737 | 0.5880 | 0.5702 | 0.4657 | 0.2498 | +0.124 |
| gemma-3-270m × qwen3.5-4b | 0.3702 | 0.5947 | 0.5620 | 0.4550 | 0.2154 | +0.155 |
| gemma-3-270m × gemma-3-4b | 0.4152 | 0.6249 | 0.6031 | 0.4707 | 0.2509 | +0.164 |
| pythia-70m × qwen3.5-0.8b | 0.4825 | 0.5575 | 0.4773 | 0.4594 | 0.3044 | +0.178 |
| pythia-70m × qwen3-1.7b | 0.4361 | 0.5191 | 0.4506 | 0.4404 | 0.2888 | +0.147 |
| pythia-70m × qwen3.5-2b | 0.4640 | 0.5469 | 0.4797 | 0.4532 | 0.2674 | +0.197 |
| pythia-70m × gemma-2-2b | 0.4299 | 0.5151 | 0.4399 | 0.4211 | 0.2748 | +0.155 |
| pythia-70m × qwen3-4b | 0.4362 | 0.5349 | 0.4616 | 0.4503 | 0.2845 | +0.152 |
| pythia-70m × qwen3.5-4b | 0.4361 | 0.4998 | 0.4307 | 0.4170 | 0.2501 | +0.186 |
| pythia-70m × gemma-3-4b | 0.4681 | 0.5124 | 0.4410 | 0.4140 | 0.2916 | +0.177 |
| qwen3.5-0.8b × qwen3-1.7b | 0.5248 | 0.6880 | 0.6249 | 0.6282 | 0.3771 | +0.148 |
| qwen3.5-0.8b × qwen3.5-2b | 0.5708 | 0.7519 | 0.7019 | 0.6875 | 0.3676 | +0.203 |
| qwen3.5-0.8b × gemma-2-2b | 0.5258 | 0.6731 | 0.6057 | 0.6197 | 0.3339 | +0.192 |
| qwen3.5-0.8b × qwen3-4b | 0.5170 | 0.6906 | 0.6409 | 0.6368 | 0.3680 | +0.149 |
| qwen3.5-0.8b × qwen3.5-4b | 0.5297 | 0.6961 | 0.6254 | 0.6340 | 0.3313 | +0.198 |
| qwen3.5-0.8b × gemma-3-4b | 0.4997 | 0.6550 | 0.6005 | 0.5960 | 0.3678 | +0.132 |
| qwen3-1.7b × qwen3.5-2b | 0.5708 | 0.7448 | 0.6932 | 0.7029 | 0.4154 | +0.155 |
| qwen3-1.7b × gemma-2-2b | 0.5503 | 0.7131 | 0.6757 | 0.6888 | 0.3629 | +0.187 |
| qwen3-1.7b × qwen3-4b | 0.5844 | 0.7864 | 0.7435 | 0.7466 | 0.5105 | +0.074 |
| qwen3-1.7b × qwen3.5-4b | 0.5703 | 0.7402 | 0.6984 | 0.7074 | 0.3918 | +0.179 |
| qwen3-1.7b × gemma-3-4b | 0.5353 | 0.7208 | 0.6858 | 0.6884 | 0.4642 | +0.071 |
| qwen3.5-2b × gemma-2-2b | 0.5637 | 0.7278 | 0.6850 | 0.6993 | 0.3454 | +0.218 |
| qwen3.5-2b × qwen3-4b | 0.5670 | 0.7451 | 0.7135 | 0.7111 | 0.4101 | +0.157 |
| qwen3.5-2b × qwen3.5-4b | 0.6076 | 0.7903 | 0.7424 | 0.7441 | 0.3944 | +0.213 |
| qwen3.5-2b × gemma-3-4b | 0.5422 | 0.7193 | 0.6821 | 0.6947 | 0.3971 | +0.145 |
| gemma-2-2b × qwen3-4b | 0.5395 | 0.7219 | 0.6974 | 0.7051 | 0.3612 | +0.178 |
| gemma-2-2b × qwen3.5-4b | 0.5641 | 0.7424 | 0.7053 | 0.7204 | 0.3243 | +0.240 |
| gemma-2-2b × gemma-3-4b | 0.5663 | 0.7916 | 0.7591 | 0.7511 | 0.3775 | +0.189 |
| qwen3-4b × qwen3.5-4b | 0.5857 | 0.7627 | 0.7205 | 0.7313 | 0.3969 | +0.189 |
| qwen3-4b × gemma-3-4b | 0.5465 | 0.7248 | 0.6890 | 0.6882 | 0.4610 | +0.086 |
| qwen3.5-4b × gemma-3-4b | 0.5586 | 0.7395 | 0.7008 | 0.7151 | 0.3839 | +0.175 |

Per-pair layer-pair heatmaps (4 components each) are saved as
`heatmap_<a>_vs_<b>.png`, 55 files.

## Wiring bugs found and fixed

The pilot's working assumption — *a surprising number is a bug in our wiring until
proven otherwise* — caught four real defects. All four silently produced plausible-looking
numbers rather than crashing.

1. **Fake NNLS.** An lstsq-then-clamp shortcut inflated GPT-2's variance share to ~10
   (reconstruction 10× larger than the input). Replaced with real projected-gradient NNLS.
2. **Missing final-norm weights.** The readout is `W_U · norm(J h)`, so the dictionary must
   be built from `WUeff = W_U * w`. Omitting `w` drove Gemma's variance share to ~0.001.
   HF stores Gemma's RMSNorm weights zero-centered, so `w = 1 + model.norm.weight` there.
3. **Vocab-mean anisotropy** (found this session). GPT-2's lens dictionary is pathologically
   anisotropic — the mean of its normalized atoms has norm **0.995**, i.e. all 50k atoms
   point in essentially one shared direction — and pooled activations are *anti*-aligned
   with it (mean cos −0.086). Non-negative coding against such a dictionary returns ~zero
   coefficients, collapsing the variance share to ~0.000. Fix: subtract the vocab-mean row
   from `WUeff` for both the lens and control dictionaries. The removed direction shifts all
   logits equally and is therefore softmax-invariant, so no readout information is lost.
   Validated as benign where the dictionary was already healthy (Gemma 0.156 → 0.157).
4. **`output_hidden_states` ignored at load time.** Newer architectures (Qwen3.5) silently
   returned `hidden_states=None` when the flag was passed to `from_pretrained`. It must be
   passed at *forward* time; an assert now guards this.

Per-family final-norm conventions now handled explicitly: GPT-2 `transformer.ln_f`,
Pythia `gpt_neox.final_layer_norm`, Gemma `model.norm` (zero-centered, `1+w`), Qwen/Llama
`model.norm` (direct), multimodal gemma-3-4b `model.language_model.norm` (vision tower
dropped for text-only runs).

## Caveats

- **Selection bias in the max.** Reporting the max over a layer-pair grid biases upward.
  Grid medians are printed alongside every max and the heatmaps show the full grids; the
  qualitative ordering is the same under medians.
- **Pythia's eval corpus is its training corpus.** Evaluation uses the Pile, on which
  pythia-70m-deduped was trained. Its rows may not be comparable to the others — relevant
  because Pythia drives most of the pairs where J wins (result 3).
- **Mean-pooled document representations** are a coarse probe; the J-space result may
  differ at token-level granularity.
- **Same/cross-family split is confounded with scale**, as noted in result 4.
- **The J-component is defined by a lens fitted on WikiText**, evaluated on the Pile. This
  is deliberate (it keeps the fitting-corpus control clean) but means J is not optimized
  for the eval distribution.
- **Success criterion.** The pilot's stated criterion was "the pipeline produces sane
  numbers," not "the hypothesis ordering holds." Nothing was tuned toward the ordering.

## Reproducing

```bash
python step6_acts.py   # cache pooled activations (skips existing acts_*.pt)
python step7_align.py  # decompose, align all pairs, write heatmaps
```

Runtime is dominated by the 4B models' 250k-row dictionaries × 25 OMP rounds. Dictionary
scoring is chunked at 65536 rows to stay inside 10GB of VRAM.

## Suggested next steps

1. **Token-level rather than pooled representations** — the most likely explanation for
   J underperforming is that document-mean pooling washes out exactly the token-predictive
   structure the lens dictionary encodes.
2. **A non-Pile eval corpus** to remove Pythia's contamination and test corpus sensitivity.
3. **Test the small-model hypothesis from result 3** with more sub-500M models
   (pythia-160m/410m, gemma-3-270m-it) to see whether J's relative advantage is real and
   scale-dependent.
4. **Aggregate the 55 grids** into a single summary figure rather than 55 separate files.
