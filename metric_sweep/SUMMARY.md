# Alternative alignment metrics: J-space vs full activation space

Every step matches the paper (same activations, same prefitted J-lenses, same k=25 non-negative OMP decomposition, same preprocessing, alignment = mean over the band x band layer-pair grid). **Only the metric that scores a layer pair varies.** Metric implementations are ported from `minyoungg/platonic-rep/metrics.py` and verified against it (`verify_metrics.py`).

- Text-text: 55 model pairs, competence = pair-mean HellaSwag acc_norm, model-label permutation p.
- Text-vision: 44 (text model, encoder) pairs; alignment averaged over the four encoders gives one point per text model (n=11), permutation p over model labels.
- kNN-family metrics use topk=10 (the paper's kappa=10); SVCCA uses cca_dim=10.
- Significance: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` otherwise.

## Pipeline validation

`mutual_knn` is the paper's own metric, so it should reproduce the published numbers exactly. It does:

| component | this sweep | paper | this sweep rho | paper rho |
|---|---|---|---|---|
| full | 0.5577 | 0.5577 | +0.770 | +0.770 |
| J | 0.4202 | 0.4202 | +0.560 | +0.560 |
| perp | 0.5057 | 0.5057 | +0.868 | +0.868 |

## Experiment 1 -- text-text alignment (55 pairs)

| metric | full | J | non-J | rho full | rho J | rho non-J |
|---|---|---|---|---|---|---|
| mutual kNN (paper) | 0.5577 | 0.4202 | 0.5057 | +0.77 *** | +0.56 * | +0.87 *** |
| CKNNA | 0.5975 | 0.4615 | 0.5505 | +0.81 *** | +0.63 * | +0.88 *** |
| CKA | 0.8036 | 0.6892 | 0.7636 | +0.75 ** | +0.09 ns | +0.80 *** |
| unbiased CKA | 0.7938 | 0.6615 | 0.7469 | +0.73 ** | -0.19 ns | +0.78 *** |
| cycle kNN | 0.9358 | 0.9122 | 0.9156 | +0.79 *** | +0.63 * | +0.78 ** |
| edit-distance kNN | 0.1609 | 0.0936 | 0.1372 | +0.78 *** | +0.62 * | +0.87 *** |
| LCS kNN | 3.5457 | 2.6809 | 3.2298 | +0.78 *** | +0.60 * | +0.87 *** |
| SVCCA | 0.6846 | 0.5869 | 0.6725 | +0.63 ** | -0.30 ns | +0.79 *** |

### Convergence rate: does the J-space still converge more slowly than the full space?

OLS slope of alignment against pair-mean HellaSwag. The paper's claim is that the J slope is about a third of the full slope (+0.23 vs +0.76, ratio 0.30).

| metric | slope full | slope J | slope non-J | J/full ratio |
|---|---|---|---|---|
| mutual kNN (paper) | +0.763 ± 0.088 | +0.226 ± 0.073 | +0.922 ± 0.083 | 0.30 |
| CKNNA | +0.863 ± 0.086 | +0.369 ± 0.091 | +1.071 ± 0.084 | 0.43 |
| CKA | +0.675 ± 0.094 | -0.003 ± 0.091 | +0.882 ± 0.104 | -0.00 |
| unbiased CKA | +0.628 ± 0.094 | -0.192 ± 0.088 | +0.795 ± 0.102 | -0.31 |
| cycle kNN | +0.200 ± 0.024 | +0.136 ± 0.037 | +0.303 ± 0.034 | 0.68 |
| edit-distance kNN | +0.423 ± 0.047 | +0.098 ± 0.026 | +0.444 ± 0.039 | 0.23 |
| LCS kNN | +5.028 ± 0.563 | +1.410 ± 0.413 | +5.756 ± 0.511 | 0.28 |
| SVCCA | +0.249 ± 0.050 | -0.137 ± 0.051 | +0.431 ± 0.052 | -0.55 |

### Component ordering

| metric | ordering by mean alignment | J lowest? |
|---|---|---|
| mutual kNN (paper) | full > non-J > J | yes |
| CKNNA | full > non-J > J | yes |
| CKA | full > non-J > J | yes |
| unbiased CKA | full > non-J > J | yes |
| cycle kNN | full > non-J > J | yes |
| edit-distance kNN | full > non-J > J | yes |
| LCS kNN | full > non-J > J | yes |
| SVCCA | full > non-J > J | yes |

## Experiment 2 -- text-vision alignment (44 pairs)

Alignment averaged over the four encoders, one point per text model (n=11).

| metric | full | J | non-J | rho full | rho J | rho non-J |
|---|---|---|---|---|---|---|
| mutual kNN (paper) | 0.0719 | 0.0726 | 0.0660 | +0.93 *** | +0.97 *** | +0.92 *** |
| CKNNA | 0.0761 | 0.0763 | 0.0710 | +0.96 *** | +0.96 *** | +0.92 *** |
| CKA | 0.2001 | 0.2038 | 0.1850 | +0.90 *** | +0.97 *** | +0.90 *** |
| unbiased CKA | 0.1687 | 0.1677 | 0.1500 | +0.86 ** | +0.94 *** | +0.86 ** |
| cycle kNN | 0.3672 | 0.3712 | 0.3448 | +0.93 *** | +0.95 *** | +0.90 *** |
| edit-distance kNN | 0.0106 | 0.0106 | 0.0096 | +0.96 *** | +0.95 *** | +0.92 *** |
| LCS kNN | 0.5712 | 0.5762 | 0.5282 | +0.93 *** | +0.97 *** | +0.92 *** |
| SVCCA | 0.3070 | 0.3029 | 0.2924 | +0.85 ** | +0.94 *** | +0.86 ** |

### Cross-modal convergence rate

The paper's second claim: across modalities the J and full rates become indistinguishable (unlike within language).

| metric | slope full | slope J | J/full ratio |
|---|---|---|---|
| mutual kNN (paper) | +0.1091 ± 0.0115 | +0.1165 ± 0.0118 | 1.07 |
| CKNNA | +0.1182 ± 0.0106 | +0.1385 ± 0.0146 | 1.17 |
| CKA | +0.2889 ± 0.0453 | +0.2527 ± 0.0295 | 0.87 |
| unbiased CKA | +0.2768 ± 0.0440 | +0.2364 ± 0.0288 | 0.85 |
| cycle kNN | +0.3910 ± 0.0554 | +0.3857 ± 0.0433 | 0.99 |
| edit-distance kNN | +0.0178 ± 0.0016 | +0.0186 ± 0.0020 | 1.05 |
| LCS kNN | +0.7889 ± 0.0843 | +0.8175 ± 0.0852 | 1.04 |
| SVCCA | +0.2470 ± 0.0386 | +0.2527 ± 0.0350 | 1.02 |

### Per-encoder J-space correlation with competence

| metric | DINOv2 | MAE | CLIP | SigLIP |
|---|---|---|---|---|
| mutual kNN (paper) | +0.97 *** | +0.95 *** | +0.97 *** | +0.97 *** |
| CKNNA | +0.96 *** | +0.94 *** | +0.96 *** | +0.98 *** |
| CKA | +0.95 *** | +0.97 *** | +0.97 *** | +0.97 *** |
| unbiased CKA | +0.94 *** | +0.95 *** | +0.94 *** | +0.97 *** |
| cycle kNN | +0.92 *** | +0.96 *** | +0.95 *** | +0.95 *** |
| edit-distance kNN | +0.92 *** | +0.94 *** | +0.90 *** | +0.97 *** |
| LCS kNN | +0.95 *** | +0.96 *** | +0.97 *** | +0.97 *** |
| SVCCA | +0.95 *** | +0.95 *** | +0.94 *** | +0.94 *** |

## Headline findings

**1. The J-space is the least-aligned component under every metric tested (8/8).** The paper's `full > non-J > J` ordering is completely metric-independent.

**2. Competence-convergence for the full and non-J components is robust (8/8 metrics, all p<0.05).** No metric disputes that more competent model pairs align more.

**3. Competence-convergence for the J-space specifically is metric-dependent (5/8 metrics reproduce it).**

   Reproduced by: mutual kNN (paper), CKNNA, cycle kNN, edit-distance kNN, LCS kNN.

   **Not** reproduced by: CKA (rho +0.09, p=0.6510), unbiased CKA (rho -0.19, p=0.4751), SVCCA (rho -0.30, p=0.2668).

**4. Where the J-space does converge, it converges more slowly than the full space (5/8 metrics have 0 < J/full slope ratio < 1).** This is the paper's central quantitative claim and it survives the metric change, though the ratio itself ranges over 0.30 (mutual kNN (paper)), 0.43 (CKNNA), 0.68 (cycle kNN), 0.23 (edit-distance kNN), 0.28 (LCS kNN).

**5. The cross-modal result is completely metric-independent (8/8 metrics).** Every metric tested -- including the three that reject within-language J convergence -- finds cross-modal J alignment rising with competence (rho +0.94 to +0.97, all p<0.05), with a J/full slope ratio of 0.85-1.17, i.e. indistinguishable from the full space. **The paper's second claim -- that the J/full gap closes across modalities -- survives every metric.**

### Why the global-geometry metrics disagree

The split is exactly along one line: the five metrics that reproduce the claim all score **local neighbourhood structure** (who is whose nearest neighbour), while the three that reject it -- CKA, unbiased CKA and SVCCA -- all score **global subspace geometry**.

This is not a numerical artifact. CKA's J values have comparable spread to every other metric (sd 0.049, range/sd 5.1). The sign is explained by *which* pairs these metrics rank highest: under CKA the most J-similar model pairs are the **least** competent ones -- gpt2 x pythia70m scores highest of all 55 pairs (0.792) at the lowest competence (0.346). CKA and SVCCA are dominated by the leading principal directions, and small models' J-spaces are low-rank and generic, so they look maximally similar to one another. As competence rises the J-space becomes richer and more model-specific in exactly the coarse geometry these metrics see, cancelling the neighbourhood-level convergence the kNN family measures.

**Practical reading.** The paper's within-language J-space convergence claim is a claim about *local neighbourhood structure*, and is worth stating that way. It is robust across five different neighbourhood measures, and it is not recovered by global subspace measures. That is a substantive scope condition on Contribution 2, not a refutation: the component ordering, the full/non-J convergence, and the entire cross-modal result hold under all eight.

## How to read this

The paper's two load-bearing text-side claims are (a) the J-space converges with competence at all, and (b) it converges markedly more slowly than the full activation space. A metric reproduces the paper if its J/full slope ratio is well below 1 and its J correlation stays positive.

Absolute alignment values are **not** comparable across metrics -- LCS kNN is a count out of 10, CKA is a normalised ratio, cycle kNN is an accuracy. Compare within a column, and compare the ratios and correlations across rows.

