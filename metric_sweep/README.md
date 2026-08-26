# Metric sweep — alignment under alternative similarity measures

Self-contained investigation of the paper's fifth limitation:

> *Fifth, we test only the mutual κ-nearest-neighbour measure of kernel
> alignment. Other methods, such as CKA, SVCCA, and CKNNA, may produce
> different results.*

This folder recomputes **both** of the paper's alignment experiments under
eight different similarity metrics, holding every other step fixed.

**Nothing outside this folder is modified.** The sweep reads (read-only) the
paper's cached activations, prefitted lenses and benchmark scores from the
parent repo, and writes only inside `metric_sweep/`.

## What is held identical to the paper

| step | source |
|---|---|
| text activations | `../acts_{model}.pt` — 1,000 pile-10k docs, mean-pooled, all band layers |
| caption activations | `../cache/text_acts/{model}_L{L}_pool.npy` — 1,024 WIT captions, 5 band layers |
| image activations | `../cache/vision/{enc}/eval_acts_L{L}.npy` — 1,024 images, 6 layers |
| J-lenses | `../lenses/` (prefitted, per layer) |
| decomposition | k=25 non-negative OMP + projected-gradient NNLS against the vocab-mean-centred `W_U·diag(w)·J_L` dictionary, 0.95-quantile pre-clamp |
| text/caption preprocessing | 0.95-quantile clamp + unit row norm (`prep`) |
| image preprocessing | per-coordinate 0.5/99.5 clip + unit row norm (`../xkernels.py`) |
| aggregation | mean over the band × band layer-pair grid, per model pair |
| competence | HellaSwag `acc_norm` from `../results/lmeval/`, model-label permutation null |

**Only the function that scores a single layer pair varies.**

## Metrics

Ported from [`minyoungg/platonic-rep`](https://github.com/minyoungg/platonic-rep)
`metrics.py` and verified against it by `verify_metrics.py`:

| folder | metric | notes |
|---|---|---|
| `mutual_knn/` | mutual kNN | the paper's own metric — validation baseline |
| `cknna/` | CKNNA | neighbour-masked CKA; the metric the Platonic paper uses |
| `cka/` | CKA | biased HSIC (classic CKA) |
| `unbiased_cka/` | unbiased CKA | unbiased HSIC estimator (Song et al. 2012) |
| `cycle_knn/` | cycle kNN | cycle-consistency accuracy through both neighbour graphs |
| `edit_knn/` | edit-distance kNN | `1 − Levenshtein(nn_A, nn_B)/topk` on ordered neighbour lists |
| `lcs_knn/` | LCS kNN | longest common subsequence of the ordered neighbour lists |
| `svcca/` | SVCCA | top-10 SVD basis + CCA, mean canonical correlation |

kNN-family metrics use `topk=10` (the paper's κ=10); SVCCA uses `cca_dim=10`.

## Layout

```
_features/          shared, metric-independent cache (Gram matrices + SVCCA bases)
common.py           decomposition + metric implementations
scorer.py           runs one metric over both experiments
build_features.py   builds _features/ (idempotent)
verify_metrics.py   ports vs reference implementations
run_all.py          driver: every metric, resumable, fault-isolated
make_summary.py     builds SUMMARY.md
SUMMARY.md          <- the findings table
<metric>/results.json, run.log
```

`_features/` is shared by every metric folder rather than duplicated: it holds
only the paper's decomposition, which is *metric-independent by construction*,
so sharing it cannot leak one metric's choices into another. Re-deriving it
eight times would cost ~40 GPU-minutes and change nothing.

## Running

```bash
../venv/Scripts/python.exe verify_metrics.py     # port checks
../venv/Scripts/python.exe build_features.py     # idempotent cache build
../venv/Scripts/python.exe run_all.py            # all 8 metrics + summary
../venv/Scripts/python.exe make_summary.py       # rebuild SUMMARY.md only
```

Every stage is idempotent — a completed metric is skipped on re-run, so the
sweep can be interrupted and resumed.

## Reference numbers (paper, mutual kNN)

Text–text, 55 pairs: full 0.5577 (ρ +0.77), J 0.4202 (ρ +0.56),
non-J 0.5057 (ρ +0.87); slopes full +0.76 ± 0.09, J +0.23 ± 0.07 (ratio 0.30).
Cross-modal J means 0.0535–0.0865 per encoder, ρ +0.95…+0.97.
