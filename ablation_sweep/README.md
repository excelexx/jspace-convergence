# Content ablation and random-dictionary null, across eight metrics

Self-contained rerun of the paper's **Table 1** — the content-ablation
motivating experiment and the Gaussian random-dictionary control — under all
eight alignment metrics rather than mutual κ-NN alone.

**This folder writes nothing outside itself.** It reads the parent repo's
activations, lenses and completed sweep results read-only, and all outputs
(feature cache, per-metric results, summary) stay in `ablation_sweep/`.

## What it computes

For each of the 55 model pairs, four rows of the paper's Table 1:

| row | dictionary | corpora |
|---|---|---|
| full | J-lens | real vs content-ablated |
| non-J | J-lens | real vs content-ablated |
| J | J-lens | real vs content-ablated |
| Gaussian | random Gaussian | real vs content-ablated |

Retention = mean of per-pair ablated/real ratios (the paper's convention, not
the ratio of means).

## What is copied vs recomputed

The **real-corpus J-lens columns** (full / J / non-J) were already produced by
the 8-metric sweep over the same 55 pairs, the same band × band grid and the
same preprocessing, so they are copied once into `real_reference.json` by
`copy_real.py`. This was validated by rebuilding those columns from scratch for
gpt2 and pythia70m and confirming an exact match (`--also_real_jlens`).

Everything else is computed here:

- ablated corpus, J-lens dictionary → full / J / non-J
- real corpus, Gaussian dictionary → J
- ablated corpus, Gaussian dictionary → J

The Gaussian dictionary follows `../xrandnull.py` exactly: `randn(V, d)` with
unit-norm rows, seeded from `md5("rand|{model}|{layer}")`, so the **same**
random dictionary serves both corpora — which is what makes their retention
comparable.

## Held identical to the paper

Same documents (1,000 pile-10k), same content-ablated corpus
(`../acts_surr_*.pt`), same prefitted J-lenses, same k=25 non-negative OMP +
NNLS decomposition against the vocab-mean-centred `W_U·diag(w)·J_L` dictionary,
same 0.95-quantile clamp and unit normalisation, same mean over the band × band
layer-pair grid. Only the metric varies.

## Layout

```
copy_real.py        pulls the real-corpus columns in, once
build_features.py   builds the ablated + Gaussian Gram/SVCCA cache (idempotent)
scorer.py           scores one metric over all four rows
run_all.py          driver: all eight metrics, resumable, PID-locked
make_summary.py     builds SUMMARY.md
_features/{real,abl}/   Gram matrices + SVCCA bases
<metric>/results.json, run.log
```

## Running

```bash
../venv/Scripts/python.exe copy_real.py
../venv/Scripts/python.exe build_features.py
../venv/Scripts/python.exe run_all.py
```

Every stage is idempotent and resumable. `run_all.py` refuses to start if
another instance is live (two concurrent drivers once contended for the GPU and
made everything ~7× slower).

## Reference (paper Table 1, mutual κ-NN)

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.5577 | 0.3009 | 54.1% |
| non-J | 0.5057 | 0.2609 | 52.0% |
| J | 0.4202 | 0.1099 | 26.3% |
| Gaussian | 0.2103 | 0.0719 | 34.7% |

`mutual_knn` runs first in the sweep precisely so this can be checked.
