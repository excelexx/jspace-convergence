# Content ablation and random-dictionary null, across eight metrics

The paper's Table 1 recomputed under every metric in the sweep. Everything upstream of the metric is the paper's: same documents, same content-ablated corpus (`../acts_surr_*.pt`), same k=25 decomposition, same preprocessing, same band x band grid mean, same 55 pairs. Retention is the mean of per-pair ablated/real ratios.

The real-corpus J-lens columns are copied from the completed 8-metric sweep and were validated against a fresh recomputation in this folder (exact match). The ablated columns and both Gaussian-dictionary columns are computed here. The Gaussian dictionary is seeded per (model, layer) so the real and ablated corpora share it, following `../xrandnull.py`.

## Validation

`mutual_knn` is the paper's own metric and must reproduce Table 1:

| row | real (here / paper) | ablated (here / paper) | retention (here / paper) |
|---|---|---|---|
| full | 0.5577 / 0.5577 | 0.3009 / 0.3009 | 54.1% / 54.1% |
| non-J | 0.5057 / 0.5057 | 0.2609 / 0.2609 | 52.0% / 52.0% |
| J | 0.4202 / 0.4202 | 0.1099 / 0.1099 | 26.3% / 26.3% |
| Gaussian | 0.2103 / 0.2103 | 0.0719 / 0.0719 | 34.7% / 34.7% |

## Retention by metric

Percentage of alignment surviving content ablation. The paper's claim is that J-space retention (26.3%) is far below full (54.1%), and that the Gaussian control (34.7%) accounts for only part of that gap.

| metric | full | non-J | J | Gaussian | J vs full | J attributable to J-space |
|---|---|---|---|---|---|---|
| mutual kNN (paper) | 54.1% | 52.0% | 26.3% | 34.7% | +27.7 pp | 30% |
| CKNNA | 49.8% | 46.7% | 20.2% | 19.3% | +29.6 pp | -3% |
| CKA | 100.8% | 103.8% | 88.1% | 82.7% | +12.7 pp | -42% |
| unbiased CKA | 101.4% | 105.1% | 85.4% | 117.9% | +16.0 pp | 203% |
| cycle kNN | 64.4% | 62.1% | 45.0% | 44.7% | +19.4 pp | -2% |
| edit-distance kNN | 37.7% | 37.4% | 18.5% | 27.7% | +19.2 pp | 48% |
| LCS kNN | 57.0% | 55.6% | 31.5% | 39.3% | +25.5 pp | 31% |
| SVCCA | 96.9% | 97.6% | 77.6% | 79.2% | +19.3 pp | 9% |

The last column is the fraction of the full-to-J retention drop that survives the Gaussian control, i.e. the part attributable to the J-space rather than to sparse coding. The paper reports about one third.

## Alignment levels

### mutual kNN (paper)

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.5577 | 0.3009 | 54.1% |
| non-J | 0.5057 | 0.2609 | 52.0% |
| J | 0.4202 | 0.1099 | 26.3% |
| Gaussian | 0.2103 | 0.0719 | 34.7% |

### CKNNA

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.5975 | 0.2962 | 49.8% |
| non-J | 0.5505 | 0.2536 | 46.7% |
| J | 0.4615 | 0.0918 | 20.2% |
| Gaussian | 0.2729 | 0.0521 | 19.3% |

### CKA

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.8036 | 0.8072 | 100.8% |
| non-J | 0.7636 | 0.7870 | 103.8% |
| J | 0.6892 | 0.6054 | 88.1% |
| Gaussian | 0.5928 | 0.4797 | 82.7% |

### unbiased CKA

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.7938 | 0.8019 | 101.4% |
| non-J | 0.7469 | 0.7796 | 105.1% |
| J | 0.6615 | 0.5631 | 85.4% |
| Gaussian | 0.3441 | 0.3889 | 117.9% |

### cycle kNN

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.9358 | 0.6027 | 64.4% |
| non-J | 0.9156 | 0.5682 | 62.1% |
| J | 0.9122 | 0.4103 | 45.0% |
| Gaussian | 0.7304 | 0.3253 | 44.7% |

### edit-distance kNN

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.1609 | 0.0593 | 37.7% |
| non-J | 0.1372 | 0.0493 | 37.4% |
| J | 0.0936 | 0.0169 | 18.5% |
| Gaussian | 0.0379 | 0.0103 | 27.7% |

### LCS kNN

| row | real | ablated | retention |
|---|---|---|---|
| full | 3.5457 | 2.0129 | 57.0% |
| non-J | 3.2298 | 1.7784 | 55.6% |
| J | 2.6809 | 0.8400 | 31.5% |
| Gaussian | 1.4887 | 0.5790 | 39.3% |

### SVCCA

| row | real | ablated | retention |
|---|---|---|---|
| full | 0.6846 | 0.6636 | 96.9% |
| non-J | 0.6725 | 0.6564 | 97.6% |
| J | 0.5869 | 0.4549 | 77.6% |
| Gaussian | 0.4384 | 0.3439 | 79.2% |

Absolute alignment values are not comparable across metrics (LCS kNN is a count out of 10, CKA a normalised ratio, cycle kNN an accuracy). Retention percentages are comparable, since they are ratios within a metric.

