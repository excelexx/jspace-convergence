# Not All Representational Convergence Is Semantic — code and artifacts

### Measuring Alignment in the Jacobian Subspace

The pipelines that produce every number and figure in the paper, the result
artifacts they wrote, and the paper source.

The paper measures representational convergence between models inside the
**J-space** — the subspace induced by the Jacobian lens — instead of the full
activation space, across three experiments:

| # | Experiment | Scale | Section |
|---|---|---|---|
| 1 | Text↔text alignment, plus content ablation | 11 language models, 55 pairs, 1,000 Pile documents | 4.1 |
| 2 | Text↔vision alignment | 4 ViT-B encoders × 11 language models = 44 pairs, 1,024 WIT image–caption pairs | 4.2 |
| 3 | Shared J-lens vocabulary read out as words | 11 models, 55 pairs, 6,000 character offsets | 4.3 |

---

## 1. Check the results without a GPU (30 seconds)

Every headline number in the paper is recomputed from the JSON artifacts
shipped in `results/` and compared against the value printed in the paper:

```bash
python verify_paper_numbers.py
```

It needs only `numpy` and `scipy`. Each line prints `OK` or `MISMATCH`; `SKIP`
marks a number whose source artifact is not in this folder (see
[Known gaps](#8-known-gaps)). The expected output ends with
`ALL CHECKED NUMBERS MATCH THE PAPER`.

It recomputes 84 numbers: Table 1 including its Gaussian-dictionary row, the
within-language and cross-modal competence correlations, the log-parameter and
1−bits-per-byte columns of `tab:competence` and `tab:crossaxes`, all twelve
per-encoder correlations in `tab:perencoder`, the word-agreement headline and
hypotheses H1–H4, the §4.3 competence correlations, the tokenisation control,
and three of the four §4.4 controls. The `SKIP` lines are numbers that cannot
be recomputed from this folder; §8 lists every one of them and says why.

The paper aggregates each model pair by the **mean** over its layer-pair grid.
`xmeanmain.py` produces the paper's tables; the verifier re-derives them
independently from the same artifacts.

## 2. Environment

Python 3.12.10, one NVIDIA GPU with ≥10 GB (everything was run on a single
RTX 3080). CUDA 12.1.

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Exact pinned versions are in `requirements.txt`. Run every script **from this
directory** — all paths are relative to it, and the scripts import each other
as top-level modules.

Correctness gate for the averaged-Jacobian estimator used by the lens-fitting
control (fast, CPU, no downloads):

```bash
python tests/test_jvp_identity.py
```

It checks the constant-tangent JVP identity against an explicit autograd
Jacobian on a toy transformer, bidirectional and causal, to 1e-5. The
half-corpus lens refits in `xlc_phase1.py` are gated on it.

## 3. Assets to download

| Asset | Where it goes | Size | How |
|---|---|---|---|
| J-lenses, 11 models | `lenses/` | ~2.4 GB | `python step2_download.py` |
| 11 language models | HF cache | ~30 GB | pulled on demand by `transformers` |
| 4 vision encoders | HF cache + `models/` | ~1.5 GB | on demand; CLIP needs `xconvert_clip.py` |
| pile-10k, WIT, WikiText-103 | HF cache | ~1 GB | pulled on demand by `datasets` (WIT is streamed) |

The lens files are `dict` with `J[layer]` a `d × d` fp16 matrix. Model ids and
lens paths are the single source of truth in `step7_align.py::MODELS`; vision
encoders are in `xvision_config.py::VISION` (`facebook/dinov2-base`,
`facebook/vit-mae-base`, `openai/clip-vit-base-patch16` converted locally,
`google/siglip-base-patch16-224`).

Gated repos (Gemma) need `huggingface-cli login` with accepted licences.

## 4. Experiment 1 — text↔text alignment and content ablation (section 4.1)

```bash
python step2_download.py         # the 11 prefitted J-lenses -> lenses/
python step6_acts.py             # pooled activations, 11 models -> acts_{model}.pt
python step7_align.py            # 55-pair alignment + random-unembedding null + heatmaps
```

`step6_acts.py` takes the first 1,000 rows of `NeelNanda/pile-10k`, truncates
to 1,500 characters then 300 tokens, and mean-pools over token positions at
the band layers (0.35 ≤ L/L_max ≤ 0.90). It skips models whose `acts_*.pt`
already exists, so it is resumable.

`step7_align.py` builds the per-layer dictionary `W_U·diag(w)·J_L`, fits the
≤25-atom non-negative sparse code, and reports m-NN alignment (κ = 10, chance
0.010) for the J component, the non-J remainder (`perp`) and the full
activation, plus the seeded
random-unembedding null (R = 5). It prints the pilot table to stdout and writes
55 layer-pair heatmaps (`heatmap_*_vs_*.png`); no script reads either back.
`results/jspace_alignment_pilot.md` is a transcript of that table from the
original run, kept because it is the only record of the
random-unembedding-row null (see [Known gaps](#8-known-gaps)).

**Content ablation (Table 1).** Content words are replaced by
frequency-matched random words, preserving length, syntax, punctuation and the
unigram frequency profile. Run these in order: the first three cache
neighbour lists, and `xmedlayer.py` pairs the J-lens arm with the
Gaussian-dictionary arm into the file Table 1 is built from.

```bash
python xsurrogate.py             # ablated corpus -> cache/surrogate_docs.json
python xsurrogate_all.py         # J-lens arm, 55 pairs -> cache/surr55/, results/surrogate_55.json
python xrandnull.py              # Gaussian-dictionary arm -> cache/randnull/
python xmedlayer.py              # pairs the two -> results/randdict_null_bylayerstat.json
```

Table 1's four rows (three components plus the Gaussian dictionary) are the
mean over 55 pairs of each pair's mean over its layer-pair grid, read from
`results/randdict_null_bylayerstat.json`. `xmeanmain.py` prints the table, its
bootstrap confidence intervals, and every competence correlation:

```bash
python xmeanmain.py              # Table 1 and every rho/p in the paper's tables
python xtokencontrol.py          # does the 300-token cap disadvantage coarser tokenisers?
```

**Lens-degradation control (appendix C.2).** Reducing the number of dictionary
directions on the real corpus gives a J-space that is degraded but computed on
documents with intact meaning — the instrument-only null for the ablation:

```bash
python xksweep.py                # sparsity sweep on real text -> cache/ksweep/
python xksweep_mean.py           # scores it at the paper's aggregation -> results/ksweep_mean.json
```

**Lens-fitting control (appendix C.1).** Every J-lens was fitted on the same
corpus, so J-space alignment might come from the shared fitting text. This
chain refits the lenses on two disjoint halves of WikiText-103 and recomputes
text↔text alignment with the two models of a pair on different halves:

```bash
python xlc_phase0.py         # lens provenance + the half split -> results/lenscontrol/half_manifest.json
python xlc_phase1.py         # refit each lens on each half -> results/lenscontrol/jfit/
python xlc_phase2.py         # h1-vs-h2 stability gate -> results/lenscontrol/phase2_gates.json
python xlc_phase4.py         # decompose the Pile eval activations -> results/lenscontrol/sparse/
python xlc_phase45_score.py  # J against the null -> results/lenscontrol/phase4_sparse.json
python xlc_median_delta.py   # the paper's +0.0002 -> results/lenscontrol/median_delta.json
python xlc_fig_median.py     # Figure 4
```

Scope is the 10 pairs among Pythia-70M, GPT-2, Gemma-3-270M, Qwen3.5-0.8B and
Gemma-3-1B; `xlc_phase1.py` is the 27.5 GPU-h step. Only the J component on
the Pile evaluation set is carried through, because that is what the appendix
figure and its +0.0002 are computed from.

**Competence.** `xeval_bench.py` runs HellaSwag (0-shot, limit 1500,
`acc_norm`) for all 11 models through one lm-eval harness into
`results/lmeval/{model}.json`. It needs `lm-eval` installed and its module
importable. `hellaswag_acc_norm` is the competence axis of Figure 1, Figure 3
and `tab:competence`.

The second competence axis is tokeniser-invariant:

```bash
python xperformance_owt.py       # 1 - bits-per-byte over 4M OpenWebText tokens
```

## 5. Experiment 2 — text↔vision alignment (section 4.2)

Ordered; each stage is idempotent and skips completed work.

```bash
python xstage_data.py                     # 1,024 WIT eval pairs -> eval_manifest.json, cache/images/eval/
python xconvert_clip.py                   # one-time CLIP pickle -> safetensors in models/
python xstage1_text.py                    # caption activations, 11 models -> cache/text_acts/
python xstage1_vision.py                  # image activations, 4 encoders -> cache/vision/
python xstage6_measB.py                   # measurement B -> results/measB.json
python xstage7_controls.py --shuffle      # shuffled-pair null -> results/control_shuffle.json
```

`--shuffle` rewrites `results/control_shuffle.json` with the 44 raw cells the
paper's 0.0096 is averaged over. The shipped copy additionally carries 40
projected cells from a withdrawn pipeline; nothing reads them.

The paper's cross-modal alignment is **measurement B**: the caption side is
decomposed into J / non-J / full components against the text J-lens dictionary,
and matched against *raw* pooled image kernels, over five evenly spaced caption
layers × six image blocks. There is no vision-side Jacobian anywhere in this
pipeline — the paper makes no claim that a vision analogue of the J-space
exists, and `xstage6_measB.py` reads only raw image activations.

The shuffled-pair control re-pairs images and captions at the winning layer
pair, 20 seeded permutations, on raw kernels both sides. It rebuilds its
neighbour sets from the Stage 1 activations directly, so it needs nothing from
the Jacobian side either.

## 6. Experiment 3 — shared J-lens vocabulary (section 4.3)

```bash
python xw_all.py          # top-25 word lists, all 255 lens layers -> cache/wordalign/
python xw_stats.py        # H1-H4 + controls -> results/wordalign/stats.json
python xw_meangrid_raw.py # 11x11 mean depth grid -> results/wordalign/mean_grid_raw.json
```

Readout is `softmax(W_U · norm(J_L h))` restricted to the 31,548 strings that
are a single token in all 11 vocabularies, top-25, compared at 6,000 character
offsets that are a token-end word boundary in all 11 tokenizers. The statistic
is Δ = matched overlap − position-shuffled overlap at matched relative depth
p = L/(N−1). Cost: about 8 minutes of extraction plus 3 minutes of statistics.

## 7. Figures and paper

The paper has four figures, in five PDF files (Figure 2 is a two-panel
figure). Their producers:

```bash
python xplot_pairlevel_fig1.py     # -> paper/07_lobf_components.pdf              (Figure 1)
python xfig_paper.py               # -> paper/06_wordalign_mean_heatmap.pdf,
                                   #    paper/08_j_minus_logit_vs_depth.pdf       (Figure 2)
python xplot_wordalign_pairs.py    # -> paper/09_wordalign_pairs_vs_competence.pdf (Figure 3)
python xlc_fig_median.py           # -> paper/10_lens_fitting_control.pdf         (Figure 4)
```


Build the paper:

```bash
cd paper && pdflatex neurips_2026 && bibtex neurips_2026 && pdflatex neurips_2026 && pdflatex neurips_2026
```

## 8. Known gaps

Read before assuming a number can be regenerated from this folder alone.

- **Some numbers are not machine-checkable**, and the verifier lists each as a
  `SKIP`. The random-unembedding-row null (55/55, mean margin +0.148) exists
  only as prose in `results/jspace_alignment_pilot.md` — `step7_align.py`
  prints it and writes no JSON, and it is a max-over-grid statistic rather
  than the paper's mean. The "~40 GPU hours" figure is an estimate; only the
  27.5 h lens-refit chain is measured anywhere. The permutation p-values are
  stochastic (rerun `xmeanmain.py` to reproduce them), and the
  convergence-rate slopes (+0.76 ± 0.09 vs +0.23 ± 0.07) are fitted by
  `xplot_pairlevel_fig1.py` into the Figure 1 legend rather than stored. The
  shuffled-pair control is another; see the bullet on it below.
- **Figure 4 cannot be rebuilt here.** `xlc_fig_median.py` and
  `xlc_median_delta.py` read `results/lenscontrol/sparse/*.pt`, the
  neighbour-set cache `xlc_phase4.py` writes; it is deliberately not shipped
  (99 MB as written by the run behind the paper, about 40 MB from the trimmed
  chain in this folder, which decomposes only the J component on the Pile eval
  set — the columns the appendix figure uses). The numbers those scripts
  produced are present in `results/lenscontrol/median_delta.json`, which also
  still carries the caption-corpus and non-J columns the earlier run computed,
  and the figure itself is in `paper/`.
- **Large caches are not shipped**, so several scripts verify but do not
  re-run from scratch: `acts_*.pt` (2 GB) for Experiment 1, `cache/surr55/`,
  `cache/randnull/` and `cache/ksweep/` for the ablation and its nulls,
  `cache/text_acts/` and `cache/vision/` for Experiment 2, and
  `cache/wordalign/*.pt` for `xw_stats.py`. `xperformance_owt.py` likewise
  reads a fixed local OpenWebText sample (`cache/owt/sample_4M.json`, a list
  of file paths) that it does not build and this folder does not carry. Every
  JSON the verifier reads is present.
- **Vocab-mean centering is mandatory.** After folding the final-norm weights
  (`WUeff = W_U · w`), subtract the vocabulary mean: `WUeff -= WUeff.mean(0)`,
  from the lens dictionary *and* the random-control dictionary. GPT-2's lens
  dictionary is pathologically anisotropic (mean normalised atom norm ≈ 0.995)
  and without centering its non-negative fit collapses to ≈ 0 variance share.
  The removed direction is softmax shift-invariant, so no readout information
  is lost. The per-layer `dict aniso` print should read ≈ 0.01–0.03; > 0.9
  means centering was lost.
- **Per-family norm conventions.** Gemma's RMSNorm is zero-centered, so the
  fold is `1 + weight`; GPT-2/Pythia LayerNorm and Qwen/Llama RMSNorm fold
  directly.
- **`output_hidden_states=True` must be passed at forward time**, not to
  `from_pretrained` — newer architectures silently return `hidden_states=None`.
  The code asserts on this.
- **Dtypes.** GPT-2 fp16, everything else bf16; Gemma overflows to NaN around
  layer 13 in fp16. Cross-modal Jacobians are fp32 with TF32 **off**
  (TF32 fails the 1% tolerance at rel 0.023).
- **`hidden_states[-1]` is post-final-norm.** The lens convention needs the
  pre-norm state, captured with a hook (`xlc_suffix.py`).
- **Vision specifics.** `attn_implementation='eager'` everywhere
  (`torch.func.jvp` does not support SDPA); MAE needs an identity `noise`
  vector or it shuffles patches even at `mask_ratio=0`; freeze parameters with
  `requires_grad_(False)` or JVP retains reverse graphs and leaks ~22 GB; JVP
  chunk 16, not 64, at T = 257.
- **m-NN is scale-invariant.** `prep()` row-normalises before building
  neighbour lists, so a component that merely shrinks cannot lose alignment —
  do not read a falling variance share as instrument degradation.
- **Top-k comparisons must be set comparisons.** Exact logit ties reorder
  `topk`, so index equality gives false failures.
- **Never `import step7_align`** — it executes on import. Use
  `crossmodal_utils.load_pilot()`, which extracts its functions by AST.
- **`matplotlib` was installed in the plotting interpreter only**, not the
  torch venv, in the original setup.

## 10. Compute

All experiments ran on one RTX 3080 (10 GB); the paper quotes about **40 GPU
hours** in total. Separately measured: word-level readout 8 min extraction +
3 min statistics; the disjoint-lens-fitting control 27.5 GPU-h across 5 models
(pythia 0.4 h, gpt2 1.6 h, gemma-3-270m 1.5 h, qwen3.5-0.8b 8.8 h,
gemma-3-1b 15.2 h — a 1.7B refit was measured at 742 h, which set the scope of
that control).

## 11. Layout

```
README.md                     this file
requirements.txt              pinned versions
verify_paper_numbers.py       recomputes the paper's numbers, no GPU needed
*.py                          39 scripts, flat (they import each other)
eval_manifest.json            the frozen 1,024-pair WIT eval set
run_manifest_lenscontrol.json lens provenance and the WikiText-103 half split
tests/                        JVP correctness gate for the lens-refit estimator
paper/                        LaTeX source, bibliography, five figure PDFs, compiled PDF
results/                      the artifacts every number is computed from
```

Script families, by prefix:

| Prefix | What it is |
|---|---|
| `step*` | Experiment 1: lens download, pooled activations, 55-pair alignment |
| `xsurrogate*`, `xrandnull`, `xmedlayer`, `xksweep*` | content ablation, its Gaussian-dictionary null, and the lens-degradation control |
| `xstage*`, `crossmodal_*`, `xvision_config`, `xkernels`, `xconvert_clip` | Experiment 2: cross-modal data, activations, measurement B, shuffle control |
| `xw_*` | Experiment 3: word-level lens readout |
| `xmeanmain`, `xtokencontrol` | the paper's tables and the tokenisation control |
| `xeval_bench`, `xperformance_owt` | competence axes: HellaSwag and 1 - bits-per-byte |
| `xlc_*` | control: disjoint lens-fitting corpora |
| `xplot_*`, `xfig_paper` | the paper's figures |

`results/jspace_alignment_pilot.md` is the text–text pilot table, and the only
record of the random-unembedding-row null.
