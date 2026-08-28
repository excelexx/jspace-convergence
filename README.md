# How Much of Platonic Convergence is Semantic? Measuring Alignment in the Jacobian Subspace

Code and artifacts for the paper. Everything that produces a number or a figure
is here, along with the result files those runs wrote and the LaTeX source.

The paper measures convergence between models inside the **J-space** — the
subspace induced by the Jacobian lens — rather than the full activation space,
across three experiments:

| # | Experiment | Scale | Paper |
|---|---|---|---|
| 1 | Text↔text alignment, plus content ablation | 11 models, 55 pairs, 1,000 Pile documents | §4.1 |
| 2 | Text↔vision alignment | 4 ViT-B encoders × 11 models = 44 pairs, 1,024 WIT pairs | §4.2 |
| 3 | Shared J-lens vocabulary, read out as words | 11 models, 55 pairs, 6,000 character offsets | §4.3 |

---

## 1. Check the paper's numbers (no GPU, ~30 seconds)

```bash
python verify_paper_numbers.py
python verify_appendix_metrics.py
```

The first recomputes 78 numbers from the JSONs in `results/` and compares each
against the paper. The second covers 102 more in the metric appendix, parsing
the tables and prose straight out of `paper/neurips_2026.tex` so an edited
number is caught automatically.

Both need only `numpy` and `scipy`. Each line prints `OK` or `MISMATCH`. A
`SKIP` marks a number whose source artifact is not shipped — see
[§6](#6-what-you-cannot-rerun-here).

Two more checks, also cheap:

```bash
python tests/test_jvp_identity.py       # JVP estimator vs explicit autograd Jacobians
python metric_sweep/verify_metrics.py   # each metric vs a feature-space reference
```

## 2. Setup

Python 3.12.10, one NVIDIA GPU with ≥10 GB (everything ran on a single RTX
3080), CUDA 12.1.

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run every script **from this directory** — paths are relative to it and the
scripts import each other as top-level modules.

Assets download on demand, except the lenses:

| Asset | Where | Size | How |
|---|---|---|---|
| J-lenses, 11 models | `lenses/` | ~2.4 GB | `python step2_download.py` |
| 11 language models | HF cache | ~30 GB | on demand |
| 4 vision encoders | HF cache + `models/` | ~1.5 GB | on demand; CLIP needs `xconvert_clip.py` |
| pile-10k, WIT, WikiText-103 | HF cache | ~1 GB | on demand (WIT streams) |

Gemma is gated — run `huggingface-cli login` with the licences accepted.

## 3. Experiment 1 — text↔text alignment (§4.1)

```bash
python step2_download.py    # 11 prefitted J-lenses -> lenses/
python step6_acts.py        # pooled activations -> acts_{model}.pt
python step7_align.py       # 55-pair alignment + random-unembedding null
```

**Content ablation (Table 1).** Run in order; the first three cache neighbour
lists and the last pairs the two arms into the file Table 1 is built from.

```bash
python xsurrogate.py        # ablated corpus -> cache/surrogate_docs.json
python xsurrogate_all.py    # J-lens arm -> cache/surr55/, results/surrogate_55.json
python xrandnull.py         # Gaussian-dictionary arm -> cache/randnull/
python xmedlayer.py         # pairs them -> results/randdict_null_bylayerstat.json
python xmeanmain.py         # prints Table 1 and every rho/p in the paper
```

**Controls and competence.**

```bash
python xtokencontrol.py     # tokeniser efficiency across the 11 models
python xksweep.py           # lens-degradation sweep on real text -> cache/ksweep/
python xksweep_mean.py      # scores it -> results/ksweep_mean.json
python xeval_bench.py       # HellaSwag 0-shot -> results/lmeval/{model}.json
python xperformance_owt.py  # 1 - bits-per-byte over 4M OpenWebText tokens
```

**Lens-fitting control (appendix C.1).** Refits every lens on two disjoint
halves of WikiText-103 and recomputes alignment with the two models of a pair
on different halves. Scope is the 10 pairs among Pythia-70M, GPT-2,
Gemma-3-270M, Qwen3.5-0.8B and Gemma-3-1B.

```bash
python xlc_phase0.py         # provenance + half split
python xlc_phase1.py         # refit each lens per half   (27.5 GPU-h)
python xlc_phase2.py         # h1-vs-h2 stability gate
python xlc_phase4.py         # decompose the Pile eval activations
python xlc_phase45_score.py  # J against the null
python xlc_median_delta.py   # the paper's +0.0002
python xlc_fig_median.py     # Figure 5
```

## 4. Experiment 2 — text↔vision alignment (§4.2)

Ordered; each stage skips completed work.

```bash
python xstage_data.py       # 1,024 WIT pairs -> eval_manifest.json, cache/images/eval/
python xconvert_clip.py     # one-time CLIP pickle -> safetensors
python xstage1_text.py      # caption activations -> cache/text_acts/
python xstage1_vision.py    # image activations -> cache/vision/
python xstage6_measB.py     # the paper's measurement -> results/measB.json
python xstage7_controls.py  # shuffled-pair null -> results/control_shuffle.json
```

The caption side is decomposed into J / non-J / full against the text J-lens
dictionary and matched against **raw** pooled image kernels, over five caption
layers × six image blocks. There is no vision-side Jacobian anywhere — the
paper claims no vision analogue of the J-space exists.

## 5. Experiment 3 — shared J-lens vocabulary (§4.3)

```bash
python xw_all.py            # top-25 word lists, 255 lens layers -> cache/wordalign/
python xw_stats.py          # per-pair overlap -> results/wordalign/stats.json
python xw_meangrid_raw.py   # 11x11 depth grid -> results/wordalign/mean_grid_raw.json
python xw_rawgrid_argmax.py # counterpart-layer offset -> results/wordalign/rawgrid_argmax.json
```

Readout is `softmax(W_U · norm(J_L h))`, restricted to the 31,548 strings that
are a single token in all 11 vocabularies, top-25, compared at 6,000 character
offsets that fall on a token-end word boundary in all 11 tokenisers. The
statistic is the raw overlap at matched relative depth λ = L/(N−1); the
position-shuffled floor is reported separately as a control. About 8 minutes of
extraction plus 3 of statistics.

## 6. Figures and the paper

```bash
python xfig_teaser.py           # -> paper/01_teaser.pdf                        (Figure 1)
python xplot_pairlevel_fig1.py  # -> paper/07_lobf_components.pdf               (Figure 2)
python xfig_paper.py            # -> paper/06_wordalign_mean_heatmap.pdf and
                                #    paper/08_j_minus_logit_vs_depth.pdf        (Figure 3)
python xplot_wordalign_pairs.py # -> paper/09_wordalign_pairs_vs_competence.pdf (Figure 4)
python xlc_fig_median.py        # -> paper/10_lens_fitting_control.pdf          (Figure 5)
```

Figure numbers are the paper's; the PDF filename prefixes have not matched them
since the teaser was added.

```bash
cd paper && pdflatex neurips_2026 && bibtex neurips_2026 && pdflatex neurips_2026 && pdflatex neurips_2026
```

## 7. What you cannot rerun here

Every JSON the verifiers read is present. These are not:

- **Large caches.** `acts_*.pt` (2 GB), `cache/surr55/`, `cache/randnull/`,
  `cache/ksweep/`, `cache/text_acts/`, `cache/vision/`, `cache/wordalign/*.pt`,
  and the fixed OpenWebText sample `cache/owt/sample_4M.json`. Scripts
  downstream of these verify but do not rebuild from scratch.
- **Figure 5.** `xlc_fig_median.py` and `xlc_median_delta.py` read
  `results/lenscontrol/sparse/*.pt`, deliberately not shipped (~40 MB). The
  numbers are in `results/lenscontrol/median_delta.json` and the figure is in
  `paper/`.
- **A few numbers are not machine-checkable** and the verifier marks each
  `SKIP`: the ~40 GPU-hour estimate, the stochastic permutation p-values
  (rerun `xmeanmain.py`), and the convergence-rate slopes (+0.76 ± 0.09 vs
  +0.23 ± 0.07), which `xplot_pairlevel_fig1.py` fits into the Figure 2 legend
  rather than storing.

## 8. Implementation notes

Hard-won details. If a run gives implausible numbers, the cause is usually here.

- **Vocab-mean centering is mandatory.** After folding the final-norm weights
  (`WUeff = W_U · w`), subtract the vocabulary mean: `WUeff -= WUeff.mean(0)` —
  from the lens dictionary *and* the random control. GPT-2's dictionary is
  pathologically anisotropic (mean normalised atom norm ≈ 0.995) and without
  centering its non-negative fit collapses to ≈ 0 variance share. The removed
  direction is softmax shift-invariant, so no readout information is lost. The
  per-layer `dict aniso` print should read ≈ 0.01–0.03; > 0.9 means centering
  was lost.
- **Per-family norm conventions.** Gemma's RMSNorm is zero-centered, so the
  fold is `1 + weight`. GPT-2/Pythia LayerNorm and Qwen/Llama RMSNorm fold
  directly.
- **Dtypes.** GPT-2 fp16, everything else bf16 — Gemma overflows to NaN around
  layer 13 in fp16. Cross-modal Jacobians are fp32 with TF32 **off** (TF32
  fails the 1% tolerance at rel 0.023).
- **Pass `output_hidden_states=True` at forward time**, not to
  `from_pretrained` — newer architectures silently return `hidden_states=None`.
  The code asserts on this.
- **`hidden_states[-1]` is post-final-norm.** The lens convention needs the
  pre-norm state, captured with a hook (`xlc_suffix.py`).
- **Never `import step7_align`** — it executes on import. Use
  `crossmodal_utils.load_pilot()`, which extracts its functions by AST.
- **m-NN is scale-invariant.** `prep()` row-normalises before building
  neighbour lists, so a component that merely shrinks cannot lose alignment.
  Do not read a falling variance share as instrument degradation.
- **Top-k comparisons must be set comparisons.** Exact logit ties reorder
  `topk`, so index equality gives false failures.
- **Vision specifics.** `attn_implementation='eager'` everywhere
  (`torch.func.jvp` does not support SDPA); MAE needs an identity `noise`
  vector or it shuffles patches even at `mask_ratio=0`; freeze parameters with
  `requires_grad_(False)` or JVP retains reverse graphs and leaks ~22 GB; JVP
  chunk 16, not 64, at T = 257.

## 9. Layout

```
verify_paper_numbers.py       recomputes the paper's numbers, no GPU
verify_appendix_metrics.py    the same for the metric appendix
*.py                          40 further scripts, flat (they import each other)
eval_manifest.json            the frozen 1,024-pair WIT eval set
run_manifest_lenscontrol.json lens provenance and the WikiText-103 half split
tests/                        JVP correctness gate for the lens-refit estimator
paper/                        LaTeX source, bibliography, six figure PDFs, PDF
results/                      the artifacts every number is computed from
metric_sweep/                 the appendix's eight-metric sweep, one folder per metric
ablation_sweep/               the content ablation repeated under each metric
```

Scripts by prefix:

| Prefix | What it is |
|---|---|
| `step*` | Experiment 1: lens download, activations, 55-pair alignment |
| `xsurrogate*`, `xrandnull`, `xmedlayer`, `xksweep*` | content ablation, its Gaussian null, lens degradation |
| `xstage*`, `crossmodal_*`, `xvision_config`, `xkernels`, `xconvert_clip` | Experiment 2 |
| `xw_*` | Experiment 3: word-level lens readout |
| `xmeanmain`, `xtokencontrol` | the paper's tables, tokenisation control |
| `xeval_bench`, `xperformance_owt` | competence axes |
| `xlc_*` | the disjoint lens-fitting control |
| `xplot_*`, `xfig_*` | the paper's figures |

Compute: about **40 GPU hours** total on one RTX 3080. Measured separately —
word-level readout 8 min + 3 min; the lens-fitting control 27.5 GPU-h across
5 models (a 1.7B refit was measured at 742 h, which set that control's scope).
