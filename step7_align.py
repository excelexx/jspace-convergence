"""Text-text alignment over the 55 model pairs, and the random-unembedding null.

Builds the per-layer J-space dictionary W_U.diag(w).J_L for each of the 11
models, fits the k=25 non-negative sparse code to the pooled activations from
step6_acts.py, and reports m-NN alignment (kappa = 10) for the J component, the
non-J remainder and the full activation over the band x band layer-pair grid.

It also runs the control reported in section 4.4: the same pipeline with the
dictionary replaced by random unembedding rows, over R_DRAWS = 5 seeded draws,
against which J-space alignment is compared per pair. The comparison uses
mean-over-grid aggregation, the same band-mean convention as every other
alignment number in the paper, and is persisted to
results/randunembed_null.json.

Output is stdout, the null JSON above, plus one layer-pair heatmap per model
pair. The reference table to compare the stdout against is
results/jspace_alignment_pilot.md.

This file is also the single source of truth for the sparse-coding conventions:
crossmodal_utils.load_pilot() parses it and re-executes the named constants and
functions below, so the cross-modal stages never re-derive them. Do not import
this module - it runs on import.
"""
import torch

torch.manual_seed(0)
DEV = "cuda"
K_SPARSE, K_NN = 25, 10
R_DRAWS = 5                                      # seeded random-dictionary null draws
V_CHUNK = 65536                                  # dictionary rows scored per matmul

MODELS = {
    "gpt2": dict(
        acts="acts_gpt2.pt", hf="gpt2",
        lens="lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
    "gemma": dict(
        acts="acts_gemma.pt", hf="google/gemma-3-1b-pt",
        lens="lenses/gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt"),
    "gemma270": dict(
        acts="acts_gemma270.pt", hf="google/gemma-3-270m",
        lens="lenses/gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt"),
    "pythia70m": dict(
        acts="acts_pythia70m.pt", hf="EleutherAI/pythia-70m-deduped",
        lens="lenses/pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"),
    "qwen08b": dict(
        acts="acts_qwen08b.pt", hf="Qwen/Qwen3.5-0.8B",
        lens="lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"),
    "qwen17b": dict(
        acts="acts_qwen17b.pt", hf="Qwen/Qwen3-1.7B",
        lens="lenses/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"),
    "qwen2b": dict(
        acts="acts_qwen2b.pt", hf="Qwen/Qwen3.5-2B-Base",
        lens="lenses/qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt"),
    "gemma2_2b": dict(
        acts="acts_gemma2_2b.pt", hf="google/gemma-2-2b",
        lens="lenses/gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt"),
    "qwen4b": dict(
        acts="acts_qwen4b.pt", hf="Qwen/Qwen3-4B",
        lens="lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"),
    "qwen35_4b": dict(
        acts="acts_qwen35_4b.pt", hf="Qwen/Qwen3.5-4B",
        lens="lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"),
    "gemma3_4b": dict(
        acts="acts_gemma3_4b.pt", hf="google/gemma-3-4b-pt",
        lens="lenses/gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt"),
}

NORM_PATHS = ["transformer.ln_f", "gpt_neox.final_layer_norm", "model.norm",
              "model.language_model.norm", "language_model.model.norm"]

def get_WU_and_w(hf):
    """Unembedding (V,d) and final-norm weights, per family convention."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(hf, dtype="auto")  # native dtype: half RAM
    W = m.get_output_embeddings().weight.detach().float().clone()   # (V, d)
    norm = None
    for attr in NORM_PATHS:
        obj = m
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            norm = obj
            break
        except AttributeError:
            continue
    assert norm is not None, f"no final norm found for {hf}"
    w = norm.weight.detach().float().clone()
    # Gemma RMSNorms store weights zero-centered (forward computes 1 + w);
    # GPT-2/Pythia LayerNorm and Llama/Qwen-style RMSNorm store them directly
    if "gemma" in type(norm).__name__.lower():
        w = 1.0 + w
    del m
    return W, w

def nnls_refit(A, h, iters=80):
    """Batched nonnegative least squares: min ||h - A c||^2, c >= 0.
       A: (n,d,s), h: (n,d,1). Projected gradient, step = 1/trace(G) <= 1/L."""
    At = A.transpose(1, 2)
    G = At @ A                                   # (n,s,s)
    b = At @ h                                   # (n,s,1)
    step = 1.0 / G.diagonal(dim1=1, dim2=2).sum(-1).clamp(min=1e-8)
    c = torch.zeros_like(b)
    for _ in range(iters):
        c = torch.clamp(c - step.view(-1, 1, 1) * (G @ c - b), min=0)
    return c

def best_atom(r, D, forbid):
    """argmax_v <r_i, D_v> per row, scored in V_CHUNK blocks to bound memory.
       forbid: (n, s) already-selected indices, or None."""
    n = r.shape[0]
    best_v = torch.full((n,), -1e30, device=DEV)
    best_i = torch.zeros(n, dtype=torch.long, device=DEV)
    for start in range(0, D.shape[0], V_CHUNK):
        S = r @ D[start:start + V_CHUNK].T       # (n, chunk)
        if forbid is not None:                   # forbid re-picking
            inchunk = (forbid >= start) & (forbid < start + S.shape[1])
            rows, cols = inchunk.nonzero(as_tuple=True)
            S[rows, forbid[rows, cols] - start] = -1e30
        v, i = S.max(dim=1)
        upd = v > best_v
        best_v = torch.where(upd, v, best_v)
        best_i = torch.where(upd, i + start, best_i)
    return best_i

def nnomp_batch(H, D):
    """H: (n,d) activations. D: (V,d) unit-norm dictionary. Returns h_J (n,d)."""
    r = H.clone()
    sel = torch.zeros(H.shape[0], K_SPARSE, dtype=torch.long, device=DEV)
    for step_i in range(K_SPARSE):
        sel[:, step_i] = best_atom(r, D, sel[:, :step_i] if step_i else None)
        A = D[sel[:, :step_i + 1]].transpose(1, 2)   # (n, d, s)
        c = nnls_refit(A, H.unsqueeze(-1))
        r = H - (A @ c).squeeze(-1)
    return H - r

def prep(X):
    q = torch.quantile(X.abs().flatten().float(), 0.95)
    X = X.clamp(-q, q)
    nrm = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return X / nrm

def neighbors(X):
    K = X @ X.T
    K.fill_diagonal_(-1e30)
    return K.topk(K_NN, dim=1).indices           # (n, k)

def mnn(Sa, Sb):
    hits = (Sa.unsqueeze(2) == Sb.unsqueeze(1)).any(-1).sum().item()
    return hits / (Sa.shape[0] * K_NN)

nbrs = {}
N_DOCS = None
for name, cfg in MODELS.items():
    print(f"\n=== decomposing {name} ===")
    acts = torch.load(cfg["acts"], weights_only=False)
    n0 = next(iter(acts.values())).shape[0]
    assert N_DOCS in (None, n0), f"doc-count mismatch: {name} has {n0}, expected {N_DOCS}"
    N_DOCS = n0
    lens = torch.load(cfg["lens"], map_location="cpu", weights_only=False)
    WU, w = get_WU_and_w(cfg["hf"])
    WUeff = WU.to(DEV) * w.to(DEV)               # fold final-norm weights in
    del WU
    # drop the vocab-mean readout direction: it shifts all logits equally, so
    # the softmax is invariant to it
    WUeff -= WUeff.mean(dim=0, keepdim=True)
    Vsz, d = WUeff.shape
    print(f"vocab {Vsz}, width {d}, layers {sorted(acts.keys())}")

    Drs = []
    for r in range(R_DRAWS):
        g = torch.Generator().manual_seed(1000 + r)
        rand_rows = torch.randperm(Vsz, generator=g)[: min(Vsz, 20000)]
        Dr = WUeff[rand_rows]
        Drs.append(Dr / Dr.norm(dim=1, keepdim=True).clamp(min=1e-8))

    nbrs[name] = {c: {} for c in ["full", "J", "perp"]}
    nbrs[name]["randJ"] = [{} for _ in range(R_DRAWS)]
    for L in sorted(acts.keys()):
        H = acts[L].to(DEV, torch.float32)
        q = torch.quantile(H.abs().flatten(), 0.95)
        H = H.clamp(-q, q)                       # tame outliers before pursuit

        J = lens["J"][L].to(DEV, torch.float32)
        D = WUeff @ J
        D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
        aniso = D.mean(dim=0).norm().item()      # ~1 = atoms collapsed to one direction

        HJ = nnomp_batch(H, D)
        perp = H - HJ

        var_share = (HJ.norm(dim=1)**2 / H.norm(dim=1)**2).mean().item()
        flag = ""
        if var_share > 1.0:
            flag = "  !! >1, decomposition broken, do not trust"
        elif var_share < 0.005:
            flag = "  !! ~0, dictionary orthogonal to acts, do not trust"
        if aniso > 0.9:
            flag += "  !! dictionary collapsed to one direction"
        print(f"  L{L:>2}: J-space variance share = {var_share:.3f}"
              f"  (dict aniso {aniso:.2f}){flag}")

        for comp, X in [("full", H), ("J", HJ), ("perp", perp)]:
            nbrs[name][comp][L] = neighbors(prep(X)).cpu()
        for r, Dr in enumerate(Drs):             # null: random unembedding rows
            HJr = nnomp_batch(H, Dr)
            nbrs[name]["randJ"][r][L] = neighbors(prep(HJr)).cpu()
        del D, HJ, HJr, perp                     # free before next layer's dictionary
    del WUeff, Drs
    torch.cuda.empty_cache()

CHANCE = K_NN / (N_DOCS - 1)

def mnn_grid(nbrs_a, nbrs_b):
    La, Lb = sorted(nbrs_a), sorted(nbrs_b)
    G = torch.tensor([[mnn(nbrs_a[a], nbrs_b[b]) for b in Lb] for a in La])
    return La, Lb, G

def grid_max(La, Lb, G):
    flat = G.argmax().item()
    ia, ib = divmod(flat, G.shape[1])
    return G.max().item(), (La[ia], Lb[ib])

names = list(MODELS)
pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
grids = {}
null_records = []
for a, b in pairs:
    print(f"\n=== alignment: {a} vs {b} (chance = {CHANCE:.3f}, n = {N_DOCS}) ===")
    results, means = {}, {}
    for comp in ["J", "full", "perp"]:
        La, Lb, G = mnn_grid(nbrs[a][comp], nbrs[b][comp])
        grids[(a, b, comp)] = (La, Lb, G)
        best, arg = grid_max(La, Lb, G)
        results[comp] = best
        means[comp] = G.mean().item()
        print(f"  {comp:>5}: max m-NN = {best:.4f}   at layers {arg}"
              f"   (grid mean {means[comp]:.4f}, median {G.median():.4f})")

    # percentile null: mean-over-grid, the same band-mean aggregation the paper
    # uses for every other alignment number (was max-over-grid before 2026-08-24)
    draws = [mnn_grid(nbrs[a]["randJ"][r], nbrs[b]["randJ"][r])[2].mean().item()
             for r in range(R_DRAWS)]
    print(f"  randJ null ({R_DRAWS} draws, grid mean): " +
          " ".join(f"{v:.4f}" for v in draws))
    assert max(draws) < 1.0 and min(draws) > CHANCE / 2, "degenerate null draws"
    n_beat = sum(means["J"] > v for v in draws)
    print(f"  J mean ({means['J']:.4f}) exceeds {n_beat}/{R_DRAWS} random-dictionary"
          f" draws, margin over the strongest {means['J'] - max(draws):+.4f}")
    null_records.append(dict(
        a=a, b=b, J_mean=round(means["J"], 6),
        null_means=[round(v, 6) for v in draws], n_beat=n_beat,
        margin_over_strongest=round(means["J"] - max(draws), 6)))

import json, os
os.makedirs("results", exist_ok=True)
_agg = dict(
    n_pairs=len(null_records),
    pairs_beating_all_draws=sum(r["n_beat"] == R_DRAWS for r in null_records),
    mean_margin_over_strongest=round(
        sum(r["margin_over_strongest"] for r in null_records) / len(null_records), 6),
    min_margin_over_strongest=round(
        min(r["margin_over_strongest"] for r in null_records), 6))
with open("results/randunembed_null.json", "w") as f:
    json.dump(dict(
        note="random-unembedding-row null (section 4.4), mean-over-grid "
             "aggregation; R_DRAWS seeded draws of min(V, 20000) rows",
        chance=CHANCE, r_draws=R_DRAWS, aggregate=_agg, pairs=null_records,
    ), f, indent=1)
print(f"\nrandJ null summary: {_agg['pairs_beating_all_draws']}/{_agg['n_pairs']}"
      f" pairs beat all {R_DRAWS} draws, mean margin"
      f" {_agg['mean_margin_over_strongest']:+.4f}"
      f" (written to results/randunembed_null.json)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPS = ["J", "full", "perp"]
for a, b in pairs:
    fig, axes = plt.subplots(1, len(COMPS), figsize=(4.2 * len(COMPS), 3.6))
    vmax = max(grids[(a, b, c)][2].max().item() for c in COMPS)
    for ax, comp in zip(axes, COMPS):
        La, Lb, G = grids[(a, b, comp)]
        im = ax.imshow(G, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        ax.set_title(comp)
        ax.set_xlabel(f"{b} layer"); ax.set_ylabel(f"{a} layer")
        ax.set_xticks(range(len(Lb)), Lb); ax.set_yticks(range(len(La)), La)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"m-NN alignment: {a} vs {b} (chance {CHANCE:.3f})")
    fig.tight_layout()
    out = f"heatmap_{a}_vs_{b}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")