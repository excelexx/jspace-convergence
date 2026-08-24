"""Experiment 3 (paper section 4.3), extraction stage: J-lens word read-outs
for all 11 text models.

For every model and every lens layer, extract the top-25 words of
    lens_L(h) = softmax(W_U . norm(J_L h))
restricted to the shared single-token word set A -- the 31,548 strings that are
a single token in all 11 vocabularies -- at the 6,000 character offsets that
are a token-end word boundary in ALL 11 tokenizers.  Two read-outs per layer:

    J      the J-lens itself
    base   the plain logit lens (identity in place of J), the paper's control
           for "the models simply predict similar next words"

Writes cache/wordalign/{model}.pt (+ _DONE markers, resumable), consumed by
xw_stats.py (per-pair statistics) and xw_meangrid.py (the mean depth grid
behind Figure 2(a)).
"""
import os, json, copy, hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from crossmodal_utils import load_pilot

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
N_DOCS, MAX_TOK, N_POS = 200, 300, 6000
TOPK = 25
SEED_POS = 31337
OUT, CACHE = "results/wordalign", "cache/wordalign"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

MODELS = {
    "gpt2": ("gpt2", "lenses/gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
    "gemma": ("google/gemma-3-1b-pt",
              "lenses/gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt"),
    "gemma270": ("google/gemma-3-270m",
                 "lenses/gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt"),
    "pythia70m": ("EleutherAI/pythia-70m-deduped",
                  "lenses/pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"),
    "qwen08b": ("Qwen/Qwen3.5-0.8B",
                "lenses/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"),
    "qwen17b": ("Qwen/Qwen3-1.7B",
                "lenses/qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"),
    "qwen2b": ("Qwen/Qwen3.5-2B-Base",
               "lenses/qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt"),
    "gemma2_2b": ("google/gemma-2-2b",
                  "lenses/gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt"),
    "qwen4b": ("Qwen/Qwen3-4B",
               "lenses/qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B",
                  "lenses/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"),
    "gemma3_4b": ("google/gemma-3-4b-pt",
                  "lenses/gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt"),
}
NORM_PATHS = load_pilot().NORM_PATHS          # pilot's per-family final-norm paths


def get_final_norm(model):
    for attr in NORM_PATHS:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return attr, obj
        except AttributeError:
            continue
    raise RuntimeError("no final norm found")


def is_wordy(s):
    core = s.strip()
    return len(core) >= 2 and core.replace("'", "").isalpha() and core.isascii()


AF = f"{OUT}/anchors.json"
if os.path.exists(AF):
    meta = json.load(open(AF))
    A = meta["strings"]
    IDS = {k: torch.tensor(v) for k, v in meta["ids"].items()}
    print(f"loaded {len(A)} gated anchors from {AF}")
else:
    prelim = f"{OUT}/anchors_prelim.json"
    if os.path.exists(prelim):
        cand = json.load(open(prelim))["strings"]
        print(f"candidate anchors from prelim cache: {len(cand)}")
    else:
        maps = {}
        for name, (hf, _) in MODELS.items():
            tok = AutoTokenizer.from_pretrained(hf)
            strs = tok.batch_decode([[i] for i in range(len(tok))])
            special = set(tok.all_special_ids)
            m = {}
            for i, s in enumerate(strs):
                if i in special or s == "" or "�" in s:
                    continue
                m.setdefault(s, i)
            maps[name] = m
            print(f"  {name:>10}: {len(m)} usable strings", flush=True)
        cand = sorted(s for s in set.intersection(*[set(v) for v in maps.values()])
                      if is_wordy(s))
        json.dump({"strings": cand}, open(prelim, "w"))
        print(f"candidate anchors (single token in all 11, wordy): {len(cand)}")

    # round-trip gate: must re-encode to exactly one token in every model
    enc_all, keep = {}, set(range(len(cand)))
    for name, (hf, _) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf)
        e = tok(cand, add_special_tokens=False)["input_ids"]
        ok = {i for i, x in enumerate(e) if len(x) == 1}
        print(f"  round-trip {name:>10}: {len(ok)}/{len(cand)}")
        enc_all[name], keep = e, keep & ok
    surv = sorted(keep)
    A = [cand[i] for i in surv]
    IDS = {}
    for name, (hf, _) in MODELS.items():
        ids = torch.tensor([enc_all[name][i][0] for i in surv])
        assert len(torch.unique(ids)) == len(ids), f"{name}: duplicate anchor ids"
        IDS[name] = ids
    json.dump({"strings": A, "ids": {k: v.tolist() for k, v in IDS.items()}},
              open(AF, "w"))
nA = len(A)
print(f"anchor set n_A = {nA}")

PF = f"{OUT}/positions.json"
docs = [d["text"][:1500] for d in
        load_dataset("NeelNanda/pile-10k", split="train").select(range(N_DOCS))]
doc_hash = hashlib.sha1("".join(docs).encode()).hexdigest()[:12]
if os.path.exists(PF):
    pm = json.load(open(PF))
    assert pm["doc_hash"] == doc_hash, "corpus changed since positions were built"
    POS = pm["pos"]
    print(f"loaded {len(POS)} positions from {PF}")
else:
    ends = {}
    for name, (hf, _) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf)
        ends[name] = [{e: i for i, (s, e) in enumerate(
            tok(d, return_offsets_mapping=True, truncation=True,
                max_length=MAX_TOK)["offset_mapping"]) if e > s} for d in docs]
    names = list(MODELS)
    allpos = []
    for di, d in enumerate(docs):
        shared = set.intersection(*[set(ends[n][di]) for n in names])
        for c in sorted(shared):
            if c < len(d) and d[c].isspace() and d[c - 1].isalnum():
                allpos.append([di, c] + [ends[n][di][c] for n in names])
    print(f"shared word-boundary positions across 11: {len(allpos)}")
    g = torch.Generator().manual_seed(SEED_POS)
    sel = torch.randperm(len(allpos), generator=g)[:N_POS].sort().values.tolist()
    POS = [allpos[i] for i in sel]
    json.dump({"doc_hash": doc_hash, "order": names, "pos": POS}, open(PF, "w"))
n_pos = len(POS)
print(f"n_pos = {n_pos}   (docs sha1 {doc_hash})")
SLOT = {n: 2 + i for i, n in enumerate(MODELS)}
BY_DOC = {}
for k, p in enumerate(POS):
    BY_DOC.setdefault(p[0], []).append((k, p))

for name, (hf, lens_path) in MODELS.items():
    done = f"{CACHE}/{name}_DONE"
    if os.path.exists(done):
        print(f"\n=== {name}: cached, skipping ===")
        continue
    print(f"\n=== {name} ===", flush=True)
    lens = torch.load(lens_path, map_location="cpu", weights_only=False)
    Ls = sorted(lens["J"].keys())
    tok = AutoTokenizer.from_pretrained(hf)

    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16)
    inner = getattr(model, "model", None)
    if inner is not None and getattr(inner, "vision_tower", None) is not None:
        inner.vision_tower = None
        inner.multi_modal_projector = None
    norm_path, norm = get_final_norm(model)
    norm32 = copy.deepcopy(norm).float()
    head = model.get_output_embeddings()
    assert head.bias is None, f"{name}: unembedding has a bias"
    WA = head.weight.detach()[IDS[name]].float().clone()          # (nA, d) on CPU
    Wfull_shape = head.weight.shape
    d = WA.shape[1]
    model = model.to(DEV).eval()
    print(f"  norm {norm_path} ({type(norm).__name__}) d={d} layers={Ls}")

    H = {L: torch.zeros(n_pos, d, dtype=torch.float32) for L in Ls}
    with torch.no_grad():
        for di, items in BY_DOC.items():
            enc = tok(docs[di], return_tensors="pt", truncation=True,
                      max_length=MAX_TOK).to(DEV)
            hs = model(**enc, output_hidden_states=True).hidden_states
            assert hs is not None and len(hs) > max(Ls), f"{name}: bad hidden_states"
            rows = torch.tensor([k for k, _ in items])
            idx = torch.tensor([p[SLOT[name]] for _, p in items], device=DEV)
            for L in Ls:
                H[L][rows] = hs[L][0].index_select(0, idx).float().cpu()

    # gate 0a needs the full head; do it before the model is freed
    with torch.no_grad():
        gg = norm32.to(DEV)(H[Ls[len(Ls) // 2]][:64].to(DEV)
                            @ lens["J"][Ls[len(Ls) // 2]].to(DEV, torch.float32).T)
        Wf = head.weight.detach().float().to(DEV)
        full, restr = gg @ Wf.T, gg @ WA.to(DEV).T
        aid = IDS[name].to(DEV)
        bitdiff = (full[:, aid] - restr).abs().max().item()
        inA = torch.zeros(Wfull_shape[0], dtype=torch.bool, device=DEV)
        inA[aid] = True
        full[:, ~inA] = -1e30
        rank = torch.full((Wfull_shape[0],), -1, dtype=torch.long, device=DEV)
        rank[aid] = torch.arange(nA, device=DEV)
        ref = rank[full.topk(TOPK, 1).indices]
        got = restr.topk(TOPK, 1).indices
        eq = sum(set(ref[i].tolist()) == set(got[i].tolist()) for i in range(64))
        tie = sum(1 for i in range(64)
                  if set(ref[i].tolist()) != set(got[i].tolist())
                  and torch.allclose(restr[i].topk(TOPK).values[-1],
                                     restr[i].topk(TOPK + 1).values[-1]))
        gate0a = (bitdiff == 0.0) and (eq + tie == 64)
        del Wf, full, restr, gg
    assert gate0a, f"{name}: stop condition 2 -- restricted readout is not exact"

    del model, head, norm
    if DEV == "cuda":
        torch.cuda.empty_cache()

    WA = WA.to(DEV)
    norm32 = norm32.to(DEV)
    out = {"layers": Ls, "d": d, "J": {}, "base": {}}

    def topk_of(x, k):
        o = torch.zeros(n_pos, k, dtype=torch.int32)
        for s in range(0, n_pos, 512):
            o[s:s + 512] = (norm32(x[s:s + 512]) @ WA.T).topk(k, 1).indices.int().cpu()
        return o

    with torch.no_grad():
        for L in Ls:
            J = lens["J"][L].to(DEV, torch.float32)
            h = H[L].to(DEV)
            out["J"][L] = topk_of(h @ J.T, TOPK)
            out["base"][L] = topk_of(h, TOPK)
            del J, h
            H[L] = None
    torch.save(out, f"{CACHE}/{name}.pt")
    open(done, "w").write("")
    print(f"  saved {CACHE}/{name}.pt", flush=True)
    del WA, norm32, out, H, lens
    if DEV == "cuda":
        torch.cuda.empty_cache()

