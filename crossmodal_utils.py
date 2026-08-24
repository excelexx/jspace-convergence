"""Load pilot functions/constants from step7_align.py WITHOUT executing its
pipeline (step7 is a flat script that runs on import). We extract the wanted
top-level defs/assigns via AST and exec only those, so the pilot conventions
(m-NN, NNLS, NOMP, norm handling, chunked scoring) are reused verbatim by the
text-vision stages and the lens-fitting control, never re-derived.
step7_align.py itself is never modified.

The extracted names are exec'd into one shared namespace, so _WANTED must
also list the helpers the wanted functions call internally (e.g. best_atom
under nnomp_batch) — dropping a name silently breaks the caller."""
import ast
import hashlib
import types

PILOT_PATH = "step7_align.py"

# constants + functions the cross-modal stages are allowed to use
_WANTED = {
    "DEV", "K_SPARSE", "K_NN", "V_CHUNK",
    "MODELS", "NORM_PATHS",
    "get_WU_and_w", "nnls_refit", "best_atom", "nnomp_batch",
    "prep", "neighbors", "mnn",
}

# the five models the lens-fitting control covers, and the 10 pairs among them
LENSFIT_SCOPE = ["pythia70m", "gpt2", "gemma270", "qwen08b", "gemma"]


def _target_names(node):
    names = []
    for t in node.targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, ast.Tuple):
            names += [e.id for e in t.elts if isinstance(e, ast.Name)]
    return names

def load_pilot(dev=None):
    src = open(PILOT_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            keep.append(node)
        elif isinstance(node, ast.Assign) and set(_target_names(node)) & _WANTED:
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), PILOT_PATH, "exec"), ns)
    missing = _WANTED - set(ns)
    assert not missing, f"pilot symbols not found in {PILOT_PATH}: {missing}"
    if dev is not None:
        ns["DEV"] = dev                          # functions read module-global DEV
    return types.SimpleNamespace(**{k: v for k, v in ns.items()
                                    if not k.startswith("__")})


def wikitext_docs():
    """WikiText-103 train, cut into documents on top-level headings, dropping
    documents under 200 characters and deduplicating by content hash. The
    returned order is what half_manifest.json's h1/h2 indices refer to, so
    xlc_phase0.py and xlc_phase1.py must both build the list from here."""
    from datasets import load_dataset

    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    docs, cur = [], []
    for row in wt:
        t = row["text"]
        if t.startswith(" = ") and not t.startswith(" = = ") and cur:
            d = "".join(cur).strip()
            if len(d) > 200:
                docs.append(d)
            cur = []
        cur.append(t)
    d = "".join(cur).strip()
    if len(d) > 200:
        docs.append(d)
    seen, unique = set(), []
    for d in docs:                                # corpus contains duplicates
        h = hashlib.sha256(d.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(d)
    return unique
