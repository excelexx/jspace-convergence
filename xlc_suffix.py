"""Text-model suffix machinery for lens refitting — hook-capture approach.

Used by xlc_phase1.py, which refits the J-lenses on disjoint halves of
WikiText-103 for the lens-fitting control in Appendix C.1. To take a Jacobian
of the readout with respect to a hidden state at layer L, we need a callable
that maps that hidden state to the model's final state.

Text decoder blocks need family-specific extras (rotary embeddings, causal or
sliding-window masks, cache positions). Instead of reconstructing them per
family, a probe forward captures each block's exact (args, kwargs) via
forward-pre-hooks; the suffix replays blocks L..end with those captured
extras, substituting our hidden state. Masks and rotary tensors depend only
on sequence length, so one capture at fixed T serves every fitting document.

The Jacobian target is the PRE-final-norm residual (the lens convention:
readout is W_U · norm(J h), so J maps to the pre-norm state) — captured as
the last block's own output, NOT hidden_states[-1], which is post-norm in
these architectures.

WIRING GATE: suffix(hs[L]) must reproduce the captured pre-norm final state
on the probe document (rel < 1e-4 in fp32) before any Jacobian accumulates."""
import torch

DROP_KWARGS = {"hidden_states", "past_key_value", "past_key_values",
               "layer_past", "use_cache", "output_attentions"}


def get_text_blocks(model):
    for attr in ("transformer", "gpt_neox", "model"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            break
    else:
        raise RuntimeError("no inner transformer")
    for attr in ("h", "layers"):
        if hasattr(inner, attr):
            return getattr(inner, attr)
    raise RuntimeError("no block list")


class TextSuffixFactory:
    """Capture per-block call extras once (fixed T), then build suffixes."""

    def __init__(self, model, ids):
        self.blocks = get_text_blocks(model)
        self.captured = {}
        self.final_prenorm = None

        def pre(i):
            def fn(module, args, kwargs):
                self.captured[i] = (args, kwargs)
            return fn

        def post(module, args, out):
            o = out[0] if isinstance(out, tuple) else out
            self.final_prenorm = o.detach()

        hs_pre = [b.register_forward_pre_hook(pre(i), with_kwargs=True)
                  for i, b in enumerate(self.blocks)]
        h_post = self.blocks[-1].register_forward_hook(post)
        with torch.no_grad():
            self.hs = model(**ids, output_hidden_states=True,
                            use_cache=False).hidden_states
        for h in hs_pre:
            h.remove()
        h_post.remove()
        assert len(self.captured) == len(self.blocks)
        self.T = self.hs[0].shape[1]

    def suffix(self, L):
        blocks, captured = self.blocks, self.captured

        def fn(H):
            x = H.unsqueeze(0)
            for i in range(L, len(blocks)):
                args, kwargs = captured[i]
                kw = {k: v for k, v in kwargs.items() if k not in DROP_KWARGS}
                out = blocks[i](x, *args[1:], **kw)
                x = out[0] if isinstance(out, tuple) else out
            return x.squeeze(0)
        return fn

    def wiring_gate(self, L):
        with torch.no_grad():
            out = self.suffix(L)(self.hs[L][0])
        ref = self.final_prenorm[0]
        return ((out - ref).norm() / ref.norm()).item()
