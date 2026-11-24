"""Layer-by-layer forward of an mlx-lm OLMoE-style MoE with pluggable routing.

We re-run the decoder blocks ourselves (same modules, same math as mlx_lm.models.olmoe) so we
can (a) log every layer's router input and true top-k and (b) replace a layer's routing with
the prerouter's prediction made from the previous layer one token earlier.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

from . import DEFAULT_MODEL  # noqa: F401  (re-export)


def load(repo: str = DEFAULT_MODEL):
    from mlx_lm import load as mlx_load

    model, tok = mlx_load(repo)
    return model, tok


def model_dir(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True)


def moe_info(model) -> Dict:
    a = model.args
    return {"n_layers": a.num_hidden_layers, "n_experts": a.num_experts, "k": a.num_experts_per_tok,
            "d": a.hidden_size, "norm_topk_prob": bool(getattr(a, "norm_topk_prob", False))}


def _topk(probs: mx.array, k: int) -> mx.array:
    return mx.stop_gradient(mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k])


def forward(model, tokens: List[int], mode: str = "true", W: Optional[np.ndarray] = None,
            b: Optional[np.ndarray] = None, record: bool = False):
    """Run one sequence. mode:
      true        each layer routes by its own router (the reference model)
      pred        Edge0 prediction-as-routing: layers >= 1 take indices AND weights from the
                  prerouter's logits (softmax-top-k, same math as the router); token 0 is true
      pred_gatew  indices from the prerouter, mixing weights from the layer's own router
    Returns (logits[S, V], records) where records = [(xn float16 np, topk int16 np)] per layer.
    """
    from mlx_lm.models.base import create_attention_mask

    inner = model.model
    info = moe_info(model)
    E, k, norm = info["n_experts"], info["k"], info["norm_topk_prob"]
    x = inner.embed_tokens(mx.array(tokens)[None])
    mask = create_attention_mask(x, None)
    S = len(tokens)
    records = []
    prev_feat = None  # previous layer's prerouter input, (S, D+2E)
    for L, layer in enumerate(inner.layers):
        h = x + layer.self_attn(layer.input_layernorm(x), mask, None)
        xn = layer.post_attention_layernorm(h).reshape(S, -1)
        moe = layer.mlp
        probs = mx.softmax(moe.gate(xn), axis=-1, precise=True)
        if mode != "true" and L >= 1 and prev_feat is not None:
            plog = prev_feat @ mx.array(W[L - 1]) + mx.array(b[L - 1])  # predicts layer L at t+1
            pprobs = mx.softmax(plog[:-1], axis=-1, precise=True)
            pprobs = mx.concatenate([probs[:1], pprobs], axis=0)  # token 0: no prediction yet
            idx = _topk(pprobs, k)
            scores = mx.take_along_axis(pprobs if mode == "pred" else probs, idx, axis=-1)
        else:
            idx = _topk(probs, k)
            scores = mx.take_along_axis(probs, idx, axis=-1)
        if norm:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = moe.switch_mlp(xn, idx)
        y = (y * scores[..., None]).sum(axis=-2)
        x = h + y.reshape(1, S, -1)
        if mode != "true" or record:
            oh = mx.zeros((S, E)).at[mx.arange(S)[:, None], idx].add(1.0)
            ohp = mx.concatenate([mx.zeros((1, E)), oh[:-1]], axis=0)
            prev_feat = mx.concatenate([xn.astype(mx.float32), oh, ohp], axis=1)
        if record:
            mx.eval(xn, idx)
            records.append((np.array(xn.astype(mx.float16)), np.array(idx.astype(mx.int16))))
    out = inner.norm(x)
    if model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(out)
    else:
        logits = model.lm_head(out)
    return logits[0], records


def nll(logits: mx.array, tokens: List[int]) -> mx.array:
    """Summed next-token NLL over positions 0..S-2."""
    lp = logits[:-1].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    tgt = mx.array(tokens[1:])[:, None]
    return -mx.take_along_axis(lp, tgt, axis=-1).sum()


def gate_weights(model) -> np.ndarray:
    """(n_layers, E, D) float32 router weights (dequantized if the gate is quantized)."""
    out = []
    for layer in model.model.layers:
        g = layer.mlp.gate
        D = model.args.hidden_size
        Wt = g(mx.eye(D, dtype=mx.float32))  # (D, E) = W^T, works for Linear and QuantizedLinear
        out.append(np.array(Wt.T.astype(mx.float32)))
    return np.stack(out)


def measure_decode_ms(model, tok, prompt: str = "def fibonacci(n):", n_tokens: int = 64, runs: int = 3) -> float:
    """Resident decode speed of the same model (median of warm runs): our per-token compute time."""
    from mlx_lm import stream_generate

    tps = []
    for i in range(runs + 1):  # run 0 is warm-up: first touch pages the mmapped weights in
        for r in stream_generate(model, tok, prompt, max_tokens=n_tokens):
            pass
        if i:
            tps.append(r.generation_tps)
    return 1000.0 / sorted(tps)[len(tps) // 2]

