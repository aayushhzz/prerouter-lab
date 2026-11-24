"""`pr capture`: log (layer-L router input, layer-L top-k) for every MoE layer of a real MLX MoE.

Falls back to the synthetic Zipf + locality generator if the checkpoint will not load, and
records why in trace_meta.json so `pr report` and the README can flag it.
"""
from __future__ import annotations

import glob
import os
import sysconfig
import traceback
from typing import List, Tuple

import numpy as np

from .trace import Trace, split_point, synthetic_trace

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
# Stable, locally available code (CPython stdlib, PSF license) to mix with the agent transcripts.
STDLIB_CODE = ["textwrap.py", "heapq.py", "shlex.py", "bisect.py", "json/decoder.py", "fnmatch.py"]


def corpus_docs() -> List[Tuple[str, str]]:
    """Agent transcripts interleaved with code, so both splits see both kinds of text."""
    agents = [(os.path.basename(p), open(p).read()) for p in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))]
    stdlib = sysconfig.get_paths()["stdlib"]
    code = []
    for rel in STDLIB_CODE:
        p = os.path.join(stdlib, rel)
        if os.path.exists(p):
            src = open(p).read()
            i = min([j for j in (src.find("\ndef "), src.find("\nclass ")) if j >= 0] or [0])
            code.append(("stdlib:" + rel, src[i:]))
    docs = []
    while agents or code:
        if agents:
            docs.append(agents.pop(0))
        if code:
            docs.append(code.pop(0))
    return docs


def capture_real(repo: str, max_doc_tokens: int = 384, max_tokens: int = 3200, log=print) -> Trace:
    import mlx.core as mx

    from . import mlx_model

    model, tok = mlx_model.load(repo)
    info = mlx_model.moe_info(model)
    log("loaded %s: %d layers, %d experts, top-%d, d=%d" % (repo, info["n_layers"], info["n_experts"], info["k"], info["d"]))
    feats = [[] for _ in range(info["n_layers"])]
    topk = [[] for _ in range(info["n_layers"])]
    doc_ids, tokens, docs_meta = [], [], []
    for name, text in corpus_docs():
        ids = tok.encode(text)[:max_doc_tokens]
        if len(tokens) + len(ids) > max_tokens:
            ids = ids[: max_tokens - len(tokens)]
        if len(ids) < 16:
            break
        _, rec = mlx_model.forward(model, ids, record=True)
        for L, (xn, idx) in enumerate(rec):
            feats[L].append(xn)
            topk[L].append(idx)
        doc_ids += [len(docs_meta)] * len(ids)
        tokens += ids
        docs_meta.append({"name": name, "n_tokens": len(ids)})
        log("  captured %-28s %4d tokens" % (name, len(ids)))
    doc = np.array(doc_ids, np.int32)
    compute_ms = mlx_model.measure_decode_ms(model, tok)
    log("resident decode: %.1f ms/token (%.1f tok/s)" % (compute_ms, 1000 / compute_ms))
    mdir = mlx_model.model_dir(repo)
    ckpt_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(mdir, "*.safetensors")))
    from .ssd import expert_nbytes_from_checkpoint

    expert_bytes = expert_nbytes_from_checkpoint(mdir)
    all_expert = expert_bytes * info["n_layers"] * info["n_experts"]
    meta = {
        "source": "mlx",
        "model": repo,
        "model_dir": mdir,
        "n_layers": info["n_layers"],
        "n_experts": info["n_experts"],
        "k": info["k"],
        "d": info["d"],
        "expert_bytes": expert_bytes,
        "checkpoint_bytes": ckpt_bytes,
        "nonexpert_bytes": max(0, ckpt_bytes - all_expert),
        "compute_ms_per_token": compute_ms,
        "compute_ms_source": "measured: resident mlx-lm decode of the same checkpoint",
        "n_tokens": len(tokens),
        "heldout_start": split_point(doc),
        "docs": docs_meta,
        "peak_memory_gb": mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else None,
    }
    trace = Trace(
        np.stack([np.concatenate(f) for f in feats]),
        np.stack([np.concatenate(t) for t in topk]),
        doc, np.array(tokens, np.int32), mlx_model.gate_weights(model), meta,
    )
    return trace


def capture(repo: str, data_dir: str, synthetic: bool = False, max_tokens: int = 3200, log=print) -> Trace:
    if synthetic:
        trace = synthetic_trace(reason="requested with --synthetic")
    else:
        try:
            trace = capture_real(repo, max_tokens=max_tokens, log=log)
        except Exception as e:  # any load/forward failure -> clearly flagged synthetic fallback
            log(traceback.format_exc(limit=3))
            log("CAPTURE FAILED (%s: %s) -> falling back to SYNTHETIC traces" % (type(e).__name__, e))
            trace = synthetic_trace(reason="capture from %s failed: %s: %s" % (repo, type(e).__name__, e))
    trace.save(data_dir)
    m = trace.meta
    log("trace: source=%s tokens=%d (train %d / held-out %d) layers=%d experts=%d k=%d -> %s"
        % (m["source"], trace.T, trace.heldout_start(), trace.T - trace.heldout_start(),
           trace.n_layers, trace.n_experts, trace.k, os.path.join(data_dir, "trace.npz")))
    return trace
