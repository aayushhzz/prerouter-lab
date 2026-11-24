"""Routing-trace container shared by capture, train, sim and ppl.

A trace holds, for every MoE layer L and token t:
  feats[L, t]  the router input of layer L (post-attention RMSNorm output), float16
  topk[L, t]   the k experts layer L's true router picked, int16
plus doc ids (a token's previous/next token only counts inside the same doc).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# OLMoE-1B-7B int4 (group 64): 3 projections x (2048x1024 packed 4-bit + bf16 scales + biases)
OLMOE_EXPERT_BYTES = 3 * (2048 * 1024 // 2 + 2 * (2048 * 1024 // 64) * 2)


@dataclass
class Trace:
    feats: np.ndarray  # (n_layers, T, D) float16
    topk: np.ndarray  # (n_layers, T, k) int16
    doc: np.ndarray  # (T,) int32
    tokens: np.ndarray  # (T,) int32 (token ids; -1 for synthetic)
    gate_w: np.ndarray  # (n_layers, E, D) float32 router weights (zeros if unknown)
    meta: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.feats.shape[0]

    @property
    def n_experts(self) -> int:
        return int(self.meta["n_experts"])

    @property
    def k(self) -> int:
        return self.topk.shape[2]

    @property
    def T(self) -> int:
        return self.feats.shape[1]

    def heldout_start(self) -> int:
        """First token index of the held-out split (always on a doc boundary)."""
        return int(self.meta["heldout_start"])

    def save(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        np.savez(
            os.path.join(data_dir, "trace.npz"),
            feats=self.feats, topk=self.topk, doc=self.doc, tokens=self.tokens, gate_w=self.gate_w,
        )
        with open(os.path.join(data_dir, "trace_meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    @staticmethod
    def load(data_dir: str) -> "Trace":
        z = np.load(os.path.join(data_dir, "trace.npz"))
        with open(os.path.join(data_dir, "trace_meta.json")) as f:
            meta = json.load(f)
        return Trace(z["feats"], z["topk"], z["doc"], z["tokens"], z["gate_w"], meta)


def split_point(doc: np.ndarray, frac: float = 0.7) -> int:
    """Token index of the first doc that starts at or after `frac` of the stream."""
    starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    target = frac * len(doc)
    later = starts[starts >= target]
    return int(later[0]) if len(later) else int(starts[-1])


def synthetic_trace(
    n_layers: int = 16,
    n_experts: int = 64,
    k: int = 8,
    d: int = 64,
    n_tokens: int = 3072,
    doc_len: int = 256,
    zipf_a: float = 1.1,
    rho: float = 0.85,
    noise: float = 1.0,
    seed: int = 0,
    expert_bytes: int = OLMOE_EXPERT_BYTES,
    reason: str = "requested",
) -> Trace:
    """Zipf-popularity + temporal-locality routing trace with a planted cross-token signal.

    Hidden states follow an AR(1) walk (locality: adjacent tokens route alike). Layer L's
    logits at token t are a fixed linear map of layer L-1's state at token t-1, plus a Zipf
    log-popularity bias and noise, so a one-token-ahead linear prerouter can learn them.
    """
    rng = np.random.default_rng(seed)
    T = n_tokens
    doc = (np.arange(T) // doc_len).astype(np.int32)
    h = np.zeros((n_layers, T, d), np.float32)
    for L in range(n_layers):
        eps = rng.standard_normal((T, d)).astype(np.float32)
        for t in range(T):
            if t == 0 or doc[t] != doc[t - 1]:
                h[L, t] = eps[t]
            else:
                h[L, t] = rho * h[L, t - 1] + np.sqrt(1 - rho**2) * eps[t]
    pop = np.log(1.0 / np.arange(1, n_experts + 1) ** zipf_a)
    topk = np.zeros((n_layers, T, k), np.int16)
    for L in range(n_layers):
        perm = rng.permutation(n_experts)  # each layer has its own popular experts
        A = rng.standard_normal((d, n_experts)).astype(np.float32) / np.sqrt(d) * 2.0
        src = np.zeros_like(h[L])
        if L == 0:
            src = h[0]
        else:
            src[1:] = h[L - 1, :-1]
            src[0] = h[L - 1, 0]
        logits = src @ A + pop[perm] + noise * rng.standard_normal((T, n_experts))
        topk[L] = np.argsort(-logits, axis=1)[:, :k]
    meta = {
        "source": "synthetic",
        "synthetic_reason": reason,
        "model": None,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "k": k,
        "d": d,
        "expert_bytes": expert_bytes,
        "nonexpert_bytes": 300 * 2**20,
        "compute_ms_per_token": 15.0,
        "compute_ms_source": "assumed (synthetic trace)",
        "n_tokens": T,
        "heldout_start": split_point(doc),
        "docs": [{"name": "synthetic-%d" % i, "n_tokens": doc_len} for i in range(int(doc.max()) + 1)],
    }
    return Trace(h.astype(np.float16), topk, doc, -np.ones(T, np.int32),
                 np.zeros((n_layers, n_experts, d), np.float32), meta)


def load_or_none(data_dir: str) -> Optional[Trace]:
    return Trace.load(data_dir) if os.path.exists(os.path.join(data_dir, "trace.npz")) else None
