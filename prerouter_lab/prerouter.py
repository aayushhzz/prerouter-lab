"""Per-layer linear prerouter: layer L's state at token t -> layer L+1's top-k at token t+1.

Head owned by layer L (paper section 3.2, simplified to linear):
  x = [ feats_L(t) , onehot(topk_L(t)) , onehot(topk_L(t-1)) ]      dim D + 2E
  logits_{L+1}(t+1) = x @ W_L + b_L
W_L is warm-started from layer L+1's router weight on the hidden-state block ("apply the next
router directly to this hidden state") and trained with softmax cross-entropy against the
true top-k set, with L2 pulling W back toward that warm start (we only learn the correction).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .trace import Trace


def onehot_sets(topk: np.ndarray, E: int) -> np.ndarray:
    """(T, k) expert ids -> (T, E) multi-hot float32."""
    out = np.zeros((topk.shape[0], E), np.float32)
    np.put_along_axis(out, topk.astype(np.int64), 1.0, axis=1)
    return out


def layer_features(trace: Trace, L: int) -> np.ndarray:
    """(T, D+2E) prerouter input for head L at every token (prev one-hot is 0 at doc starts)."""
    E = trace.n_experts
    cur = onehot_sets(trace.topk[L], E)
    prev = np.zeros_like(cur)
    same = trace.doc[1:] == trace.doc[:-1]
    prev[1:][same] = cur[:-1][same]
    return np.concatenate([trace.feats[L].astype(np.float32), cur, prev], axis=1)


def pair_rows(trace: Trace, lo: int, hi: int) -> np.ndarray:
    """Token indices t in [lo, hi) where t+1 is in the same doc and also < hi."""
    t = np.arange(lo, hi - 1)
    return t[trace.doc[t] == trace.doc[t + 1]]


def topk_idx(logits: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-logits, axis=1)[:, :k]


def recall_at_k(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean |pred ∩ true| / k over rows; both (N, k) expert ids."""
    if len(true) == 0:
        return float("nan")
    k = true.shape[1]
    hits = (pred[:, :, None] == true[:, None, :]).any(axis=2).sum(axis=1)
    return float(hits.mean() / k)


def train_head(X: np.ndarray, Y: np.ndarray, E: int, W0: np.ndarray,
               steps: int = 300, lr: float = 3e-3, l2: float = 1e-3, seed: int = 0):
    """Full-batch Adam on softmax CE vs uniform-over-true-top-k targets. Returns (W, b)."""
    k = Y.shape[1]
    target = onehot_sets(Y, E) / k
    W = W0.copy()
    b = np.zeros(E, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(X)
    for i in range(1, steps + 1):
        z = X @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - target) / n
        gW = X.T @ g + l2 * (W - W0)
        gb = g.sum(axis=0)
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW * gW
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * gb * gb
        c1, c2 = 1 - b1**i, 1 - b2**i
        W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
        b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
    return W.astype(np.float32), b.astype(np.float32)


def warm_start(trace: Trace, L: int) -> np.ndarray:
    D, E = trace.feats.shape[2], trace.n_experts
    W0 = np.zeros((D + 2 * E, E), np.float32)
    W0[:D] = trace.gate_w[L + 1].T  # next layer's router applied to this layer's state
    return W0


def train_all(trace: Trace, steps: int = 600, lr: float = 3e-3, l2: float = 1e-2,
              log=print) -> Dict:
    """Train heads 0..n-2; return weights plus held-out recall@k for the head and baselines."""
    E, k, split = trace.n_experts, trace.k, trace.heldout_start()
    tr_rows = pair_rows(trace, 0, split)
    te_rows = pair_rows(trace, split, trace.T)
    Ws, bs, per_layer = [], [], []
    has_gate = bool(np.abs(trace.gate_w).sum() > 0)
    t0 = time.time()
    for L in range(trace.n_layers - 1):
        X = layer_features(trace, L)
        Y = trace.topk[L + 1]
        W0 = warm_start(trace, L)
        W, b = train_head(X[tr_rows], Y[tr_rows + 1], E, W0, steps=steps, lr=lr, l2=l2)
        Ws.append(W)
        bs.append(b)
        true = Y[te_rows + 1]
        counts = np.bincount(Y[tr_rows + 1].ravel().astype(np.int64), minlength=E)
        freq = np.argsort(-counts, kind="stable")[:k]
        row = {
            "head": L,
            "target_layer": L + 1,
            "prerouter": recall_at_k(topk_idx(X[te_rows] @ W + b, k), true),
            "prev_token": recall_at_k(Y[te_rows], true),  # layer L+1's experts at token t
            "most_frequent": recall_at_k(np.tile(freq, (len(true), 1)), true),
            "prerouter_train": recall_at_k(topk_idx(X[tr_rows] @ W + b, k), Y[tr_rows + 1]),
        }
        if has_gate:
            row["next_router_zero_shot"] = recall_at_k(topk_idx(X[te_rows] @ W0, k), true)
        per_layer.append(row)
        log("head %2d -> layer %2d  recall@%d  prerouter %.3f | prev-token %.3f | most-frequent %.3f%s"
            % (L, L + 1, k, row["prerouter"], row["prev_token"], row["most_frequent"],
               (" | next-router 0-shot %.3f" % row["next_router_zero_shot"]) if has_gate else ""))
    keys = [c for c in per_layer[0] if c not in ("head", "target_layer")]
    mean = {c: float(np.mean([r[c] for r in per_layer])) for c in keys}
    return {
        "W": np.stack(Ws), "b": np.stack(bs),
        "results": {"k": k, "n_train_pairs": int(len(tr_rows)), "n_heldout_pairs": int(len(te_rows)),
                    "per_layer": per_layer, "mean": mean, "train_seconds": time.time() - t0,
                    "source": trace.meta.get("source"), "steps": steps, "l2": l2, "lr": lr},
    }


def save(out: Dict, data_dir: str) -> None:
    np.savez(os.path.join(data_dir, "prerouter.npz"), W=out["W"], b=out["b"])
    with open(os.path.join(data_dir, "train_results.json"), "w") as f:
        json.dump(out["results"], f, indent=2)


def load(data_dir: str):
    z = np.load(os.path.join(data_dir, "prerouter.npz"))
    return z["W"], z["b"]


def predicted_sets(trace: Trace, W: np.ndarray, b: np.ndarray, lo: int, hi: int) -> List[List[Optional[frozenset]]]:
    """pred[t-lo][L] = experts the prerouter stages for layer L at token t (None when no
    prediction exists: layer 0, or the first token of a doc)."""
    k = trace.k
    out: List[List[Optional[frozenset]]] = [[None] * trace.n_layers for _ in range(hi - lo)]
    for L in range(trace.n_layers - 1):
        X = layer_features(trace, L)[lo:hi]
        idx = topk_idx(X @ W[L] + b[L], k)
        for t in range(lo + 1, hi):
            if trace.doc[t] == trace.doc[t - 1]:
                out[t - lo][L + 1] = frozenset(int(e) for e in idx[t - 1 - lo])
    return out
