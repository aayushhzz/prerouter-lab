"""Replay routing traces through an expert cache under three policies.

Timing model per decode token (a deliberately small model, see PAPER_NOTES.md):
  - compute takes `compute_ms` per token (measured from a resident MLX decode);
  - every expert read costs `read_ms` (= expert bytes / measured QD1 SSD bandwidth);
  - a demand miss is read on the critical path: it adds read_ms to the stall;
  - prefetch reads for token t+1 are issued during token t and overlap its compute, so only
    the part above compute_ms is exposed as stall at token t+1.

Policies:
  lru       on-demand LRU: route by the true router, read misses when routed.
  prefetch  LRU + predicted prefetch: stage the prerouter's prediction one token ahead, but
            still route by the true router (mispredictions = wasted reads + demand misses).
  par       prediction-as-routing (Edge0): stage the prediction AND route by it, so the staged
            set is the routed set by construction (layer 0 / doc starts route truly on demand).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

POLICIES = ("lru", "prefetch", "par")
POLICY_LABELS = {"lru": "On-demand LRU", "prefetch": "LRU + predicted prefetch",
                 "par": "Prediction-as-routing (Edge0)"}

Sets = Sequence[Sequence[Optional[frozenset]]]


class ExpertCache:
    """LRU over (layer, expert) keys; staged keys are pinned until their token consumes them."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lru: "OrderedDict[tuple, None]" = OrderedDict()
        self.pinned: set = set()

    def __contains__(self, key) -> bool:
        return key in self.lru

    def touch(self, key) -> None:
        self.lru.move_to_end(key)

    def insert(self, key, pin: bool = False) -> None:
        self.lru[key] = None
        if pin:
            self.pinned.add(key)
        while len(self.lru) > self.capacity:
            victim = next((v for v in self.lru if v not in self.pinned), None)
            if victim is None:  # everything pinned: capacity below one token's working set
                raise ValueError("cache capacity %d too small for staged set" % self.capacity)
            del self.lru[victim]


def simulate(true_sets: Sets, pred_sets: Sets, policy: str, capacity: int,
             read_ms: float, compute_ms: float, expert_bytes: int) -> Dict:
    """true_sets[t][L] / pred_sets[t][L]: frozensets of expert ids (pred may be None)."""
    assert policy in POLICIES
    T, n_layers = len(true_sets), len(true_sets[0])
    cache = ExpertCache(capacity)
    demand = prefetch = wasted = staged_used = 0
    stall_ms = 0.0
    exposed_next = 0.0
    staged: Dict[tuple, None] = {}
    for t in range(T):
        stall_ms += exposed_next
        exposed_next = 0.0
        for L in range(n_layers):
            pred = pred_sets[t][L]
            routed = pred if (policy == "par" and pred is not None) else true_sets[t][L]
            for e in routed:
                key = (L, e)
                if key in cache:
                    cache.touch(key)
                else:
                    demand += 1
                    stall_ms += read_ms
                    cache.insert(key)
            for key in [s for s in staged if s[0] == L]:
                if key[1] in routed:
                    staged_used += 1
                else:
                    wasted += 1
                cache.pinned.discard(key)
                del staged[key]
        if policy != "lru" and t + 1 < T:
            n = 0
            for L in range(n_layers):
                pred = pred_sets[t + 1][L]
                for e in pred or ():
                    key = (L, e)
                    if key not in cache:
                        cache.insert(key, pin=True)
                        staged[key] = None
                        n += 1
            prefetch += n
            exposed_next = max(0.0, n * read_ms - compute_ms)
    reads = demand + prefetch
    stall_per_tok = stall_ms / T
    return {
        "policy": policy,
        "capacity": capacity,
        "tokens": T,
        "demand_reads": demand,
        "prefetch_reads": prefetch,
        "wasted_reads": wasted,
        "staged_used": staged_used,
        "bytes_per_token": reads * expert_bytes / T,
        "stall_ms_per_token": stall_per_tok,
        "tok_s": 1000.0 / (compute_ms + stall_per_tok),
    }


def capacities(n_layers: int, k: int, n_experts: int) -> List[int]:
    """Cache sizes from one token's working set up to the whole model."""
    base = n_layers * k
    total = n_layers * n_experts
    caps = sorted({min(total, int(base * f)) for f in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)} | {total})
    return caps


def sweep(true_sets: Sets, pred_sets: Sets, caps: List[int], read_ms: float, compute_ms: float,
          expert_bytes: int, nonexpert_bytes: int) -> List[Dict]:
    rows = []
    for c in caps:
        for p in POLICIES:
            r = simulate(true_sets, pred_sets, p, c, read_ms, compute_ms, expert_bytes)
            r["ram_gib"] = (c * expert_bytes + nonexpert_bytes) / 2**30
            rows.append(r)
    return rows


def routing_stats(true_sets: Sets, pred_sets: Sets) -> Dict:
    """How spread out the routed experts are, true router vs prerouter, on predicted layers."""
    T, n_layers = len(true_sets), len(true_sets[0])
    rows = [(t, L) for t in range(T) for L in range(n_layers) if pred_sets[t][L] is not None]
    layers = sorted({L for _, L in rows})
    uniq_true = [len(set().union(*(true_sets[t][L] for t, l in rows if l == L))) for L in layers]
    uniq_pred = [len(set().union(*(pred_sets[t][L] for t, l in rows if l == L))) for L in layers]
    k = len(true_sets[0][0])
    adj = [len(true_sets[t][L] & true_sets[t - 1][L]) / k for t in range(1, T) for L in range(n_layers)]
    both = [(t, L) for t, L in rows if t > 0 and pred_sets[t - 1][L] is not None]
    adj_pred = [len(pred_sets[t][L] & pred_sets[t - 1][L]) / k for t, L in both]
    return {"unique_experts_per_layer_true": sum(uniq_true) / len(layers),
            "unique_experts_per_layer_pred": sum(uniq_pred) / len(layers),
            "adjacent_token_overlap_true": sum(adj) / len(adj),
            "adjacent_token_overlap_pred": sum(adj_pred) / len(adj_pred) if adj_pred else float("nan")}
