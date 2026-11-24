import numpy as np
import pytest

from prerouter_lab import prerouter, sim
from prerouter_lab.trace import synthetic_trace


@pytest.fixture(scope="module")
def replay():
    tr = synthetic_trace(n_layers=6, n_experts=32, k=4, d=32, n_tokens=1024, doc_len=256, seed=1)
    out = prerouter.train_all(tr, steps=80, log=lambda *a: None)
    lo, hi = tr.heldout_start(), tr.T
    true = [[frozenset(int(e) for e in tr.topk[L, t]) for L in range(tr.n_layers)] for t in range(lo, hi)]
    pred = prerouter.predicted_sets(tr, out["W"], out["b"], lo, hi)
    return tr, true, pred


@pytest.mark.parametrize("cap_mult", [1, 2, 4])
def test_prediction_as_routing_has_zero_wasted_reads(replay, cap_mult):
    tr, true, pred = replay
    cap = tr.n_layers * tr.k * cap_mult
    r = sim.simulate(true, pred, "par", cap, read_ms=1.0, compute_ms=10.0, expert_bytes=100)
    assert r["prefetch_reads"] > 0          # it really staged experts ahead of time
    assert r["wasted_reads"] == 0           # every staged expert was routed to
    assert r["staged_used"] == r["prefetch_reads"]


def test_predicted_prefetch_wastes_reads_when_prediction_is_imperfect(replay):
    tr, true, pred = replay
    r = sim.simulate(true, pred, "prefetch", tr.n_layers * tr.k * 2, 1.0, 10.0, 100)
    assert r["wasted_reads"] > 0
    assert r["staged_used"] + r["wasted_reads"] == r["prefetch_reads"]


def test_lru_reads_only_cold_misses_when_everything_fits(replay):
    tr, true, pred = replay
    r = sim.simulate(true, pred, "lru", tr.n_layers * tr.n_experts, 1.0, 10.0, 100)
    unique = {(L, e) for row in true for L, s in enumerate(row) for e in s}
    assert r["demand_reads"] == len(unique) and r["prefetch_reads"] == 0


def test_exposed_prefetch_only_counts_time_above_compute():
    # 2 layers, 1 expert each, cache fits everything; prediction is perfect for layer 1.
    true = [[frozenset({0}), frozenset({t})] for t in range(4)]
    pred = [[None, None]] + [[None, frozenset({t})] for t in range(1, 4)]
    hidden = sim.simulate(true, pred, "par", 64, read_ms=5.0, compute_ms=10.0, expert_bytes=1)
    exposed = sim.simulate(true, pred, "par", 64, read_ms=15.0, compute_ms=10.0, expert_bytes=1)
    # demand: token 0 reads layer0 e0 and layer1 e0; later tokens prefetch 1 expert each
    assert hidden["demand_reads"] == 2 and hidden["prefetch_reads"] == 3
    assert hidden["stall_ms_per_token"] == pytest.approx(2 * 5.0 / 4)
    assert exposed["stall_ms_per_token"] == pytest.approx((2 * 15.0 + 3 * 5.0) / 4)


def test_capacities_span_working_set_to_full_model():
    caps = sim.capacities(16, 8, 64)
    assert caps[0] == 128 and caps[-1] == 1024 and len(caps) >= 4


def test_routing_stats_counts_distinct_experts():
    true = [[frozenset({0, 1}), frozenset({2, 3})], [frozenset({0, 1}), frozenset({4, 5})]]
    pred = [[None, None], [None, frozenset({2, 3})]]
    st = sim.routing_stats(true, pred)
    assert st["unique_experts_per_layer_true"] == 2 and st["unique_experts_per_layer_pred"] == 2
    assert st["adjacent_token_overlap_true"] == (1.0 + 0.0) / 2
