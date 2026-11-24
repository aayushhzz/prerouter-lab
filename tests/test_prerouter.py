import numpy as np

from prerouter_lab import prerouter
from prerouter_lab.trace import split_point, synthetic_trace


def test_recall_at_k_counts_set_overlap():
    pred = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    true = np.array([[3, 2, 9, 9], [7, 6, 5, 4]])
    assert prerouter.recall_at_k(pred, true) == (2 / 4 + 4 / 4) / 2


def test_features_have_current_and_previous_routing_and_reset_at_doc_start():
    tr = synthetic_trace(n_layers=3, n_experts=8, k=2, d=4, n_tokens=20, doc_len=10)
    X = prerouter.layer_features(tr, 0)
    assert X.shape == (20, 4 + 2 * 8)
    cur, prev = X[:, 4:12], X[:, 12:]
    assert (cur.sum(1) == 2).all()
    assert prev[0].sum() == 0 and prev[10].sum() == 0      # doc starts
    np.testing.assert_array_equal(prev[5], cur[4])


def test_pairs_never_cross_doc_boundary():
    tr = synthetic_trace(n_layers=2, n_experts=8, k=2, d=4, n_tokens=30, doc_len=10)
    rows = prerouter.pair_rows(tr, 0, 30)
    assert 9 not in rows and 19 not in rows and 29 not in rows


def test_split_lands_on_doc_boundary():
    doc = np.repeat(np.arange(8), 10)
    assert split_point(doc, 0.75) == 60


def test_prerouter_beats_baselines_on_planted_signal():
    tr = synthetic_trace(n_layers=4, n_experts=32, k=4, d=32, n_tokens=2048, doc_len=256, seed=2)
    out = prerouter.train_all(tr, steps=150, log=lambda *a: None)
    m = out["results"]["mean"]
    assert m["prerouter"] > m["prev_token"] > m["most_frequent"]
    assert out["W"].shape == (3, 32 + 2 * 32, 32)


def test_predicted_sets_none_for_layer0_and_doc_start():
    tr = synthetic_trace(n_layers=3, n_experts=8, k=2, d=4, n_tokens=40, doc_len=20)
    out = prerouter.train_all(tr, steps=5, log=lambda *a: None)
    sets = prerouter.predicted_sets(tr, out["W"], out["b"], 20, 40)
    assert sets[0] == [None, None, None]                   # token 20 starts a doc
    assert sets[1][0] is None and len(sets[1][1]) == 2
