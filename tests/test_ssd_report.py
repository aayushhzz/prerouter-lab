import json
import os

from prerouter_lab import prerouter, report, ssd
from prerouter_lab.backend import FakeBackend
from prerouter_lab.cli import main
from prerouter_lab.trace import synthetic_trace


def test_write_and_measure_expert_files(tmp_path):
    n = ssd.write_experts(str(tmp_path), ssd.random_blobs(2, 4, 4096), 4096)
    assert n == 8 and len(os.listdir(tmp_path)) == 8
    assert ssd.write_experts(str(tmp_path), ssd.random_blobs(2, 4, 4096), 4096) == 0  # idempotent
    bw = ssd.measure_bandwidth(str(tmp_path), n_files=8)
    assert bw["bytes_read"] == 8 * 4096 and bw["mb_per_s"] > 0


def test_pipeline_on_synthetic_traces_flags_synthetic_in_report(tmp_path, monkeypatch):
    d = str(tmp_path / "data")
    tr = synthetic_trace(n_layers=4, n_experts=16, k=2, d=16, n_tokens=512, doc_len=128, expert_bytes=8192, reason="test")
    tr.save(d)
    main(["--data", d, "train", "--steps", "20"])
    main(["--data", d, "sim", "--n-read", "8"])
    main(["--data", d, "ppl"])  # skips politely on synthetic traces
    out = str(tmp_path / "report.html")
    main(["--data", d, "report", "--backend", "fake", "--out", out])
    html = open(out).read()
    assert "SYNTHETIC TRACES" in html and "<svg" in html and "Recall@2" in html
    assert "Fake analyst" in html
    rows = json.load(open(os.path.join(d, "sim_results.json")))["rows"]
    assert {r["policy"] for r in rows} == {"lru", "prefetch", "par"}
    assert len({r["capacity"] for r in rows}) >= 4
    assert all(r["wasted_reads"] == 0 for r in rows if r["policy"] == "par")
