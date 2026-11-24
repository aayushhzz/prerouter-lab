"""pr: capture -> train -> sim -> ppl -> report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from . import DEFAULT_MODEL


def _dump(obj, data_dir: str, name: str) -> str:
    p = os.path.join(data_dir, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p


def cmd_capture(a):
    from .capture import capture

    capture(a.model, a.data, synthetic=a.synthetic, max_tokens=a.max_tokens)


def cmd_train(a):
    from . import prerouter
    from .trace import Trace

    trace = Trace.load(a.data)
    out = prerouter.train_all(trace, steps=a.steps, lr=a.lr, l2=a.l2)
    prerouter.save(out, a.data)
    m = out["results"]["mean"]
    print("MEAN recall@%d  prerouter %.3f | prev-token %.3f | most-frequent %.3f  (%.1fs, source=%s)"
          % (trace.k, m["prerouter"], m["prev_token"], m["most_frequent"], out["results"]["train_seconds"], trace.meta["source"]))


def cmd_sim(a):
    from . import prerouter, sim, ssd
    from .trace import Trace

    trace = Trace.load(a.data)
    meta = trace.meta
    W, b = prerouter.load(a.data)
    root = a.experts_dir or os.path.join(a.data, "experts")
    nbytes = meta["expert_bytes"]
    if meta["source"] == "mlx":
        blobs = ssd.checkpoint_blobs(meta["model_dir"], trace.n_layers, trace.n_experts)
    else:
        blobs = ssd.random_blobs(trace.n_layers, trace.n_experts, nbytes)
    n = ssd.write_experts(root, blobs, nbytes)
    print("expert files: %d written, %d total under %s (%.2f MB each)"
          % (n, trace.n_layers * trace.n_experts, root, nbytes / 1e6))
    bw = ssd.measure_bandwidth(root, n_files=a.n_read)
    print("SSD read: %d files, %.0f MB/s, %.2f ms/file mean (p50 %.2f, p95 %.2f), F_NOCACHE"
          % (bw["files_read"], bw["mb_per_s"], bw["ms_per_file_mean"], bw["ms_per_file_p50"], bw["ms_per_file_p95"]))
    if bw["mb_per_s"] > 20000:
        print("WARNING: >20 GB/s suggests reads were served from memory, not the SSD")
    read_ms = nbytes / (bw["mb_per_s"] * 1e6) * 1000
    lo, hi = trace.heldout_start(), trace.T
    true_sets = [[frozenset(int(e) for e in trace.topk[L, t]) for L in range(trace.n_layers)] for t in range(lo, hi)]
    pred_sets = prerouter.predicted_sets(trace, W, b, lo, hi)
    caps = sim.capacities(trace.n_layers, trace.k, trace.n_experts)
    rows = sim.sweep(true_sets, pred_sets, caps, read_ms, meta["compute_ms_per_token"], nbytes, meta["nonexpert_bytes"])
    print("%-30s %6s %7s %10s %9s %7s %7s" % ("policy", "cache", "RAM GiB", "MB/token", "stall ms", "tok/s", "wasted"))
    for r in rows:
        print("%-30s %6d %7.2f %10.1f %9.1f %7.1f %7d" % (sim.POLICY_LABELS[r["policy"]], r["capacity"], r["ram_gib"],
                                                       r["bytes_per_token"] / 1e6, r["stall_ms_per_token"], r["tok_s"], r["wasted_reads"]))
    stats = sim.routing_stats(true_sets, pred_sets)
    print("distinct experts per predicted layer: true %.1f, predicted %.1f; adjacent-token overlap: true %.2f, predicted %.2f"
          % (stats["unique_experts_per_layer_true"], stats["unique_experts_per_layer_pred"], stats["adjacent_token_overlap_true"], stats["adjacent_token_overlap_pred"]))
    _dump({"ssd": bw, "read_ms": read_ms, "compute_ms": meta["compute_ms_per_token"], "rows": rows,
           "routing_stats": stats, "source": meta["source"]}, a.data, "sim_results.json")


def cmd_ppl(a):
    from . import ppl, prerouter
    from .trace import Trace

    trace = Trace.load(a.data)
    if trace.meta["source"] != "mlx":
        print("pr ppl: trace is SYNTHETIC (%s); perplexity needs a real checkpoint. Skipping." % trace.meta.get("synthetic_reason"))
        return
    W, b = prerouter.load(a.data)
    out = ppl.perplexity(trace, W, b)
    _dump(out, a.data, "ppl_results.json")


def cmd_report(a):
    from . import report
    from .backend import ANALYST_SCHEMA, analyst_prompt, get_backend

    def load(n):
        p = os.path.join(a.data, n)
        return json.load(open(p)) if os.path.exists(p) else None

    summary = report.summary_for_analyst(load("trace_meta.json") or {}, load("train_results.json"),
                                         load("sim_results.json"), load("ppl_results.json"))
    key = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    cached = load("analysis.json")
    analysis = None
    backend = get_backend(a.backend, a.llm_model)
    if backend is not None:
        if cached and cached.get("key") == key and cached.get("backend") == a.backend and not a.refresh:
            analysis = cached["analysis"]
            print("analyst: reusing cached %s answer for unchanged results" % a.backend)
        else:
            try:
                analysis = backend.complete_json(analyst_prompt(summary), ANALYST_SCHEMA)
                _dump({"key": key, "backend": a.backend, "analysis": analysis}, a.data, "analysis.json")
                print("analyst (%s): %s" % (", ".join(analysis["_meta"].get("models") or [a.backend]), analysis["headline"]))
            except Exception as e:  # report still renders without the analyst section
                print("analyst backend %s failed: %s" % (a.backend, e), file=sys.stderr)
    out = report.build(a.data, a.out, analysis)
    print("wrote %s" % os.path.abspath(out))


def main(argv=None):
    p = argparse.ArgumentParser(prog="pr", description=__doc__)
    p.add_argument("--data", default="data", help="working directory for traces and results")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("capture", help="log per-layer router inputs and true top-k from an MLX MoE")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--synthetic", action="store_true", help="skip the checkpoint, use Zipf+locality traces")
    s.add_argument("--max-tokens", type=int, default=3200)
    s.set_defaults(fn=cmd_capture)
    s = sub.add_parser("train", help="fit per-layer linear prerouters, report recall@k vs baselines")
    s.add_argument("--steps", type=int, default=600)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--l2", type=float, default=1e-2)
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("sim", help="write experts to SSD, measure bandwidth, replay cache policies")
    s.add_argument("--experts-dir", default=None)
    s.add_argument("--n-read", type=int, default=96, help="expert files to read for the bandwidth measurement")
    s.set_defaults(fn=cmd_sim)
    s = sub.add_parser("ppl", help="held-out perplexity: true router vs predicted routing")
    s.set_defaults(fn=cmd_ppl)
    s = sub.add_parser("report", help="write report.html (optionally with an analyst agent's read)")
    s.add_argument("--out", default="report.html")
    s.add_argument("--backend", default="claude", choices=["claude", "ollama", "fake", "none"])
    s.add_argument("--llm-model", default=None, help="model override for the analyst backend")
    s.add_argument("--refresh", action="store_true", help="ask the analyst again even if results are unchanged")
    s.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
