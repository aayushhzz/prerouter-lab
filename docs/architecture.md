# prerouter-lab Architecture

This document summarizes the implementation architecture of `prerouter-lab`. For paper theory and a concept-to-code table, see [`PAPER_NOTES.md`](../PAPER_NOTES.md).

## 1. Trace Capture ([`capture.py`](../prerouter_lab/capture.py), [`mlx_model.py`](../prerouter_lab/mlx_model.py), [`trace.py`](../prerouter_lab/trace.py))

- Runs `OLMoE-1B-7B-0125-Instruct-4bit` layer-by-layer over a mixed corpus of agent transcripts ([`prerouter_lab/corpus/`](../prerouter_lab/corpus/)) and CPython stdlib modules.
- Logs post-attention normalized hidden states `feats[L, t]`, true top-8 expert indices `topk[L, t]`, document boundaries `doc[t]`, and layer gate weights `gate_w[L]` into `data/trace.npz` and `data/trace_meta.json`.
- Splits train vs. held-out strictly on document boundaries to avoid cross-document leakage.
- Automatically falls back to a deterministic Zipf + locality synthetic trace when `--synthetic` is passed or the checkpoint cannot load.

## 2. Linear Prerouter Training ([`prerouter.py`](../prerouter_lab/prerouter.py))

For each layer `L` in `0 .. n_layers - 2`, a linear head predicts layer `L+1`'s top-`k` experts at token `t+1`:

```text
x_L(t) = [ feats_L(t), multi_hot(topk_L(t)), multi_hot(topk_L(t-1)) ]   # dim D + 2E
logits_{L+1}(t+1) = x_L(t) @ W_L + b_L
```

- `W_L[:D]` is **warm-started** from layer `L+1`'s router weight matrix `gate_w[L + 1].T`.
- Trained in pure NumPy with full-batch Adam against uniform softmax cross-entropy over the true top-`k` set, with L2 regularization pulling `W_L` toward the warm start `W0`.
- Evaluates held-out `recall@k` against three baselines: previous token, most-frequent experts, and next-layer router zero-shot.

## 3. SSD Benchmark & Cache Simulation ([`ssd.py`](../prerouter_lab/ssd.py), [`sim.py`](../prerouter_lab/sim.py))

- Materializes all `n_layers * n_experts` (1,024) int4 expert weight blobs as individual files under `data/experts/`.
- Measures uncached sequential/random file read latency on macOS using `fcntl.F_NOCACHE` (`measure_bandwidth`).
- Replays the held-out trace across 7 cache capacities under three policies:
  1. **On-demand LRU**: fetches missed true experts synchronously.
  2. **LRU + predicted prefetch**: prefetches predicted experts overlapped with one token of compute, but still routes with the true router (incurring wasted reads on mispredictions).
  3. **Prediction-as-routing (Edge0)**: routes directly using the prerouter's predicted set, guaranteeing zero wasted reads.

## 4. Perplexity & Reporting ([`ppl.py`](../prerouter_lab/ppl.py), [`report.py`](../prerouter_lab/report.py), [`backend.py`](../prerouter_lab/backend.py))

- `ppl.py` runs real forward passes over held-out documents in three modes (true router, pure PAR, and PAR experts with original gate weights), capturing student-path hidden-state drift.
- `report.py` renders a self-contained `report.html` with inline SVG plots and optional schema-validated commentary from the analyst backend (`claude`, `ollama`, `fake`, or `none`).
