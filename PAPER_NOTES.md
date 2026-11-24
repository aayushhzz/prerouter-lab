# Paper notes: Edge0, "The Other Half of the Memory Wall" (arXiv 2609.18063)

## The mechanism, in my words

A mixture-of-experts model only runs K of its E experts per layer per token, so it is cheap to
*compute* but not cheap to *store*: every expert still has to live somewhere. Edge0 leaves the
quantized expert weights on the SSD and pulls in only the ones a token needs.

The obvious version of that stalls. Which experts layer N+1 needs depends on layer N's output,
so a naive streaming engine finds out what to read only when it is already too late, and waits for
the disk once per layer per token.

Edge0 adds a small trained **prerouter** head per layer. The head owned by layer N looks at layer
N's router input (post-attention norm output) at token t, plus which experts layer N used at t and
at t-1, and predicts layer N+1's top-K **for the next token, t+1**. Leading by a whole token (not
by one layer as in Pre-gated MoE) means all heads can run in one batch per step and the reads get a
full decode step to hide behind.

The twist: the prediction is not a hint for a cache. **It is the routing.** At decode, layer N+1
routes to exactly the experts the head predicted, using the same softmax-top-k math as its real
router. So the staged set and the routed set are identical by construction. Nothing staged is
wasted, nothing routed is missing, and nothing has to be recomputed or dropped. The price moves from
runtime (misses, fallback loads) to quality (the model now runs with slightly different routing).
Edge0 then pays that quality back in training: the heads are distilled from the next layer's true
router first, and an **unmerged recovery LoRA** (trained on the "student path", with prerouter
routing on) recovers both the int4 and the routing-replacement loss. It stays unmerged because LoRA
deltas fall below the 4-bit quantization step and would vanish on re-quantization.

The paper's own measurements say the win is "conservation, not prediction quality": the same bytes
get read, but earlier and off the critical path, in fewer and larger loads. It also comes with a
cost: staged experts are unreclaimable resident memory taken from the page cache.

## Concept to code map

| Paper concept | Where it lives here | Notes |
|---|---|---|
| Router input of layer N (post-attention norm output) | `prerouter_lab/mlx_model.py: forward()` (`xn`) | Same modules as `mlx_lm.models.olmoe`, re-run layer by layer |
| Softmax-top-k routing (Appendix A) | `mlx_model.py: forward()`, `_topk()` | OLMoE uses raw softmax probabilities (`norm_topk_prob=False`) |
| Logging (layer-N state, layer-N+1 true top-k) | `prerouter_lab/capture.py: capture_real()` | Teacher-forced full-sequence pass; per-position routing is identical to decode |
| Prerouter head input `[h_N(t), onehot(topk_N(t)), onehot(topk_N(t-1))]` | `prerouter.py: layer_features()` | dim D + 2E = 2048 + 128 for OLMoE |
| One-token lead: head N at t predicts layer N+1 at t+1 | `prerouter.py: pair_rows()`, `predicted_sets()`; `mlx_model.py` (`plog[:-1]` shift) | Pairs never cross a document boundary |
| Linear residual path warm-started from the next layer's router | `prerouter.py: warm_start()`, `train_head()` | L2 pulls weights toward the warm start, so the head learns a correction |
| Phase 1: distill heads to imitate next router | `prerouter.py: train_head()` | Softmax cross-entropy vs. the true top-k set (not full router logits) |
| Recall of predicted vs true expert sets | `prerouter.py: recall_at_k()`, `train_all()` | With previous-token, most-frequent and next-router zero-shot baselines |
| Experts as SSD-resident files | `ssd.py: write_experts()`, `checkpoint_blobs()` | One file per (layer, expert) holding its real int4 weights, scales and biases |
| Storage tier speed | `ssd.py: measure_bandwidth()` | QD1 whole-file reads with macOS `F_NOCACHE` (bypasses the page cache) |
| Bounded expert pool / LRU | `sim.py: ExpertCache` | Global LRU over (layer, expert); staged entries are pinned until consumed |
| On-demand streaming baseline | `sim.py: simulate(policy="lru")` | |
| Prefetch with a predictor (pre-gating style) | `sim.py: simulate(policy="prefetch")` | True routing; mispredictions = wasted reads + demand misses |
| Prediction-as-routing | `sim.py: simulate(policy="par")`; `mlx_model.py: forward(mode="pred")` | Zero wasted reads is asserted in `tests/test_sim.py` |
| Reads overlapped with a decode step | `sim.py` (`exposed_next = max(0, n*read_ms - compute_ms)`) | |
| Quality cost of routing replacement | `ppl.py: perplexity()` | True router vs `pred` vs `pred_gatew` |
| Recovery LoRA | not implemented | See below |

## What I simplified, and why

- **Model and scale.** OLMoE-1B-7B (16 layers, 64 experts, top-8, ~3.5 MB per int4 expert, 3.9 GB
  checkpoint) instead of a 35B model. It fits in 16 GB, so "does not fit in RAM" is *simulated* by
  the RAM budget sweep, not experienced.
- **Linear head instead of fc1, GELU, fc2 plus a linear residual.** The brief asked for a linear head, and with
  about 2.3k training tokens an MLP would overfit. The linear part is the paper's residual path,
  with the same warm start.
- **Training signal.** Cross-entropy against the true top-k set, not distillation of full router
  logits. No Phase 2 (SFT on the student path) and no Phase 3 (on-policy distillation): those need
  real training runs over millions of rows.
- **No recovery LoRA.** It is what makes prediction-as-routing usable in the paper, but training it
  needs a full backward pass through the student path over a large corpus. `pr ppl` therefore shows
  the *raw* routing-replacement cost, before any recovery. It is an upper bound on what Edge0 pays.
- **Timing is modeled, not executed.** `pr sim` replays held-out traces through a cache model with
  one measured number (QD1 SSD bandwidth, which gives ms per expert read) and one measured compute
  time (resident mlx-lm decode of the same checkpoint). There is no real streaming executor, no page
  cache, no read coalescing or queue depth above 1, and no graph-assembly costs (the paper's §6
  floor). Prefetch is allowed to overlap exactly one token of compute.
- **Feature drift in the sim.** The sim replays features captured under true routing. When routing
  is replaced, later layers' hidden states (and so the predictions) would drift. `pr ppl` does model
  this: its forward pass feeds each head the student path's own states and routing.
- **Layer 0 and document starts** route on demand by their own router, like Edge0's unstaged early
  layers.
- **Teacher-forced capture.** Routing is logged from full-sequence forward passes over the corpus
  rather than from token-by-token generation. For a causal model the per-position routing is the same.
