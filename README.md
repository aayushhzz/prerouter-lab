# prerouter-lab

A small, measurable replication of the core idea in **"The Other Half of the Memory Wall: Serving
35B MoEs from SSD with Trained Routing Prediction"** (Edge0, arXiv
[2609.18063](https://arxiv.org/abs/2609.18063), official code at
[Edge0-AI/edge0](https://github.com/Edge0-AI/edge0); this repo is an independent implementation and
copies no code from it). It runs on a 16 GB Apple Silicon Mac with a real 4-bit MoE
(`mlx-community/OLMoE-1B-7B-0125-Instruct-4bit`, which was verified to load and run).

## What the paper says and why it matters

Mixture-of-experts sparsity cuts the compute per token, not the bytes you have to store. A
35B-class MoE is about 19.5 GB at 4-bit, so on a laptop it either doesn't fit or crowds out everything else.
Edge0 leaves the experts on the SSD and streams them in. A naive version stalls on every layer,
because layer N+1's experts are only known once layer N has finished. Edge0 trains a tiny per-layer
**prerouter** that predicts layer N+1's top-k **one token ahead** from layer N's state, so the
SSD reads overlap compute. The twist is that **the prediction is used as the routing**. The staged
experts are then exactly the routed experts, so no read is wasted and nothing is dropped. The quality
cost moves into training (an unmerged recovery LoRA). The paper reports a 35B MoE at about 20 tok/s
inside 3 GiB of active memory, and +80-84% decode from the prerouter on a 16 GB machine.

`PAPER_NOTES.md` has the mechanism in more detail, a concept-to-code table and the list of
simplifications.

## What this project does

| Command | What it does |
|---|---|
| `pr capture` | Loads the MLX MoE and re-runs its decoder layer by layer over ~3.2k tokens of agent transcripts and code. For every layer it logs the router input (post-attention norm output) and the true top-8. It also measures the model's resident decode speed. If the checkpoint fails to load, it falls back to a **synthetic Zipf + locality** trace and flags this in `trace_meta.json` and in the report. `--synthetic` forces the fallback. |
| `pr train` | Fits 15 per-layer linear prerouters (layer L at token t, predicting layer L+1 at token t+1) with numpy. Each is warm-started from the next layer's router weights. It reports held-out recall@8 against previous-token, most-frequent and next-router zero-shot baselines. |
| `pr sim` | Writes each of the 1,024 experts' real int4 weights to its own file (3.54 MB each, 3.4 GB in all). It measures SSD read bandwidth with cache-bypassing (`F_NOCACHE`) reads, then replays the held-out trace through an expert cache at 7 sizes under 3 policies: **on-demand LRU**, **LRU + predicted prefetch**, and **prediction-as-routing (Edge0)**. |
| `pr ppl` | Held-out perplexity with the true router vs. the prerouter's prediction as the routing (and a variant that keeps the layer's own gate weights). |
| `pr report` | Writes `report.html`: a tok/s vs RAM-budget curve per policy, the recall@k table, the simulation and SSD tables, perplexity, and a short read from an **analyst agent** (Claude, via your local Claude Code login). |

### Where Claude comes in

The research pipeline (capture, train, sim, ppl) is plain numpy and MLX and needs no LLM. The single
model call is the analyst agent in `pr report`. It receives the results JSON and returns a
schema-checked headline, findings and caveats. All model calls go through one small interface in
`prerouter_lab/backend.py`:

- `claude` (default) shells out to `claude -p --output-format json --json-schema ... --tools ""`.
  This uses your **Claude subscription through the locally logged-in Claude Code CLI**.
  `ANTHROPIC_API_KEY` is removed from the child environment, and no key is needed. `--llm-model`
  overrides the CLI's default model.
- `ollama` is an optional local fallback (`--backend ollama --llm-model qwen3:4b-instruct`).
- `fake` is used by the tests. `none` skips the analyst section.

The answer is cached in `data/analysis.json`, keyed by a hash of the results, so re-running
`pr report` on unchanged results makes no new call (`--refresh` forces a new one).

> This setup is for your personal, local use. Sharing or deploying it to other people requires
> switching to API-key authentication, per Anthropic's Agent SDK / Claude Code terms.

## Setup

```bash
cd prerouter-lab
python3 -m venv .venv                      # Python 3.9 works (tested on 3.9.6)
.venv/bin/pip install -e '.[test]'         # numpy, mlx-lm (pulls mlx), pytest
```

The first `pr capture` downloads the 3.9 GB checkpoint into the Hugging Face cache.

## Run

```bash
source .venv/bin/activate
pr capture            # ~20-30 s once the model is cached
pr train              # ~25 s (numpy, 15 heads)
pr sim                # writes data/experts (3.4 GB, once), measures SSD, replays 3 policies x 7 cache sizes
pr ppl                # ~2 min: 3 routing modes over the held-out docs
pr report             # one Claude call; writes report.html
python -m pytest -q   # offline tests (fake backend, synthetic traces)
```

Use `--data DIR` before the subcommand to work in another directory. Run `rm -rf data/experts` to
free the 3.4 GB of expert files.

## Demo output (real run, M2 Pro 16 GB, OLMoE-1B-7B 4-bit)

The capture used 3,200 tokens across 9 docs (4 agent transcripts interleaved with 5 CPython stdlib
files). The split is 2,304 train tokens and 896 held-out tokens. The held-out set is the last
agent transcript plus `bisect.py` and `json/decoder.py`.

**Prerouter recall@8 on the held-out split (mean over 15 heads, 893 pairs):**

| predictor | recall@8 |
|---|---|
| **linear prerouter** | **0.647** (train 0.874) |
| most-frequent experts | 0.491 |
| previous token's experts | 0.477 |
| next layer's router applied zero-shot | 0.462 |

**SSD:** 96 expert files, 3,979 MB/s at QD1 with `F_NOCACHE`, 0.89 ms per 3.54 MB expert.
**Compute:** measured resident decode was 39.4 ms/token in this run. Across runs on this machine,
which was swapping, it ranged from 16 to 39 ms/token. That number sets the ceiling of every curve,
so tok/s values from different runs are not comparable.

**Simulation (`pr sim`, 896 held-out tokens; 4 of 7 cache sizes shown):**

```
policy                          cache RAM GiB   MB/token  stall ms   tok/s  wasted
On-demand LRU                     128    0.67      302.1      75.9     8.7       0
LRU + predicted prefetch          128    0.67      539.2      97.5     7.3   21873
Prediction-as-routing (Edge0)     128    0.67      447.1      75.7     8.7       0
On-demand LRU                     256    1.09      169.6      42.6    12.2       0
LRU + predicted prefetch          256    1.09      229.0      33.6    13.7   12912
Prediction-as-routing (Edge0)     256    1.09      122.3       7.5    21.3       0
On-demand LRU                     512    1.94       68.0      17.1    17.7       0
LRU + predicted prefetch          512    1.94       85.3      13.9    18.7    4148
Prediction-as-routing (Edge0)     512    1.94       31.6       1.3    24.6       0
On-demand LRU                    1024    3.63        4.0       1.0    24.7       0
LRU + predicted prefetch         1024    3.63        4.0       0.7    24.9     127
Prediction-as-routing (Edge0)    1024    3.63        3.7       0.2    25.2       0
distinct experts per predicted layer: true 62.9, predicted 57.2; adjacent-token overlap: true 0.46, predicted 0.59
```

**Perplexity (`pr ppl`, 3 held-out docs, 893 predicted tokens, no recovery LoRA):**

| routing | perplexity | change |
|---|---|---|
| true router | 8.55 | |
| prediction-as-routing (prerouter picks experts **and** weights) | 153.5 | +1695% |
| prerouter picks experts, layer's own gate weights them | 19.28 | +125% |

**Analyst agent (`pr report`, one `claude -p` call, ~18 s, the CLI's default model):**

> Using the prerouter's prediction as the routing (PAR) nearly doubles simulated throughput in the
> 0.9–1.5 GiB RAM range, but it raises perplexity from 8.55 to 19.28 even with gate weights, so the
> speedup is paid for in model quality.

(The agent also flagged as an "anomaly" that at 0.67 GiB, PAR reads more bytes than LRU but has the
same stall time. That is expected: most of PAR's reads there are prefetches that overlap the 39 ms
of compute.)

### How to read it

- **Prediction-as-routing never wastes a read.** In every row its wasted reads are 0, and
  `tests/test_sim.py` asserts this on a replayed trace. Predicted prefetch with true routing wastes
  thousands of reads and is *slower* than plain LRU when the cache is tight. This matches the paper's
  point that staging without a set that is correct by definition either drops experts or churns.
- **Hiding latency alone buys little here.** OLMoE's compute is 16-39 ms/token, while a fully
  cold token needs about 128 x 0.9 ms of reads. Overlap can hide at most one token of compute, so
  LRU + prefetch gains only 1-2 tok/s. The paper notes the same thing: the gains shrink on fast
  storage and small models.
- **Most of prediction-as-routing's speed comes from changing the routing.** The predicted routing
  is smoother: adjacent tokens share 59% of their experts vs 46% under the true router. It also
  touches fewer distinct experts, so it gets more cache hits and reads fewer bytes. That same change
  is why perplexity rises.
- **The quality bill is real without recovery.** With weights taken from the prerouter's own
  softmax, perplexity goes from 8.6 to 153. Keeping the layer's own gate weights cuts that to 19.3.
  Edge0 pays this bill with a recovery LoRA trained on the student path, which is not implemented
  here.

## Limits (read before quoting numbers)

- **Throughput is modeled, not run.** No streaming executor exists here. Tok/s comes from
  `compute_ms + stall` with one measured SSD number (QD1, whole-file reads) and one measured compute
  number. The model ignores queue depth, read coalescing, page-cache effects, unified-memory
  contention and graph-assembly time. Prefetch is allowed to overlap exactly one token of compute.
- **Small scale.** The model is 1B active / 7B total, with ~3.2k captured tokens and 896 held-out.
  The prerouter overfits (train 0.87 vs held-out 0.65 recall). The `--steps 600 --l2 1e-2` defaults
  were picked from three configurations judged on held-out recall, so the held-out number is mildly
  optimistic.
- **The sim replays features captured under true routing.** Under prediction-as-routing, the hidden
  states of later layers would drift. `pr ppl` does model that drift, because its forward pass feeds
  each head the student path's own states.
- **No recovery LoRA and no SFT or on-policy phases.** The perplexity numbers are the raw cost of
  routing replacement, an upper bound on what Edge0 pays after recovery.
- **The model fits in RAM.** "Does not fit" is simulated by the RAM-budget sweep, not experienced.
- **The synthetic fallback** (`pr capture --synthetic`, or automatic when the checkpoint fails) gives
  a planted, easier-to-predict trace. The report shows a warning banner, and `pr ppl` is skipped for it.
  The demo above used the **real** checkpoint.

## Layout

```
prerouter_lab/
  cli.py         pr capture | train | sim | ppl | report
  mlx_model.py   layer-by-layer OLMoE forward with logging and replaceable routing
  capture.py     corpus (agent transcripts + CPython stdlib code) and trace capture, synthetic fallback
  trace.py       trace container, doc-boundary split, Zipf + locality generator
  prerouter.py   features, linear head training, recall@k and baselines, predicted sets
  ssd.py         one file per expert, F_NOCACHE bandwidth measurement
  sim.py         expert cache and the three policies
  ppl.py         held-out perplexity under three routing modes
  report.py      report.html (inline SVG, light and dark)
  backend.py     Claude CLI / Ollama / fake backends for the analyst agent
tests/           offline tests (no Claude or Ollama calls, no model download)
paper/edge0.pdf  the paper
```
