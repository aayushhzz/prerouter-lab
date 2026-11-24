"""`pr report`: one self-contained report.html (inline SVG chart, no external assets)."""
from __future__ import annotations

import html
import json
import os
from typing import Dict, List, Optional

from .sim import POLICIES, POLICY_LABELS

COLORS = {"lru": "var(--c1)", "prefetch": "var(--c2)", "par": "var(--c3)"}


def _load(data_dir: str, name: str) -> Optional[Dict]:
    p = os.path.join(data_dir, name)
    return json.load(open(p)) if os.path.exists(p) else None


def summary_for_analyst(meta, train, sim, ppl) -> Dict:
    s: Dict = {"trace_source": meta.get("source"), "model": meta.get("model"), "k": meta.get("k"),
               "n_experts": meta.get("n_experts"), "n_layers": meta.get("n_layers")}
    if train:
        s["recall_at_k_mean"] = {k: round(v, 3) for k, v in train["mean"].items()}
    if sim:
        s["ssd_mb_per_s"] = round(sim["ssd"]["mb_per_s"], 1)
        s["read_ms_per_expert"] = round(sim["read_ms"], 3)
        s["compute_ms_per_token"] = round(sim["compute_ms"], 2)
        s["sim"] = [{"policy": r["policy"], "ram_gib": round(r["ram_gib"], 2), "tok_s": round(r["tok_s"], 2),
                     "stall_ms": round(r["stall_ms_per_token"], 1), "MB_per_tok": round(r["bytes_per_token"] / 1e6, 1),
                     "wasted_reads": r["wasted_reads"]} for r in sim["rows"]]
        if sim.get("routing_stats"):
            s["routing_stats"] = {k: round(v, 2) for k, v in sim["routing_stats"].items()}
    if ppl:
        s["perplexity"] = {m: round(v["ppl"], 3) for m, v in ppl["modes"].items()}
    return s


def chart(rows: List[Dict], w: int = 720, h: int = 360) -> str:
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 44
    xs = [r["ram_gib"] for r in rows]
    ys = [r["tok_s"] for r in rows]
    x0, x1 = min(xs), max(xs)
    y1 = max(ys) * 1.08
    X = lambda v: pad_l + (v - x0) / (x1 - x0 or 1) * (w - pad_l - pad_r)
    Y = lambda v: h - pad_b - v / y1 * (h - pad_t - pad_b)
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Modeled tokens per second vs RAM budget" class="chart">' % (w, h)]
    for i in range(6):
        v = y1 * i / 5
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" class="grid"/>' % (pad_l, w - pad_r, Y(v), Y(v)))
        out.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%.0f</text>' % (pad_l - 6, Y(v) + 4, v))
    for v in sorted(set(round(x, 2) for x in xs)):
        out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%.2f</text>' % (X(v), h - pad_b + 16, v))
    out.append('<text x="%d" y="%d" class="axis" text-anchor="middle">RAM budget (GiB: expert cache + resident non-expert weights)</text>'
               % ((pad_l + w - pad_r) // 2, h - 6))
    out.append('<text x="14" y="%d" class="axis" text-anchor="middle" transform="rotate(-90 14 %d)">modeled tok/s</text>'
               % (h // 2, h // 2))
    for p in POLICIES:
        pts = sorted((r["ram_gib"], r["tok_s"]) for r in rows if r["policy"] == p)
        path = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", X(a), Y(b)) for i, (a, b) in enumerate(pts))
        out.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.5"/>' % (path, COLORS[p]))
        for a, b in pts:
            out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"><title>%s: %.2f GiB, %.1f tok/s</title></circle>'
                       % (X(a), Y(b), COLORS[p], POLICY_LABELS[p], a, b))
    out.append("</svg>")
    legend = "".join('<span class="key"><i style="background:%s"></i>%s</span>' % (COLORS[p], POLICY_LABELS[p]) for p in POLICIES)
    return '<div class="legend">%s</div>%s' % (legend, "".join(out))


def _table(head: List[str], rows: List[List], num_from: int = 1) -> str:
    th = "".join("<th%s>%s</th>" % (' class="n"' if i >= num_from else "", html.escape(h)) for i, h in enumerate(head))
    body = "".join("<tr>%s</tr>" % "".join("<td%s>%s</td>" % (' class="n"' if i >= num_from else "", c)
                                            for i, c in enumerate(r)) for r in rows)
    return '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (th, body)


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#5f6368;--line:#e3e3e0;--card:#fff;--warn-bg:#fff4e5;--warn-fg:#7a4100;
--c1:#8a8f98;--c2:#2f6fdb;--c3:#d0342c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#ececec;--muted:#a0a4ab;--line:#2c2c30;
--card:#1c1c1f;--warn-bg:#3a2a12;--warn-fg:#ffcf8a;--c1:#a3a8b0;--c2:#6ea0ff;--c3:#ff6b61}}
:root[data-theme="dark"]{--bg:#141416;--fg:#ececec;--muted:#a0a4ab;--line:#2c2c30;--card:#1c1c1f;--warn-bg:#3a2a12;
--warn-fg:#ffcf8a;--c1:#a3a8b0;--c2:#6ea0ff;--c3:#ff6b61}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}
p.sub{color:var(--muted);margin:0 0 20px}.warn{background:var(--warn-bg);color:var(--warn-fg);padding:12px 14px;border-radius:8px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.card span{color:var(--muted);font-size:13px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}.n{text-align:right}
.chart{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
.chart .grid{stroke:var(--line)}.chart .tick{fill:var(--muted);font-size:11px}.chart .axis{fill:var(--muted);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 8px;font-size:13px}.key i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
ul{padding-left:20px}code{font-size:13px}
"""


def build(data_dir: str, out_path: str, analysis: Optional[Dict] = None) -> str:
    meta = _load(data_dir, "trace_meta.json") or {}
    train = _load(data_dir, "train_results.json")
    sim = _load(data_dir, "sim_results.json")
    ppl = _load(data_dir, "ppl_results.json")
    synthetic = meta.get("source") != "mlx"
    parts = ["<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
             "<title>Prerouter Lab Report</title><style>%s</style></head><body><main>" % CSS,
             "<h1>Prerouter Lab: routing prediction for SSD-streamed experts</h1>",
             "<p class=sub>A small-scale replication of Edge0's prerouter (arXiv 2609.18063) on %s.</p>"
             % html.escape(meta.get("model") or "synthetic traces")]
    if synthetic:
        parts.append("<div class=warn><b>SYNTHETIC TRACES.</b> These results come from the Zipf + locality trace generator, "
                     "not a real MoE checkpoint. Reason: %s</div>" % html.escape(str(meta.get("synthetic_reason"))))
    cards = []
    if train:
        m = train["mean"]
        cards.append(("%.3f" % m["prerouter"], "prerouter recall@%d (held-out)" % train["k"]))
        cards.append(("%.3f" % m["prev_token"], "previous-token baseline"))
    if sim:
        cards.append(("%.0f MB/s" % sim["ssd"]["mb_per_s"], "measured SSD read (QD1, no cache)"))
    if ppl:
        cards.append(("%+.1f%%" % ppl["modes"]["pred"]["delta_pct"], "perplexity change, prediction-as-routing"))
    if cards:
        parts.append("<div class=cards>%s</div>" % "".join("<div class=card><b>%s</b><span>%s</span></div>" % c for c in cards))
    if analysis:
        parts.append("<h2>Analyst read (%s)</h2><p><b>%s</b></p><ul>%s</ul><p class=sub>Caveats:</p><ul>%s</ul>" % (
            html.escape(", ".join(analysis.get("_meta", {}).get("models") or [analysis.get("_meta", {}).get("backend", "")])),
            html.escape(analysis["headline"]),
            "".join("<li>%s</li>" % html.escape(f) for f in analysis["findings"]),
            "".join("<li>%s</li>" % html.escape(c) for c in analysis["caveats"])))
    if sim:
        parts.append("<h2>Modeled decode speed vs RAM budget</h2>")
        parts.append("<p class=sub>Replay of %d held-out tokens. Expert read = %.2f ms (%.2f MB at %.0f MB/s measured); compute = %.1f ms/token (%s).</p>"
                     % (sim["rows"][0]["tokens"], sim["read_ms"], meta.get("expert_bytes", 0) / 1e6, sim["ssd"]["mb_per_s"],
                        sim["compute_ms"], html.escape(meta.get("compute_ms_source", ""))))
        parts.append(chart(sim["rows"]))
        st = sim.get("routing_stats")
        if st:
            parts.append("<p class=sub>Why prediction-as-routing also reads fewer bytes: it changes the routing itself. Adjacent tokens share "
                         "%.0f%% of a layer's experts under the prerouter vs %.0f%% under the true router, and each predicted layer touched "
                         "%.1f distinct experts over the replay vs %.1f. Smoother routing means more cache hits, and it is also part of "
                         "the perplexity cost below.</p>"
                         % (100 * st["adjacent_token_overlap_pred"], 100 * st["adjacent_token_overlap_true"],
                            st["unique_experts_per_layer_pred"], st["unique_experts_per_layer_true"]))
        rows = []
        for r in sorted(sim["rows"], key=lambda r: (r["capacity"], POLICIES.index(r["policy"]))):
            rows.append([POLICY_LABELS[r["policy"]], r["capacity"], "%.2f" % r["ram_gib"],
                         "%.1f" % (r["bytes_per_token"] / 1e6), "%.1f" % r["stall_ms_per_token"], "%.1f" % r["tok_s"],
                         r["demand_reads"], r["prefetch_reads"], r["wasted_reads"]])
        parts.append("<h2>Simulation table</h2>")
        parts.append(_table(["policy", "cache (experts)", "RAM GiB", "MB read/token", "stall ms/token", "tok/s",
                             "demand reads", "prefetch reads", "wasted reads"], rows, num_from=1))
        s = sim["ssd"]
        parts.append("<h2>SSD measurement</h2>")
        parts.append(_table(["files read", "bytes", "MB/s", "ms/file mean", "p50", "p95", "cache bypass"],
                            [[s["files_read"], s["bytes_read"], "%.0f" % s["mb_per_s"], "%.2f" % s["ms_per_file_mean"],
                              "%.2f" % s["ms_per_file_p50"], "%.2f" % s["ms_per_file_p95"], s["cache_bypass"]]], num_from=0))
    if train:
        k = train["k"]
        has0 = "next_router_zero_shot" in train["mean"]
        head = ["head (layer L to L+1)", "prerouter", "prev-token", "most-frequent"] + (["next-router 0-shot"] if has0 else []) + ["prerouter (train)"]
        rows = []
        for r in train["per_layer"] + [dict(train["mean"], head="mean", target_layer="")]:
            name = "mean" if r["head"] == "mean" else "%d to %d" % (r["head"], r["target_layer"])
            rows.append([name, "%.3f" % r["prerouter"], "%.3f" % r["prev_token"], "%.3f" % r["most_frequent"]]
                        + (["%.3f" % r["next_router_zero_shot"]] if has0 else []) + ["%.3f" % r["prerouter_train"]])
        parts.append("<h2>Recall@%d of layer L+1's experts at token t+1, held-out split</h2>" % k)
        parts.append("<p class=sub>%d train pairs, %d held-out pairs. Prediction is made from layer L's router input and routing at token t.</p>"
                     % (train["n_train_pairs"], train["n_heldout_pairs"]))
        parts.append(_table(head, rows))
    if ppl:
        parts.append("<h2>Perplexity cost of replacing the router</h2>")
        parts.append("<p class=sub>%d held-out docs, %d predicted tokens. No recovery LoRA (see README).</p>" % (ppl["n_docs"], ppl["n_tokens"]))
        parts.append(_table(["routing", "perplexity", "change vs true"],
                            [[html.escape(v["label"]), "%.3f" % v["ppl"], "%+.1f%%" % v["delta_pct"]] for v in ppl["modes"].values()]))
    elif synthetic:
        parts.append("<h2>Perplexity</h2><p class=sub>Not available: synthetic traces have no language model.</p>")
    parts.append("</main></body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path
