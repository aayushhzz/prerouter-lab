"""`pr ppl`: held-out perplexity with the true router vs the prerouter's prediction as routing."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .trace import Trace

MODES = {
    "true": "True router (reference)",
    "pred": "Prediction-as-routing (prerouter picks experts and weights)",
    "pred_gatew": "Prerouter picks experts, layer's own gate weights them",
}


def heldout_docs(trace: Trace):
    split = trace.heldout_start()
    for d in np.unique(trace.doc[split:]):
        yield trace.tokens[trace.doc == d].tolist()


def perplexity(trace: Trace, W: np.ndarray, b: np.ndarray, log=print) -> Dict:
    if trace.meta.get("source") != "mlx":
        raise RuntimeError("pr ppl needs a real MLX capture (trace source is %r)" % trace.meta.get("source"))
    import mlx.core as mx

    from . import mlx_model

    model, _ = mlx_model.load(trace.meta["model"])
    docs = list(heldout_docs(trace))
    out = {"n_docs": len(docs), "n_tokens": int(sum(len(d) - 1 for d in docs)), "modes": {}}
    for mode, label in MODES.items():
        total, n = 0.0, 0
        for ids in docs:
            logits, _ = mlx_model.forward(model, ids, mode=mode, W=W, b=b)
            v = mlx_model.nll(logits, ids)
            mx.eval(v)
            total += float(v)
            n += len(ids) - 1
        ppl = math.exp(total / n)
        out["modes"][mode] = {"label": label, "nll": total / n, "ppl": ppl}
        log("%-12s ppl %8.3f   (%s)" % (mode, ppl, label))
    ref = out["modes"]["true"]["ppl"]
    for m in out["modes"].values():
        m["delta_pct"] = 100.0 * (m["ppl"] / ref - 1)
    return out
