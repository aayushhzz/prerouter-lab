"""One file per expert on the real SSD, and a cache-bypassing read-bandwidth measurement."""
from __future__ import annotations

import fcntl
import glob
import math
import os
import random
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

F_NOCACHE = 48  # macOS fcntl: do not populate / serve from the unified buffer cache


def _nocache(fd: int) -> None:
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:  # not macOS: fall back to normal (possibly cached) IO
        pass


def expert_path(root: str, layer: int, expert: int) -> str:
    return os.path.join(root, "L%02d_E%03d.bin" % (layer, expert))


def write_experts(root: str, blobs: Iterable[Tuple[int, int, Callable[[], bytes]]], expected: int) -> int:
    """blobs yields (layer, expert, get_bytes). Skips files already at the expected size."""
    os.makedirs(root, exist_ok=True)
    written = 0
    for layer, expert, get in blobs:
        p = expert_path(root, layer, expert)
        if os.path.exists(p) and os.path.getsize(p) == expected:
            continue
        data = get()
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            _nocache(fd)
            os.write(fd, data)
        finally:
            os.close(fd)
        written += 1
    return written


def read_file(path: str, buf: bytearray) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        _nocache(fd)
        n = 0
        view = memoryview(buf)
        while True:
            got = os.readv(fd, [view[n:]])
            if got == 0:
                break
            n += got
        return n
    finally:
        os.close(fd)


def measure_bandwidth(root: str, n_files: int = 96, seed: int = 0) -> Dict:
    """Read n random expert files whole, one at a time (queue depth 1), bypassing the cache."""
    files = sorted(glob.glob(os.path.join(root, "*.bin")))
    if not files:
        raise FileNotFoundError("no expert files under %s" % root)
    sample = random.Random(seed).sample(files, min(n_files, len(files)))
    size = max(os.path.getsize(f) for f in sample)
    buf = bytearray(size)
    times, total = [], 0
    for f in sample:
        t0 = time.perf_counter()
        total += read_file(f, buf)
        times.append(time.perf_counter() - t0)
    secs = sum(times)
    times.sort()
    return {
        "files_read": len(sample),
        "bytes_read": total,
        "seconds": secs,
        "mb_per_s": total / secs / 1e6,
        "ms_per_file_mean": 1000 * secs / len(sample),
        "ms_per_file_p50": 1000 * times[len(times) // 2],
        "ms_per_file_p95": 1000 * times[math.ceil(len(times) * 0.95) - 1],  # nearest-rank
        "cache_bypass": "F_NOCACHE",
    }


def random_blobs(n_layers: int, n_experts: int, nbytes: int, seed: int = 0):
    """Synthetic fallback: incompressible random bytes of the real expert size."""
    for L in range(n_layers):
        for e in range(n_experts):
            yield L, e, (lambda: os.urandom(nbytes))


def checkpoint_blobs(model_dir: str, n_layers: int, n_experts: int):
    """Real int4 expert weights from an mlx-community MoE checkpoint, one blob per expert."""
    import mlx.core as mx
    import numpy as np

    weights: Dict = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        weights.update(mx.load(f))
    parts = [(proj, part) for proj in ("gate_proj", "up_proj", "down_proj") for part in ("weight", "scales", "biases")]
    for L in range(n_layers):
        for e in range(n_experts):
            def get(L=L, e=e):
                arrs = []
                for proj, part in parts:
                    per_expert = "model.layers.%d.mlp.experts.%d.%s.%s" % (L, e, proj, part)  # unstacked layout
                    stacked = "model.layers.%d.mlp.switch_mlp.%s.%s" % (L, proj, part)
                    if per_expert in weights:
                        arrs.append(weights[per_expert])
                    elif stacked in weights:
                        arrs.append(weights[stacked][e])
                return b"".join(np.array(a.view(mx.uint8)).tobytes() for a in arrs)
            yield L, e, get


def expert_nbytes_from_checkpoint(model_dir: str, layer: int = 0) -> Optional[int]:
    total = 0
    for _, _, get in checkpoint_blobs(model_dir, layer + 1, 1):
        total = len(get())
    return total or None
