#!/usr/bin/env python3
"""MLX 0.32.1 bf16 GEMM at the Qwen3-8B prefill shapes, two ways, same shapes and M as the P1 spike bench.

  single: one `mx.eval(x @ w.T)` per iteration, 20 iterations after 5 warm-ups (the phase C method, includes the
          per-eval launch/sync round trip; this is what docs/research/metal-prefill-plan.md section 2.5 quotes)
  chain : `mx.eval` of 8 independent `x @ w_i.T` (8 distinct weights) in one graph, 5 evals after 2 warm-ups, time per
          GEMM = total / 8 (back-to-back GPU work, comparable to the spike's chain of dispatches)

Run with /Users/josef/.local/share/uv/tools/mlx-lm/bin/python inside a fleetq slot. One JSON line per point.
"""
import json
import sys
import time

import mlx.core as mx

print("mlx", mx.__version__, mx.metal.device_info().get("architecture"), file=sys.stderr)

SHAPES = {
    "qo": (4096, 4096),
    "kv": (1024, 4096),
    "gu": (12288, 4096),
    "down": (4096, 12288),
}
NW = 8


def timeit(f, n, warm):
    for _ in range(warm):
        f()
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


for M in (256, 1024, 2048):
    for name, (N, K) in SHAPES.items():
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        ws = [mx.random.normal((N, K)).astype(mx.bfloat16) for _ in range(NW)]
        mx.eval(x, *ws)
        flop = 2.0 * M * N * K
        dt1 = timeit(lambda: mx.eval(x @ ws[0].T), 20, 5)
        dtc = timeit(lambda: mx.eval(*[x @ w.T for w in ws]), 5, 2) / NW
        print(
            json.dumps(
                dict(
                    shape=name, M=M, N=N, K=K,
                    single_us=dt1 * 1e6, single_tflops=flop / dt1 / 1e12,
                    chain_us=dtc * 1e6, chain_tflops=flop / dtc / 1e12,
                )
            ),
            flush=True,
        )
