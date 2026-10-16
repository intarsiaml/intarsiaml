#!/usr/bin/env python3
"""MLX (0.32.1) bf16 GEMM and SDPA-prefill throughput at the Qwen3-8B prefill shapes, same machine/slot as the intarsia profile.
Run with /Users/josef/.local/share/uv/tools/mlx-lm/bin/python. Reference ceiling for what the steel/NAX GEMM reaches."""
import time, json, sys
import mlx.core as mx
print("mlx", mx.__version__, mx.metal.device_info().get("architecture"), file=sys.stderr)
def bench(f, n=20, warm=5):
    for _ in range(warm): mx.eval(f())
    t = time.perf_counter()
    for _ in range(n): mx.eval(f())
    return (time.perf_counter() - t) / n
rows = []
shapes = {"q/o 4096x4096": (4096, 4096), "k/v 1024x4096": (1024, 4096), "gate/up 12288x4096": (12288, 4096), "down 4096x12288": (4096, 12288), "lm_head 151936x4096": (151936, 4096)}
for M in (256, 1024, 2048, 4096):
    for name, (N, K) in shapes.items():
        x = mx.random.normal((M, K)).astype(mx.bfloat16); w = mx.random.normal((N, K)).astype(mx.bfloat16); mx.eval(x, w)
        dt = bench(lambda: x @ w.T)
        rows.append(dict(kind="gemm", M=M, N=N, K=K, name=name, us=dt * 1e6, tflops=2 * M * N * K / dt / 1e12))
for s in (4096, 8192):
    x = mx.random.normal((s, s)).astype(mx.bfloat16); w = mx.random.normal((s, s)).astype(mx.bfloat16); mx.eval(x, w)
    dt = bench(lambda: x @ w.T, n=10)
    rows.append(dict(kind="gemm", M=s, N=s, K=s, name=f"square {s}", us=dt * 1e6, tflops=2 * s ** 3 / dt / 1e12))
# fp16 / fp32 square for the hardware peak picture
for dt_, nm in ((mx.float16, "fp16"), (mx.float32, "fp32")):
    s = 8192; x = mx.random.normal((s, s)).astype(dt_); w = mx.random.normal((s, s)).astype(dt_); mx.eval(x, w)
    dt = bench(lambda: x @ w.T, n=10)
    rows.append(dict(kind="gemm", M=s, N=s, K=s, name=f"square 8192 {nm}", us=dt * 1e6, tflops=2 * s ** 3 / dt / 1e12))
# SDPA prefill: 32 q heads, 8 kv heads, D=128, causal. (T queries at the end of a cache of length S.)
for T, S in ((1024, 1024), (4096, 4096), (256, 1024), (256, 4096)):
    q = mx.random.normal((1, 32, T, 128)).astype(mx.bfloat16); k = mx.random.normal((1, 8, S, 128)).astype(mx.bfloat16); v = mx.random.normal((1, 8, S, 128)).astype(mx.bfloat16); mx.eval(q, k, v)
    mask = "causal"
    dt = bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask=mask), n=10)
    vis = T * (S - T) + T * (T + 1) / 2   # causal visible (q,k) pairs, queries are the last T of S
    fl = 4 * 128 * 32 * vis
    rows.append(dict(kind="sdpa", T=T, S=S, us=dt * 1e6, tflops=fl / dt / 1e12))
for r in rows: print(json.dumps(r))
