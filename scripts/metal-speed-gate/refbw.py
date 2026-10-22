#!/Users/josef/.local/share/uv/tools/mlx-lm/bin/python
"""Fixed Metal reference workload for the clock-normalised speed gate (docs/benchmarking.md "Metal speed gate").

Streams four distinct 256 MiB bf16 matrices (1 GiB, larger than the SoC cache) through a matrix-vector product, the same
access pattern as a decode GEMV, and prints one JSON line {"gbs": <median GB/s>, "reps": n}. It does not touch intarsia, so
its value moves with the machine state (GPU clock, DRAM clock, CPU contention) and not with the code under test.
usage: refbw.py [--reps 7] [--passes 20]
"""
import argparse, json, statistics as st, time
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--passes", type=int, default=20)  # sweeps of the 4 matrices per timed rep
    a = ap.parse_args()
    rows, cols, n = 8192, 16384, 4
    ws = [mx.random.normal((rows, cols)).astype(mx.bfloat16) for _ in range(n)]
    x = mx.random.normal((cols,)).astype(mx.bfloat16)
    mx.eval(ws, x)
    nbytes = rows * cols * 2

    def sweep():
        ys = [w @ x for _ in range(a.passes) for w in ws]
        mx.eval(ys)

    sweep()  # warmup
    gbs = []
    for _ in range(a.reps):
        t = time.perf_counter()
        sweep()
        gbs.append(nbytes * n * a.passes / (time.perf_counter() - t) / 1e9)
    print(json.dumps({"gbs": round(st.median(gbs), 1), "reps": a.reps, "min": round(min(gbs), 1), "max": round(max(gbs), 1)}))


if __name__ == "__main__":
    main()
