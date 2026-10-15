#!/usr/bin/env python3
"""Prefill tables from INTARSIA_METAL_PROFILE traces (crates/intarsia-backend-metal/src/profile.rs).
usage: analyze_prefill.py <trace-dir> <1k|4k> <chunk> [prompt_tokens]
kernel-<P>-c<chunk>.jsonl(.gz): one command buffer per launch (GPU timestamps). cb-<P>-c<chunk>.jsonl(.gz): stream as normal.
Matmul TFLOPS = 2*M*N*K / kernel time (kernel time includes a few us of per-command-buffer cost; negligible at prefill sizes).
Attention FLOPs = 4*D*H*(causal visible q.k pairs) with D=128, H=32; chunk i of size C starts at pos0 = i*C (36 attn_prefill per chunk)."""
import gzip, json, os, sys, collections
d, tag, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
LAYERS, H, D = 36, 32, 128
def load(p):
    f = gzip.open(p + ".gz", "rt") if os.path.exists(p + ".gz") else open(p)
    return [json.loads(l) for l in f]
k = load(f"{d}/kernel-{tag}-c{chunk}.jsonl")
tab = collections.OrderedDict()
cls = collections.Counter()
attn_idx = 0
tot_flops = 0.0
for r in k:
    dur = r["e"] - r["s"]
    op = r["op"]
    key = (op, r["t"], r["n"], r["k"])
    fl = 0.0
    if op.startswith("matmul") or op.startswith("logits"):
        fl = 2.0 * r["t"] * r["n"] * r["k"]
        c = "matmul"
    elif op.startswith("attn_prefill"):
        ci = attn_idx // LAYERS; attn_idx += 1
        pos0 = ci * chunk; T = r["t"]
        fl = 4.0 * D * H * (T * pos0 + T * (T + 1) / 2)
        key = (op, T, pos0, 0)
        c = "attention"
    elif op in ("embedding_bf16",):
        c = "elementwise/row"
    else:
        c = "elementwise/row"
    e = tab.setdefault(key, [0, 0.0, fl])
    e[0] += 1; e[1] += dur
    cls[c] += dur
total = sum(cls.values())
span = k[-1]["e"] - k[0]["s"]
print(f"### {tag} chunk {chunk}: kernel-mode trace, {len(k)} launches\n")
# aggregate by op/shape for matmul and by op for the rest
agg = collections.OrderedDict()
for (op, t, n, kk), (c, s, fl) in tab.items():
    if op.startswith("attn_prefill"):
        key = (op, "", "")
    elif op.startswith("matmul") or op.startswith("logits"):
        key = (op, f"M={t}", f"{n}x{kk}")
    else:
        key = (op, "", "")
    a = agg.setdefault(key, [0, 0.0, 0.0]); a[0] += c; a[1] += s; a[2] += fl * c
print("| kernel | M | n x k | calls | us/call | total ms | share | TFLOPS |\n|---|---|---|---:|---:|---:|---:|---:|")
for key, (c, s, fl) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    tf = f"{fl / s / 1e12:.2f}" if fl else ""
    print(f"| {key[0]} | {key[1]} | {key[2]} | {c} | {s / c * 1e6:.0f} | {s * 1e3:.1f} | {100 * s / total:.1f}% | {tf} |")
mm = [v for kk, v in agg.items() if kk[0].startswith("matmul") or kk[0].startswith("logits")]
mmf = sum(v[2] for v in mm); mmt = sum(v[1] for v in mm)
print(f"\nSum of kernel time {total * 1e3:.1f} ms; first-start to last-end span {span * 1e3:.1f} ms (kernel mode serialises launches, so the span includes host submit latency).\n")
print("| class | ms | share |\n|---|---:|---:|")
for c, t in cls.most_common():
    print(f"| {c} | {t * 1e3:.1f} | {100 * t / total:.1f}% |")
print(f"\nMatmul: {mmf / 1e12:.2f} TFLOP in {mmt * 1e3:.1f} ms = {mmf / mmt / 1e12:.2f} TFLOPS aggregate.")
at = [v for kk, v in agg.items() if kk[0].startswith("attn_prefill")]
if at: print(f"Attention: {at[0][2] / 1e12:.2f} TFLOP (causal) in {at[0][1] * 1e3:.1f} ms = {at[0][2] / at[0][1] / 1e12:.2f} TFLOPS.")
# per-chunk attention time (shows the quadratic growth)
if attn_idx:
    per = collections.defaultdict(float)
    ai = 0
    for r in k:
        if r["op"].startswith("attn_prefill"):
            per[ai // LAYERS] += r["e"] - r["s"]; ai += 1
    print("\nattention ms per chunk (36 layers): " + ", ".join(f"c{ci}:{v * 1e3:.0f}" for ci, v in sorted(per.items())))
try:
    c = load(f"{d}/cb-{tag}-c{chunk}.jsonl")
    busy = sum(r["e"] - r["s"] for r in c); sp = c[-1]["e"] - c[0]["s"]
    print(f"\ncb mode (stream as normal): {len(c)} command buffers, GPU busy {busy * 1e3:.1f} ms of span {sp * 1e3:.1f} ms ({100 * busy / sp:.1f}% busy).")
except FileNotFoundError:
    pass
