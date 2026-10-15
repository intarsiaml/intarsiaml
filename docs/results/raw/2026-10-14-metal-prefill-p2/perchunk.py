#!/usr/bin/env python3
"""Per-chunk attention time and TFLOPS from the kernel-mode traces. usage: perchunk.py <trace-dir> <1k|4k|16k> [chunk=256]
Prints a markdown table (chunk index, pos0, T, pre ms, new ms, speedup, new TFLOPS) and writes <trace-dir>/perchunk-<tag>.csv.
FLOPs per chunk and layer: 4 * D * H * (T * pos0 + T * (T + 1) / 2), D = 128, H = 32 (causal q.k pairs)."""
import gzip, json, sys

d, tag = sys.argv[1], sys.argv[2]
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 256
LAYERS, H, D = 36, 32, 128


def load(label):
    with gzip.open(f"{d}/kernel-{tag}-{label}-c{chunk}.jsonl.gz", "rt") as f:
        recs = [json.loads(l) for l in f]
    per, rows, i = {}, {}, 0
    for r in recs:
        if r["op"].startswith("attn_prefill"):
            ci = i // LAYERS
            i += 1
            per[ci] = per.get(ci, 0.0) + (r["e"] - r["s"])
            rows[ci] = r["t"]
    return per, rows


pre, rows = load("pre")
new, _ = load("new")
out = ["| chunk | pos0 | T | pre ms | new ms | speedup | new TFLOPS |", "|---:|---:|---:|---:|---:|---:|---:|"]
csv = ["chunk,pos0,T,pre_ms,new_ms,speedup,new_tflops"]
for ci in sorted(new):
    t = rows[ci]
    pos0 = ci * chunk
    fl = 4.0 * D * H * (t * pos0 + t * (t + 1) / 2) * LAYERS
    tf = fl / new[ci] / 1e12
    out.append(f"| {ci} | {pos0} | {t} | {pre[ci]*1e3:.0f} | {new[ci]*1e3:.1f} | {pre[ci]/new[ci]:.1f}x | {tf:.1f} |")
    csv.append(f"{ci},{pos0},{t},{pre[ci]*1e3:.2f},{new[ci]*1e3:.2f},{pre[ci]/new[ci]:.2f},{tf:.2f}")
print("\n".join(out))
open(f"{d}/perchunk-{tag}.csv", "w").write("\n".join(csv) + "\n")
