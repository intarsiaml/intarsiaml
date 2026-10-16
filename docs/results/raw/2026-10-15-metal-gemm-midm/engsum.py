#!/usr/bin/env python3
"""Per-call GPU us of each NAX GEMM class (kernel-mode traces eng/k-<size>-<tile>-r<n>.jsonl), median over layers and rounds.
python3 engsum.py [dir]"""
import glob, json, re, statistics, sys, collections
d = sys.argv[1] if len(sys.argv) > 1 else "eng"
cls = {(4096, 4096): "qo", (2048, 4096): "kv", (24576, 4096): "gu", (4096, 12288): "down"}
data = collections.defaultdict(list)   # (size, class, tile) -> [(us, run id)]
attn = {}                              # run id -> total attention GPU us (tile independent: the clock reference)
tm = {}
for f in sorted(glob.glob(f"{d}/k-*-r*.jsonl")):
    m = re.fullmatch(rf"{d}/k-(\d+)-(\w+)-r(\d+)\.jsonl", f)
    if not m or "macmon" in f:
        continue
    size, tile = int(m.group(1)), m.group(2)
    run = f
    attn[run] = 0.0
    for l in open(f):
        r = json.loads(l)
        if r["op"].startswith("attn_prefill"):
            attn[run] += (r["e"] - r["s"]) * 1e6
        if r["op"] in ("matmul_bf16_nax", "matmul_pair_bf16_nax") and r["t"] > 1:
            c = cls.get((r["n"], r["k"]))
            if c:
                data[(size, c, tile)].append(((r["e"] - r["s"]) * 1e6, run))
                tm[size] = r["t"]
tiles = sorted({k[2] for k in data})
ref = {}
for (size, c, tile), v in data.items():
    ref.setdefault(size, set()).update(run for _, run in v)
ref = {size: statistics.median(attn[r] for r in runs) for size, runs in ref.items()}
def norm(size, v):  # scale each call by the run's attention time against the size's median attention time
    return statistics.median(us * ref[size] / attn[run] for us, run in v)
print("size(M) class | normalised us per call (raw median in brackets) per forced tile | best (normalised)")
print("tiles: " + " ".join(tiles))
for size in sorted({k[0] for k in data}):
    for c in ("qo", "kv", "gu", "down"):
        row = {t: norm(size, data[(size, c, t)]) for t in tiles if data.get((size, c, t))}
        raw = {t: statistics.median(us for us, _ in data[(size, c, t)]) for t in row}
        if not row:
            continue
        best = min(row, key=row.get)
        print(f"{size}(M={tm[size]}) {c:4} | " + " | ".join(f"{row[t]:7.0f} ({raw[t]:6.0f})" for t in tiles if t in row) + f" | {best}")

# total NAX GEMM time per run (all classes), normalised by the run's attention time
tot = collections.defaultdict(list)
for (size, c, tile), v in data.items():
    pass
runs = collections.defaultdict(lambda: [0.0, 0.0])
for (size, c, tile), v in data.items():
    for us, run in v:
        runs[run][0] += us
        runs[run][1] = attn[run]
by = collections.defaultdict(list)
for run, (g, a) in runs.items():
    m = re.fullmatch(rf"{d}/k-(\d+)-(\w+)-r(\d+)\.jsonl", run)
    by[(int(m.group(1)), m.group(2))].append((g, a))
print("\nsize | total NAX GEMM us per prefill, attention-normalised (raw) per arm | vs first arm")
for size in sorted({k[0] for k in by}):
    arms = sorted({k[1] for k in by if k[0] == size})
    refa = statistics.median(a for k in by if k[0] == size for _, a in by[k])
    vals = {a: statistics.median(g * refa / at for g, at in by[(size, a)]) for a in arms}
    raws = {a: statistics.median(g for g, _ in by[(size, a)]) for a in arms}
    base = vals.get("p45", list(vals.values())[0])
    print(f"{size}(M={tm[size]}) | " + " | ".join(f"{a} {vals[a]/1000:.1f} ms ({raws[a]/1000:.1f}) {100*(vals[a]/base-1):+.1f}%" for a in arms) + f" | n={len(by[(size, arms[0])])}")
