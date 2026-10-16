#!/usr/bin/env python3
"""Chunk 256 vs 2048: GPU busy (sum of command-buffer GPU time), first-start-to-last-end span and the engine's prefill wall ms.
python3 cbchunk.py"""
import glob, json, re, statistics, collections
d = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob("cbchunk/cb-*-r*.jsonl")):
    m = re.fullmatch(r"cbchunk/cb-(\d+)-c(\d+)-r(\d+)-(\d+)\.jsonl", f)
    if not m: continue
    S, C = int(m.group(1)), int(m.group(2))
    rs = [json.loads(l) for l in open(f)]
    d[(S, C)]["busy"].append(sum(r["e"] - r["s"] for r in rs) * 1e3)
    d[(S, C)]["span"].append((rs[-1]["e"] - rs[0]["s"]) * 1e3)
    d[(S, C)]["cbs"].append(len(rs))
    t = re.findall(r"prefill (\d+) ms", open(f.replace(".jsonl", ".err")).read())
    if t: d[(S, C)]["wall"].append(float(t[-1]))
print("prompt | chunk | n | GPU busy ms median [min..max] | span ms | prefill wall ms | command buffers")
for (S, C), v in sorted(d.items()):
    f = lambda k: f"{statistics.median(v[k]):.0f} [{min(v[k]):.0f}..{max(v[k]):.0f}]" if v[k] else "-"
    print(f"{S} | {C} | {len(v['busy'])} | {f('busy')} | {f('span')} | {f('wall')} | {statistics.median(v['cbs']):.0f}")
for S in sorted({k[0] for k in d}):
    a, b = d[(S, 256)], d[(S, 2048)]
    if a["busy"] and b["busy"]:
        print(f"{S}: c256 / c2048: busy {statistics.median(a['busy'])/statistics.median(b['busy']):.3f}, span {statistics.median(a['span'])/statistics.median(b['span']):.3f}"
              + (f", wall {statistics.median(a['wall'])/statistics.median(b['wall']):.3f}" if a["wall"] and b["wall"] else ""))
