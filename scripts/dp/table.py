#!/usr/bin/env python3
"""Lane dp: tables and the verdict from scripts/dp/window.sh's output directory.  table.py <dir>
Aggregate = the SUM over servers of (completion tokens / that server's wall), median of 3 bursts per visit (load.py)."""
import glob, json, os, statistics, sys

d = sys.argv[1]
L = {}
for f in sorted(glob.glob(os.path.join(d, "[TDX][0-9]-load.json"))):
    L[os.path.basename(f)[:2]] = {r["streams"]: r for r in json.load(open(f))["rows"]}
NS = [1, 8, 16, 32]


def cell(v, n, key, fmt="{:.1f}"):
    r = L.get(v, {}).get(n)
    return fmt.format(r[key]) if r and r.get(key) is not None else "-"


print("### Aggregate decode tok/s (sum over servers; wall includes prefill), median of 3 bursts per visit\n")
print("| total streams | T1 | D1 | D1/T1 | T2 | D2 | D2/T2 | combined D1 / D2 (all tokens / longest wall) |")
print("|---|---|---|---|---|---|---|---|")
ratio = {}
for n in NS:
    row = [str(n)]
    for k in (1, 2):
        t, dd = L.get(f"T{k}", {}).get(n), L.get(f"D{k}", {}).get(n)
        row += [cell(f"T{k}", n, "agg_sum_median"), cell(f"D{k}", n, "agg_sum_median")]
        if t and dd:
            ratio[(n, k)] = dd["agg_sum_median"] / t["agg_sum_median"]
            row.append(f"{ratio[(n, k)]:.2f}x")
        else:
            row.append("-")
    row.append(f"{cell('D1', n, 'agg_combined_median')} / {cell('D2', n, 'agg_combined_median')}")
    print("| " + " | ".join(row) + " |")

print("\n### Per-stream decode tok/s (median over streams and bursts) and TTFT p50 / p95 (s, pooled over the visit's bursts)\n")
print("| total streams | arm | visit | per-stream decode | TTFT p50 | TTFT p95 | TTFT max | errors |")
print("|---|---|---|---|---|---|---|---|")
for n in NS:
    for v in ("T1", "D1", "T2", "D2"):
        r = L.get(v, {}).get(n)
        if r:
            print(f"| {n} | {v[0]} | {v} | {r['stream_decode_median']:.1f} | {r['ttft_p50']:.2f} | {r['ttft_p95']:.2f} | {r['ttft_max']:.2f} | {r['errors']} |")
if "X1" in L:
    print("\n### Extra (outside the rule): T with --max-batch 32\n")
    print("| total streams | X1 agg | per-stream decode | TTFT p50 / p95 | D1 / D2 agg for comparison |")
    print("|---|---|---|---|---|")
    for n, r in sorted(L["X1"].items()):
        print(f"| {n} | {r['agg_sum_median']:.1f} | {r['stream_decode_median']:.1f} | {r['ttft_p50']:.2f} / {r['ttft_p95']:.2f} | {cell('D1', n, 'agg_sum_median')} / {cell('D2', n, 'agg_sum_median')} |")

print("\n### Verdict against the rule (D/T >= 1.3 at 16 and at 32 total streams, in both visits)\n")
ok = True
for n in (16, 32):
    for k in (1, 2):
        r = ratio.get((n, k))
        print(f"- N={n}, visit pair {k}: D/T = {'-' if r is None else f'{r:.2f}x'} -> {'pass' if r and r >= 1.3 else 'FAIL'}")
        ok = ok and bool(r and r >= 1.3)
print(f"\n**D is {'' if ok else 'NOT '}better for throughput by the rule.**")
print("\n### 1-stream latency cost of D against T\n")
for k in (1, 2):
    t, dd = L.get(f"T{k}", {}).get(1), L.get(f"D{k}", {}).get(1)
    if t and dd:
        print(f"- pair {k}: tok/s T {t['agg_sum_median']:.1f} vs D {dd['agg_sum_median']:.1f} ({(dd['agg_sum_median'] / t['agg_sum_median'] - 1) * 100:+.1f} %), "
              f"TTFT p50 T {t['ttft_p50']:.3f} vs D {dd['ttft_p50']:.3f} s, TTFT p95 T {t['ttft_p95']:.3f} vs D {dd['ttft_p95']:.3f} s")

print("\n### Peak VRAM per card (GiB, sysfs mem_info_vram_used, sampled every second during the visit)\n")
base = open(os.path.join(d, "vram-baseline.txt")).read().split() if os.path.exists(os.path.join(d, "vram-baseline.txt")) else []
if base:
    print(f"Baseline after radiance stopped: card0 {int(base[0]) / 2**30:.2f}, card1 {int(base[1]) / 2**30:.2f} GiB.\n")
print("| visit / probe | card0 peak | card1 peak | samples |")
print("|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(d, "*-vram.tsv"))):
    rows = [l.split() for l in open(f) if len(l.split()) == 3]
    if rows:
        print(f"| {os.path.basename(f)[:-9]} | {max(int(r[1]) for r in rows) / 2**30:.2f} | {max(int(r[2]) for r in rows) / 2**30:.2f} | {len(rows)} |")

print("\n### Identity (greedy text sha256, 4 fixed prompts: solo on each server, then all at once on every server)\n")
for f in sorted(glob.glob(os.path.join(d, "[TDX][0-9]-ident.json"))):
    j = json.load(open(f))
    print(f"- {os.path.basename(f)[:2]}: identical across servers and load = {j['same_across_servers_and_load']}; prompt 0..3 shas: {list(j['solo'].values())[0]}")
for a, b in (("T1", "T2"), ("D1", "D2")):
    fa, fb = os.path.join(d, f"{a}-ident.json"), os.path.join(d, f"{b}-ident.json")
    if os.path.exists(fa) and os.path.exists(fb):
        ja, jb = json.load(open(fa)), json.load(open(fb))
        print(f"- {a} vs {b} (fresh servers): solo shas equal = {list(ja['solo'].values())[0] == list(jb['solo'].values())[0]}")

print("\n### CPU-starvation check (D1): 16 streams on server 0 alone against the same server's share of the 32-stream dual run\n")
if "D1" in L and 32 in L["D1"] and os.path.exists(os.path.join(d, "D1-solo16.json")):
    solo = json.load(open(os.path.join(d, "D1-solo16.json")))["rows"][0]
    dual = L["D1"][32]
    s_solo = statistics.median(b["per_server"][next(iter(b["per_server"]))]["agg_tok_s"] for b in solo["bursts"])
    per_srv = {}
    for b in dual["bursts"]:
        for s, v in b["per_server"].items():
            per_srv.setdefault(s, []).append(v["agg_tok_s"])
    print(f"- server 0 alone, 16 streams: {s_solo:.1f} tok/s; in the dual run (16 + 16), per server: " + ", ".join(f"{s.split(':')[-1].split('/')[0]} = {statistics.median(v):.1f}" for s, v in per_srv.items())
          + f" ({[f'{statistics.median(v) / s_solo * 100:.0f} % of solo' for v in per_srv.values()]})")
