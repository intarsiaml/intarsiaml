#!/usr/bin/env python3
"""Per-decode-step kernel table from a rocprofv3 kernel_trace.csv.
usage: analyze.py kernel_trace.csv [step_end_substr=argmax]
A step = launches up to and including an `argmax` kernel. The steady-state step shape (launch count) is the mode."""
import csv, sys, collections, re

path = sys.argv[1]
endk = sys.argv[2] if len(sys.argv) > 2 else "argmax"
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], r.get("Grid_Size_X", ""), r.get("Workgroup_Size_X", "")))
rows.sort()

def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void ", "", n)
    d = 0
    for i, ch in enumerate(n):          # cut the argument list: the first '(' outside the template angle brackets
        if ch == "<": d += 1
        elif ch == ">": d -= 1
        elif ch == "(" and d == 0:
            n = n[:i]
            break
    return n.replace("ik_bf16", "bf16")[:70]

steps, cur = [], []
for r in rows:
    cur.append(r)
    if endk in r[2]:
        steps.append(cur)
        cur = []
cnt = collections.Counter(len(s) for s in steps)
print("steps:", len(steps), "launch-count histogram (top):", cnt.most_common(4))
mode = cnt.most_common(1)[0][0]
dec = [s for s in steps if len(s) == mode]
# drop the first decode step (warm-up)
dec = dec[2:]
n = len(dec)
print(f"decode steps analysed: {n}, launches/step {mode}")
agg = collections.defaultdict(lambda: [0, 0.0])
tot_k = tot_w = 0.0
gaps_in = 0.0
for s in dec:
    wall = s[-1][1] - s[0][0]
    k = sum(e - b for b, e, *_ in s)
    tot_k += k; tot_w += wall
    for b, e, name, *_ in s:
        a = agg[short(name)]
        a[0] += 1; a[1] += (e - b)
print(f"per step: kernel time {tot_k/n/1e6:.3f} ms, step wall (first start to last end) {tot_w/n/1e6:.3f} ms, idle inside step {(tot_w-tot_k)/n/1e6:.3f} ms ({100*(tot_w-tot_k)/tot_w:.1f}% of wall)")
# gap between steps (end of argmax -> start of next step's first kernel), only for consecutive decode steps
ig = []
for a, b in zip(steps, steps[1:]):
    if len(a) == mode and len(b) == mode:
        ig.append(b[0][0] - a[-1][1])
if ig:
    ig.sort()
    print(f"inter-step gap (argmax end -> next step start): median {ig[len(ig)//2]/1e3:.0f} us, mean {sum(ig)/len(ig)/1e3:.0f} us")
    print(f"=> full per-token period ~ {(tot_w/n + sum(ig)/len(ig))/1e6:.3f} ms")
# intra-step gap histogram
gs = []
for s in dec:
    for x, y in zip(s, s[1:]):
        gs.append(y[0] - x[1])
gs.sort()
if gs:
    q = lambda p: gs[int(p * (len(gs) - 1))]
    print(f"intra-step gaps (ns): p10 {q(.1)} p50 {q(.5)} p90 {q(.9)} p99 {q(.99)} max {gs[-1]}; sum/step {sum(gs)/n/1e6:.3f} ms")
print(f"\n{'kernel':<72}{'n/step':>7}{'us/step':>10}{'avg us':>8}{'%kern':>7}")
for name, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{name:<72}{c/n:>7.1f}{t/n/1e3:>10.1f}{t/c/1e3:>8.1f}{100*t/tot_k:>6.1f}%")

# fp8 GEMV launches grouped by grid (the 35B's decode shapes; blocks = grid / workgroup): achieved GB/s against 627.
# bytes per grid: out/o_proj 2048x4096 (128 blocks at 16 rows), attention q 8192x2048 (512), GDN qkv+z pair 12288x2048 (768),
# attention k+v pair 1024x2048 (64). Other grids are listed without a figure.
MB = {128: 8.389, 512: 16.777, 768: 25.166, 64: 2.097}
sh = collections.defaultdict(lambda: [0, 0.0])
for s in dec:
    for b, e, name, gx, wx in s:
        if "fp8_gemv" in name and gx and wx:
            blocks = int(gx) // int(wx)
            a = sh[(short(name), blocks)]
            a[0] += 1; a[1] += (e - b)
print(f"\n{'fp8 kernel / blocks':<50}{'n/step':>7}{'avg us':>8}{'MB':>8}{'GB/s':>7}{'%627':>6}{'us/step':>9}")
for (name, blocks), (c, t) in sorted(sh.items(), key=lambda kv: -kv[1][1]):
    avg = t / c / 1e3
    mb = MB.get(blocks)
    g = f"{1e3 * mb / avg:>7.0f}{100 * 1e3 * mb / avg / 627:>5.0f}%" if mb else f"{'':>12}"
    print(f"{name + ' g' + str(blocks):<50}{c/n:>7.1f}{avg:>8.1f}{(mb or 0):>8.2f}{g}{t/n/1e3:>9.1f}")
