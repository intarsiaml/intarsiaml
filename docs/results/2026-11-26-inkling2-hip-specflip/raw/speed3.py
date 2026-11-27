#!/usr/bin/env python3
"""Lane ink-sampled-adapt speed analysis (from lane ink-sampled's speed.py). `speed3.py DIR [--macmon]`: DIR holds `<arm>-r<rep>.err`
(+ `.t`) with arms off / fix / pol and temperature suffix 1 (1.0) / 07 (0.7), one process per arm and rep, speed.json (warm, short_a/b/c,
len1k, len4k). Per class (short = the three short prompts pooled: sum tokens / sum ms) and per prompt: ratio of two arms per rep (pairs
within a rep), geometric mean and a 95 % t-interval over reps on the log ratios, for fix/off, pol/off and pol/fix. Acceptance, tokens per
pass and plain-step tokens (tokens - 1 - passes - accepted) per arm from the `spec[name]` lines. Ids: every process of one temperature must
emit the same ids (same seed; the rule is seed-exact). Then the pre-registered rule (results doc section 3). With --macmon (Metal) each
process's GPU seconds above 65 W and mean GPU power, from DIR/macmon.jsonl and the `.t` start / end stamps."""
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

d = Path(sys.argv[1])
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
spl = re.compile(r"^spec\[([^\]]*)\]: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")
CLASSES = {"short": ["short_a", "short_b", "short_c"], "1k": ["len1k"], "4k": ["len4k"]}
PROMPTS = ["short_b", "short_a", "short_c", "len1k", "len4k"]
TEMPS = [("1", "1.0"), ("07", "0.7")]


def read(f):
    rec = {"dec": {}, "ids": {}, "spec": {}}
    for line in f.read_text(errors="replace").splitlines():
        m = dec.match(line)
        if m:
            rec["dec"][m.group(1)] = (int(m.group(2)), int(m.group(3)))
        m = idl.match(line)
        if m:
            rec["ids"][m.group(1)] = m.group(2).strip()
        m = spl.match(line)
        if m:
            rec["spec"][m.group(1)] = tuple(int(x) for x in m.groups()[1:])
    return rec


mm = []
if "--macmon" in sys.argv and (d / "macmon.jsonl").exists():
    for l in open(d / "macmon.jsonl"):
        try:
            j = json.loads(l)
            mm.append((datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp(), j["gpu_power"]))
        except Exception:
            pass

data = {}
for e in sorted(d.glob("*-r*.err")):
    arm, rep = e.stem.split("-r")
    data[(arm, int(rep))] = read(e)
reps = sorted({r for _, r in data})
print(f"processes: {len(data)}, reps {reps}")
ids_ok = True
for t, tl in TEMPS:
    ids = {k: v["ids"] for k, v in data.items() if k[0].endswith(t) and not (t == "1" and k[0].endswith("07"))}
    ref = next(iter(ids.values()), None)
    bad = [k for k, v in ids.items() if v != ref]
    ids_ok &= not bad
    print(f"T {tl}: ids identical across {len(ids)} processes (all arms): {'yes' if not bad else 'NO ' + str(bad)}")


def rate(rec, names):
    tok = sum(rec["dec"][n][0] for n in names if n in rec["dec"])
    ms = sum(rec["dec"][n][1] for n in names if n in rec["dec"])
    return tok / ms * 1e3 if ms else None


def ratio(num, den, t, names):
    """geo mean, CI, n of rate(num) / rate(den) over reps with both"""
    lr = []
    for r in reps:
        a, b = data.get((f"{den}{t}", r)), data.get((f"{num}{t}", r))
        if a and b:
            x, y = rate(a, names), rate(b, names)
            if x and y:
                lr.append(math.log(y / x))
    if not lr:
        return None
    n = len(lr)
    m = sum(lr) / n
    if n > 1:
        sd = math.sqrt(sum((v - m) ** 2 for v in lr) / (n - 1))
        h = T975.get(n - 1, 2.0) * sd / math.sqrt(n)
        ci = (math.exp(m - h), math.exp(m + h))
    else:
        ci = None
    return math.exp(m), ci, n


def fmt(r):
    if not r:
        return "-"
    g, ci, n = r
    return f"**{g:.3f}** [{ci[0]:.3f}-{ci[1]:.3f}]" if ci else f"**{g:.3f}**"


def stats(arm, t, names):
    """acceptance, tokens per pass, plain-step tokens per process (mean over reps)"""
    p = q = acc = tok = 0
    nr = 0
    for r in reps:
        b = data.get((f"{arm}{t}", r))
        if not b:
            continue
        nr += 1
        for nm in names:
            if nm in b["spec"]:
                p += b["spec"][nm][0]
                q += b["spec"][nm][1]
                acc += b["spec"][nm][2]
            if nm in b["dec"]:
                tok += b["dec"][nm][0] - 1
    if not nr:
        return "-"
    a = f"{100 * acc / q:.1f} %" if q else "-"
    tpp = f"{(p + acc) / p:.2f}" if p else "-"
    plain = (tok - p - acc) / nr
    return f"{a}, {tpp}, {plain:.0f}"


def mean_rate(arm, t, names):
    v = [rate(data[(f"{arm}{t}", r)], names) for r in reps if (f"{arm}{t}", r) in data]
    v = [x for x in v if x]
    return sum(v) / len(v) if v else float("nan")


print("\n| T | class | off tok/s | fix tok/s | pol tok/s | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |")
print("|---|---|---:|---:|---:|---|---|---|---|")
res = {}
for t, tl in TEMPS:
    for cls, names in CLASSES.items():
        f, p, pf = ratio("fix", "off", t, names), ratio("pol", "off", t, names), ratio("pol", "fix", t, names)
        res[(tl, cls)] = (f, p, pf)
        print(
            f"| {tl} | {cls} | {mean_rate('off', t, names):.2f} | {mean_rate('fix', t, names):.2f} | {mean_rate('pol', t, names):.2f} | "
            f"{fmt(f)} | {fmt(p)} | {fmt(pf)} | {stats('fix', t, names)} / {stats('pol', t, names)} |"
        )

print("\n| T | prompt | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |")
print("|---|---|---|---|---|---|")
pp = {}
for t, tl in TEMPS:
    for nm in PROMPTS:
        f, p, pf = ratio("fix", "off", t, [nm]), ratio("pol", "off", t, [nm]), ratio("pol", "fix", t, [nm])
        pp[(tl, nm)] = p
        print(f"| {tl} | {nm} | {fmt(f)} | {fmt(p)} | {fmt(pf)} | {stats('fix', t, [nm])} / {stats('pol', t, [nm])} |")

print("\npre-registered rule (results doc section 3):")
c1 = all(v and v[0] >= 0.98 for v in pp.values())
worst = min(((v[0], k) for k, v in pp.items() if v), default=None)
print(f"  1. every prompt pol / off >= 0.98: {'PASS' if c1 else 'FAIL'} (lowest {worst[0]:.3f} at T {worst[1][0]} {worst[1][1]})" if worst else "  1. n/a")
c2 = True
for k, (f, p, _) in res.items():
    if f and p:
        ok = p[0] >= 0.98 * f[0]
        c2 &= ok
        print(f"  2. T {k[0]} {k[1]}: pol / off {p[0]:.3f} vs 0.98 x fix / off {0.98 * f[0]:.3f}: {'ok' if ok else 'FAIL'}")
print(f"  2. every class keeps >= 98 % of fixed k's ratio: {'PASS' if c2 else 'FAIL'}")
print(f"  3 (speed processes). ids identical across all arms: {'PASS' if ids_ok else 'FAIL'}")

if mm:
    print("\nper process: GPU s above 65 W, mean GPU W (macmon, 250 ms samples), short / 1k / 4k tok/s")
    for k in sorted(data, key=lambda k: (k[1], k[0])):
        tf = d / f"{k[0]}-r{k[1]}.t"
        if not tf.exists():
            continue
        t0, t1 = (float(x) for x in tf.read_text().split()[:2])
        s = [w for (ts, w) in mm if t0 <= ts <= t1]
        hi = sum(1 for w in s if w > 65) * 0.25
        mw = sum(s) / len(s) if s else float("nan")
        rs = " ".join(f"{rate(data[k], v) or 0:.2f}" for v in CLASSES.values())
        print(f"  {k[0]:6s} r{k[1]}  {hi:5.1f} s  {mw:5.1f} W  {rs}")

# lane ink-hip-specflip (sampled DFlash2, the coordinator's gate): `--ref`: bar 2 against #543's DSpark-adaptive ratios on HIP
if "--ref" in sys.argv:
    REF = {("1.0", "short"): 1.172, ("1.0", "1k"): 1.042, ("1.0", "4k"): 1.466, ("0.7", "short"): 1.163, ("0.7", "1k"): 1.068, ("0.7", "4k"): 1.458}
    print("\nthe coordinator's gate (sampled DFlash2 vs flip A's DSpark-adaptive, #543 HIP ratios):")
    print(f"  1. every prompt pol / off >= 0.98: {'PASS' if c1 else 'FAIL'} (lowest {worst[0]:.3f} at T {worst[1][0]} {worst[1][1]})" if worst else "  1. n/a")
    g2 = True
    for k, (f, p, _) in res.items():
        if p:
            ok = p[0] >= 0.98 * REF[k]
            g2 &= ok
            print(f"  2. T {k[0]} {k[1]}: pol / off {p[0]:.3f} vs 0.98 x {REF[k]} = {0.98 * REF[k]:.3f}: {'ok' if ok else 'FAIL'}")
    print(f"  2. every class >= 0.98 x DSpark-adaptive: {'PASS' if g2 else 'FAIL'}")
    print(f"  ids identical across the arms (speed processes): {'PASS' if ids_ok else 'FAIL'}")
