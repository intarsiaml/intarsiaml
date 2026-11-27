#!/usr/bin/env python3
"""Lane ink-metal-sampled2 gate analysis (from lane ink-sampled-adapt's speed3.py, results doc 2026-11-27 section 1.2).
`speed4.py DIR [--macmon]`: DIR holds `<arm><t>-s<seed>.err` (+ `.t` start / end stamps), arm off / fix / pol, t 1 (T 1.0) or 07 (T 0.7), one process per
(arm, T, seed), `generate --show-ids` over speed.json (warm, short_a, short_b, short_c, len1k, len4k), 128 tokens.
Per (prompt or class, T, seed): ratio of the two arms' tok/s (class = pooled tokens / pooled ms); over seeds: geometric mean and a 95 % t-interval on the
log ratios (n seeds, t table below). Clause 1 / 2 on the point estimates, exactly as pre-registered; the interval and the strict reading are printed too."""
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

d = Path(sys.argv[1])
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
spl = re.compile(r"^spec\[([^\]]*)\]: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")
CLASSES = {"short": ["short_a", "short_b", "short_c"], "1k": ["len1k"], "4k": ["len4k"]}
PROMPTS = ["short_b", "short_a", "short_c", "len1k", "len4k"]
TEMPS = [("1", "1.0"), ("07", "0.7")]


def read(f):
    rec = {"dec": {}, "ids": {}, "spec": {}}
    for line in f.read_text(errors="replace").splitlines():
        if m := dec.match(line):
            rec["dec"][m.group(1)] = (int(m.group(2)), int(m.group(3)))
        elif m := idl.match(line):
            rec["ids"][m.group(1)] = m.group(2).strip()
        elif m := spl.match(line):
            rec["spec"][m.group(1)] = tuple(int(x) for x in m.groups()[1:])
    return rec


mm = []
if "--macmon" in sys.argv:
    for f in sorted(d.glob("macmon*.jsonl")) + sorted(d.parent.glob("macmon*.jsonl")):
        for l in open(f):
            try:
                j = json.loads(l)
                mm.append((datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp(), j["gpu_power"]))
            except Exception:
                pass

data = {}
for e in sorted(d.glob("*-s*.err")):
    a, s = e.stem.split("-s")
    t = "07" if a.endswith("07") else "1"
    data[(a[: -len(t)], t, int(s))] = read(e)
seeds = sorted({k[2] for k in data})
print(f"processes: {len(data)}, seeds {seeds}")


def rate(rec, names):
    tok = sum(rec["dec"][n][0] for n in names if n in rec["dec"])
    ms = sum(rec["dec"][n][1] for n in names if n in rec["dec"])
    return tok / ms * 1000 if ms else None


def ratio(num, den, t, names):
    lr, per = [], {}
    for s in seeds:
        a, b = data.get((num, t, s)), data.get((den, t, s))
        if not a or not b:
            continue
        x, y = rate(a, names), rate(b, names)
        if x and y:
            lr.append(math.log(x / y))
            per[s] = x / y
    if not lr:
        return None
    n = len(lr)
    m = sum(lr) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in lr) / (n - 1)) if n > 1 else float("nan")
    h = T975.get(n - 1, 2.0) * sd / math.sqrt(n) if n > 1 else float("nan")
    return math.exp(m), math.exp(m - h), math.exp(m + h), n, per


def fmt(r):
    return "-" if not r else f"**{r[0]:.3f}** [{r[1]:.3f}-{r[2]:.3f}]"


def per_seed(r):
    return "-" if not r else " ".join(f"{s}:{v:.3f}" for s, v in sorted(r[4].items()))


def stats(arm, t, names):
    p = q = a = tok = 0
    for s in seeds:
        r = data.get((arm, t, s))
        if not r:
            continue
        for n in names:
            if n in r["spec"]:
                pa, pr, ac = r["spec"][n]
                p, q, a = p + pa, q + pr, a + ac
                tok += r["dec"][n][0]
    if not p:
        return "-"
    return f"{100 * a / q:.1f} %, {(p + a) / p:.2f}, {(tok - 1 * len(seeds) - p - a) / len(seeds):.0f}" if q else "-"


def mean_rate(arm, t, names):
    v = [rate(data[(arm, t, s)], names) for s in seeds if (arm, t, s) in data]
    v = [x for x in v if x]
    return sum(v) / len(v) if v else float("nan")


ids_ok = True
for t, tl in TEMPS:
    for s in seeds:
        ids = {k: v["ids"] for k, v in data.items() if k[1] == t and k[2] == s}
        ref = next(iter(ids.values()), None)
        bad = [k for k, v in ids.items() if v != ref]
        ids_ok &= not bad
        if ids:
            print(f"T {tl} seed {s}: ids identical across {len(ids)} processes (all arms): {'yes' if not bad else 'NO ' + str(bad)}")

print("\n| T | class | off tok/s | fix tok/s | pol tok/s | fix / off | pol / off | pol / fix | acc., tok/pass, plain tokens: fix / pol |")
print("|---|---|---:|---:|---:|---|---|---|---|")
res = {}
for t, tl in TEMPS:
    for cls, names in CLASSES.items():
        f, p, pf = ratio("fix", "off", t, names), ratio("pol", "off", t, names), ratio("pol", "fix", t, names)
        res[(tl, cls)] = (f, p, pf)
        print(f"| {tl} | {cls} | {mean_rate('off', t, names):.2f} | {mean_rate('fix', t, names):.2f} | {mean_rate('pol', t, names):.2f} | "
              f"{fmt(f)} | {fmt(p)} | {fmt(pf)} | {stats('fix', t, names)} / {stats('pol', t, names)} |")
print("\n| T | prompt | fix / off | pol / off | pol / fix | per seed pol / off | acc., tok/pass, plain tokens: fix / pol |")
print("|---|---|---|---|---|---|---|")
pp = {}
for t, tl in TEMPS:
    for nm in PROMPTS:
        f, p, pf = ratio("fix", "off", t, [nm]), ratio("pol", "off", t, [nm]), ratio("pol", "fix", t, [nm])
        pp[(tl, nm)] = (f, p, pf)
        print(f"| {tl} | {nm} | {fmt(f)} | {fmt(p)} | {fmt(pf)} | {per_seed(p)} | {stats('fix', t, [nm])} / {stats('pol', t, [nm])} |")

print("\npre-registered rule (results doc section 1.2):")
c1 = all(v[1] and v[1][0] >= 0.98 for v in pp.values())
worst = sorted(((v[1][0], k) for k, v in pp.items() if v[1]))[:3]
print(f"  1. every prompt pol / off >= 0.98 (point estimate): {'PASS' if c1 else 'FAIL'}; lowest " + ", ".join(f"{w:.3f} T {k[0]} {k[1]}" for w, k in worst))
strict1 = [k for k, v in pp.items() if v[1] and v[1][1] >= 0.98]
print(f"     strict (CI lower bound >= 0.98): {len(strict1)} of {len(pp)} cells")
c2 = True
for k, (f, p, _) in res.items():
    if f and p:
        ok = p[0] >= 0.98 * f[0]
        c2 &= ok
        gain = (p[0] - 1) >= 0.98 * (f[0] - 1)
        print(f"  2. T {k[0]} {k[1]}: pol / off {p[0]:.3f} vs 0.98 x fix / off {0.98 * f[0]:.3f}: {'ok' if ok else 'FAIL'}; gain form {p[0] - 1:+.3f} vs {0.98 * (f[0] - 1):+.3f}: {'ok' if gain else 'miss'}")
print(f"  2. every class keeps >= 98 % of fixed k3's ratio: {'PASS' if c2 else 'FAIL'}")
print(f"  3 (speed processes). ids identical across all arms of a seed and temperature: {'PASS' if ids_ok else 'FAIL'}")
if mm:
    print("\nper process: GPU s above 65 W, mean GPU W (macmon, 250 ms samples), short / 1k / 4k tok/s")
    for k in sorted(data, key=lambda k: (k[2], k[1], k[0])):
        tf = d / f"{k[0]}{k[1]}-s{k[2]}.t"
        if not tf.exists():
            continue
        t0, t1 = (float(x) for x in tf.read_text().split()[:2])
        s = [w for (ts, w) in mm if t0 <= ts <= t1]
        hi = sum(1 for w in s if w > 65) * 0.25
        mw = sum(s) / len(s) if s else float("nan")
        rs = " ".join(f"{rate(data[k], v) or 0:.2f}" for v in CLASSES.values())
        print(f"  {k[0]:4s} T{k[1]:2s} s{k[2]:<3d} {hi:5.1f} s  {mw:5.1f} W  {rs}")
