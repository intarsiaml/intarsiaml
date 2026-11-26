#!/usr/bin/env python3
"""Lane ink-sampled speed analysis. `speed.py DIR [--macmon]`: DIR holds `<arm>-r<rep>.err` (+ `.out`, `.t`) from slotB.sh (Metal) or
w2.sh (HIP): arms off1 / on1 (temperature 1.0) and off07 / on07 (0.7), one process per arm and rep, speed.json (warm, short_a/b/c, len1k,
len4k). Decode tok/s per class (short = the three short prompts pooled: sum tokens / sum ms), ratio on / off per rep (pairs), geometric
mean and a 95 % t-interval over reps on the log ratios. Acceptance per pass per class from the `spec[name]` lines. Ids: every arm and rep
of one temperature must emit the same ids (same seed; the rule is seed-exact). With --macmon (Metal), each process's GPU seconds above
65 W and its mean GPU power, from DIR/macmon.jsonl and the `.t` start / end stamps."""
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


def read(f):
    rec = {"dec": {}, "ids": {}, "spec": {}}
    for line in f.read_text(errors="replace").splitlines():
        m = dec.match(line)
        if m:
            rec["dec"][m.group(1)] = (int(m.group(2)) - 1, int(m.group(3)))
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
for t in ["1", "07"]:
    ids = {k: v["ids"] for k, v in data.items() if k[0].endswith(t)}
    ref = next(iter(ids.values()), None)
    bad = [k for k, v in ids.items() if v != ref]
    print(f"T {'1.0' if t == '1' else '0.7'}: ids identical across {len(ids)} processes (on and off): {'yes' if not bad else 'NO ' + str(bad)}")


def rate(rec, names):
    tok = sum(rec["dec"][n][0] for n in names if n in rec["dec"])
    ms = sum(rec["dec"][n][1] for n in names if n in rec["dec"])
    return tok / ms * 1e3 if ms else None


print("\n| T | class | off tok/s (mean) | on tok/s (mean) | on / off (geo mean) | 95 % CI | pairs | acceptance | tokens / pass |")
print("|---|---|---:|---:|---:|---|---:|---:|---:|")
for t, tl in [("1", "1.0"), ("07", "0.7")]:
    for cls, names in CLASSES.items():
        lr, offs, ons = [], [], []
        for r in reps:
            a, b = data.get((f"off{t}", r)), data.get((f"on{t}", r))
            if not a or not b:
                continue
            x, y = rate(a, names), rate(b, names)
            if x and y:
                offs.append(x)
                ons.append(y)
                lr.append(math.log(y / x))
        if not lr:
            continue
        n = len(lr)
        m = sum(lr) / n
        if n > 1:
            sd = math.sqrt(sum((v - m) ** 2 for v in lr) / (n - 1))
            h = T975.get(n - 1, 2.0) * sd / math.sqrt(n)
            ci = f"[{math.exp(m - h):.3f}, {math.exp(m + h):.3f}]"
        else:
            ci = "n/a"
        p = q = acc = 0
        for r in reps:
            b = data.get((f"on{t}", r))
            for nm in names:
                if b and nm in b["spec"]:
                    p += b["spec"][nm][0]
                    q += b["spec"][nm][1]
                    acc += b["spec"][nm][2]
        accs = f"{100 * acc / q:.1f} %" if q else "-"
        tpp = f"{(p + acc) / p:.2f}" if p else "-"
        print(f"| {tl} | {cls} | {sum(offs) / n:.2f} | {sum(ons) / n:.2f} | {math.exp(m):.3f} | {ci} | {n} | {accs} | {tpp} |")

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
        print(f"  {k[0]:5s} r{k[1]}  {hi:5.1f} s  {mw:5.1f} W  {rs}")
