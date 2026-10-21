#!/usr/bin/env python3
"""clock.py <profile-dir>: per generate run, decode tok/s vs GPU MHz / GPU W / CPU load / temp in the decode window
(end - decode_ms .. end - 0.15 s, from times.txt and the run's .err), from that run's macmon log (100 ms). Prints
a table and the Pearson r of decode tok/s against each covariate (clk4k-* and ab*-* runs)."""
import datetime as dt, gzip, json, os, re, statistics as st, sys

D = sys.argv[1]
ends = dict(l.split()[0:3:2] for l in open(os.path.join(D, "times.txt")))
rows = []
for name, end in ends.items():
    err = open(os.path.join(D, f"{name}.err")).read()
    m = re.search(r"decode (\d+) ms \(([\d.]+) tok/s\)", err)
    if not m:
        continue
    dms, tps = int(m.group(1)), float(m.group(2))
    end = float(end); t0, t1 = end - dms / 1e3 - 0.1, end - 0.15
    f = os.path.join(D, f"{name}-macmon.jsonl.gz")
    s = []
    for l in gzip.open(f, "rt"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        t = dt.datetime.fromisoformat(d["timestamp"]).timestamp()
        if t0 <= t <= t1:
            s.append((d["gpu_usage"][0], d["gpu_usage"][1], d["gpu_power"], d["cpu_usage_pct"], d["temp"]["gpu_temp_avg"], d.get("sys_power", 0)))
    if not s:
        continue
    md = lambda i: st.median(x[i] for x in s)
    rows.append((name, tps, md(0), md(1), md(2), md(3), md(4), md(5), len(s)))
print("| run | decode tok/s | GPU MHz | GPU util | GPU W | CPU load | GPU temp C | sys W | samples |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r[0]} | {r[1]:.2f} | {r[2]:.0f} | {r[3]:.2f} | {r[4]:.1f} | {r[5]:.2f} | {r[6]:.0f} | {r[7]:.0f} | {r[8]} |")
sel = [r for r in rows if r[0].startswith(("clk4k", "ab"))]
def pr(i):
    x = [r[i] for r in sel]; y = [r[1] for r in sel]
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y)); den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else float("nan")
print(f"\n4k runs (n={len(sel)}): r(tok/s, MHz)={pr(2):+.2f}  r(tok/s, GPU W)={pr(4):+.2f}  r(tok/s, CPU load)={pr(5):+.2f}  r(tok/s, temp)={pr(6):+.2f}")
x=[r[2] for r in sel]; y=[r[1] for r in sel]; mx,my=st.mean(x),st.mean(y); b=sum((a-mx)*(c-my) for a,c in zip(x,y))/sum((a-mx)**2 for a in x)
print(f"fit: tok/s = {my:.2f} + {b:.4f} x (MHz - {mx:.0f}); at 1450 MHz {my+b*(1450-mx):.2f}, at 1600 MHz {my+b*(1600-mx):.2f}")
