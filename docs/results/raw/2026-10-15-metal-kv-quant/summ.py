#!/usr/bin/env python3
"""usage: summ.py <dir with *.err and runs.tsv> <macmon.jsonl>: one row per generate run: kv dtype, ctx, prefill / decode speed,
GPU clock (median over the busy samples of the decode window; of the prefill window), flag when under 1450 MHz."""
import sys, os, re, json, datetime, statistics as st
d, mm = sys.argv[1], sys.argv[2]
samples = []
for l in open(mm):
    try:
        j = json.loads(l)
        samples.append((datetime.datetime.fromisoformat(j["timestamp"]).timestamp(), j["gpu_usage"][0], j["gpu_usage"][1], j["gpu_power"], j["temp"]["gpu_temp_avg"]))
    except Exception:
        pass
def clock(a, b):
    w = [s for s in samples if a <= s[0] <= b and s[2] > 0.3]
    if len(w) < 3: return None
    f = sorted(x[1] for x in w)
    return st.median(f), 100 * sum(1 for x in f if x < 1450) / len(f), st.median(x[3] for x in w), max(x[4] for x in w), len(w)
print(f"{'run':<28} {'prefill ms':>10} {'prefill t/s':>11} {'decode t/s':>10} | {'decode MHz':>10} {'%<1450':>7} {'W':>4} {'maxC':>5} {'n':>4} | {'prefill MHz':>11} {'%<1450':>7}  flag")
for line in open(os.path.join(d, "runs.tsv")):
    s, e, lab = line.rstrip("\n").split("\t")
    s, e = float(s), float(e)
    err = open(os.path.join(d, lab + ".err")).read()
    m = re.search(r"prefill (\d+) ms \(([\d.]+) tok/s\) \| decode (\d+) ms \(([\d.]+) tok/s\)", err)
    if not m: print(f"{lab:<28} no timing line"); continue
    pre_ms, pre_ts, dec_ms, dec_ts = int(m[1]), float(m[2]), int(m[3]), float(m[4])
    cd = clock(e - dec_ms / 1000 - 0.2, e)
    cp = clock(s, e - dec_ms / 1000 - 0.2)
    flag = ""
    if cd and (cd[0] < 1450 or cd[1] >= 40): flag = "DECODE-CLOCK<1450"
    if cp and (cp[0] < 1450 or cp[1] >= 40): flag += " PREFILL-CLOCK<1450"
    f = lambda c, i, fmt: (fmt % c[i]) if c else "-"
    print(f"{lab:<28} {pre_ms:>10} {pre_ts:>11.1f} {dec_ts:>10.2f} | {f(cd,0,'%.0f'):>10} {f(cd,1,'%.0f'):>7} {f(cd,2,'%.0f'):>4} {f(cd,3,'%.0f'):>5} {f(cd,4,'%d'):>4} | {f(cp,0,'%.0f'):>11} {f(cp,1,'%.0f'):>7}  {flag}")
