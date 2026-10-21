#!/usr/bin/env python3
"""servesum.py <gates-dir>...: the served A/B per arm: bench speed rows (TTFT s, decode tok/s at 128 and 4k), the median
wall time of the 20 short EOS-terminated chat requests (eosreq.py), and the macmon medians over busy samples (GPU MHz,
CPU load; G3's admission window is >= 1300 MHz and <= 0.30)."""
import gzip, json, os, re, statistics as st, sys

for d in sys.argv[1:]:
    print(f"## {d}")
    print(f"{'arm':7} {'ttft128':>8} {'dec128':>7} {'ttft4k':>7} {'dec4k':>6} {'eos wall ms':>12} {'MHz':>5} {'cpu':>5} admitted")
    arms = {"base": [], "fb": []}
    for arm in ["base-1", "fb-1", "fb-2", "base-2", "base-3", "fb-3"]:
        a = os.path.join(d, "ab-serve", arm)
        rows = {}
        for l in open(a + "-bench.txt"):
            m = re.match(r"\s+(128|4096)\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)", l)
            if m:
                rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        m = re.search(r"median wall ([\d.]+) ms", open(a + "-eos.txt").read())
        eos = float(m.group(1)) if m else float("nan")
        s = []
        for l in gzip.open(a + "-macmon.jsonl.gz", "rt"):
            try:
                x = json.loads(l)
            except Exception:
                continue
            if x["gpu_usage"][1] > 0.5:
                s.append((x["gpu_usage"][0], x["cpu_usage_pct"]))
        mhz, cpu = st.median(v[0] for v in s), st.median(v[1] for v in s)
        ok = mhz >= 1300 and cpu <= 0.30
        arms[arm.split("-")[0]].append((rows["128"][1], rows["4096"][1], eos))
        print(f"{arm:7} {rows['128'][0]:8.3f} {rows['128'][1]:7.1f} {rows['4096'][0]:7.3f} {rows['4096'][1]:6.1f} {eos:12.1f} {mhz:5.0f} {cpu:5.2f} {'yes' if ok else 'no'}")
    for k, v in arms.items():
        print(f"{k}: mean decode 128 {st.mean(x[0] for x in v):.2f}, 4k {st.mean(x[1] for x in v):.2f} tok/s; eos wall {st.mean(x[2] for x in v):.1f} ms")
