#!/usr/bin/env python3
"""Lane mtt: TTFT table from ttft.sh logs. usage: table.py LOG... ; one row per arm instance, then medians per arm kind
(tag without its trailing digit and 'cb')."""
import re, statistics as st, sys, collections
for log in sys.argv[1:]:
    arms, cur = [], None
    for l in open(log, errors="replace"):
        m = re.match(r"## (\S+) \[(.*?)\] \[(.*?)\] (\S+) thermal \S+ (\d+) load ([\d.]+)s", l)
        if m: cur = dict(tag=m.group(1), t=m.group(4), th=m.group(5), load=float(m.group(6)), ttft=[], gaps=[]); arms.append(cur); continue
        m = re.match(r"nonce \S+: ttft (\d+) ms, events \d+, first gaps ms \[(.*?)\]", l)
        if m and cur:
            cur["ttft"].append(int(m.group(1))); g = [float(x) for x in m.group(2).split(",") if x.strip()]
            nz = [x for x in g if x > 0.5]; cur["gaps"] += nz[1:]
    print(f"\n{log.split('/')[-1]}\n| arm | start | thermal level | load s | TTFT ms | step ms (median) |\n|---|---|---:|---:|---|---:|")
    kinds = collections.defaultdict(lambda: dict(ttft=[], load=[]))
    for a in arms:
        print(f"| {a['tag']} | {a['t']} | {a['th']} | {a['load']:.1f} | {' / '.join(map(str, a['ttft']))} | {st.median(a['gaps']) if a['gaps'] else float('nan'):.1f} |")
        k = re.sub(r"\d*(cb)?$", "", a["tag"]); kinds[k]["ttft"] += a["ttft"]; kinds[k]["load"].append(a["load"])
    print("\n| arm kind | calls | TTFT median ms | min..max | load s |\n|---|---:|---:|---|---|")
    for k, v in kinds.items():
        print(f"| {k} | {len(v['ttft'])} | {st.median(v['ttft']):.0f} | {min(v['ttft'])}..{max(v['ttft'])} | {' / '.join(f'{x:.1f}' for x in v['load'])} |")
