#!/usr/bin/env python3
"""sideeager: summary table over the *.req.txt files of nsys_req.py (one per traced server). Per server: slow launches per request (r0-r1 warm-up chat-short, r2-r4 PRE code-gen 300,
r5-r6 MID code-gen 800), total slow of all launches, side-stream eager kernels in the session, and the mean over requests r0-r3 (before any request-ending eager step: the same state in both arms) of the median graph-replay
span (ms) and of the median idle inside the replay (us). usage: nsys_tab.py FILE.req.txt ..."""
import re, sys, statistics as st
print(f"{'server':6s} {'slow per request r0..r6':34s} {'slow/launches':14s} {'side eager kern':16s} {'span r0-r3 ms':21s} idle us")
spans = {}
for f in sys.argv[1:]:
    a = f.split("/")[-1].split(".")[0]; sl = []; sp = []; idl = []; tot = n = side = 0
    for l in open(f):
        m = re.match(r" req (\d+): +(\d+) launches, slow +(\d+),.*median idle ([\d.]+) us, span ([\d.]+) ms.*inside \[M (\d+) S (\d+)\]", l)
        if m:
            sl.append(int(m.group(3))); n += int(m.group(2)); tot += int(m.group(3))
            if int(m.group(1)) <= 3: sp.append(float(m.group(5))); idl.append(float(m.group(4)))
        m = re.match(r" req \d+: .*eager before \[M (\d+) S (\d+)\]", l)
        if m: side += int(m.group(2))
    spans[a] = st.mean(sp) if sp else float("nan")
    print(f"{a:6s} {' '.join(f'{x:4d}' for x in sl):34s} {tot:5d}/{n:<8d} {side:<16d} {spans[a]:.3f} (n={len(sp)})".ljust(100) + f" {st.mean(idl) if idl else float('nan'):.1f}")
for g in "bk":
    v = [x for a, x in spans.items() if a.startswith(g)]
    if len(v) > 1: print(f"{g}: {len(v)} servers, span r0-r3 mean {st.mean(v):.3f} ms, sd {st.stdev(v):.3f}, range {min(v):.3f}-{max(v):.3f}")
