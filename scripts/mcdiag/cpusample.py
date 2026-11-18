#!/usr/bin/env python3
"""mcdiag: every INTERVAL seconds print `<unix time> <tid> <cpu core> <cpu-seconds in the interval>` of the busiest thread of PID (from /proc/PID/task/*/stat), so a
decode thread that moves between the Cortex-X925 (cpu 5-9, 15-19) and Cortex-A725 (0-4, 10-14) cores of a GB10 shows up. usage: cpusample.py PID [INTERVAL]"""
import os, sys, time
pid = int(sys.argv[1]); iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
hz = os.sysconf("SC_CLK_TCK"); prev = {}
def snap():
    out = {}
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            f = open(f"/proc/{pid}/task/{t}/stat").read()
            r = f[f.rindex(")") + 2:].split()
            out[t] = (int(r[11]) + int(r[12]), int(r[36]))
    except (FileNotFoundError, ProcessLookupError):
        return None
    return out
while True:
    s = snap()
    if s is None:
        break
    best = max(((v[0] - prev.get(t, (0, 0))[0], t, v[1]) for t, v in s.items()), default=None)
    if prev and best and best[0] > 0:
        print(f"{time.time():.3f} {best[1]} {best[2]} {best[0] / hz:.2f}", flush=True)
    prev = s
    time.sleep(iv)
