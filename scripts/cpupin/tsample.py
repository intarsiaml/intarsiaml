#!/usr/bin/env python3
"""cpupin: every INTERVAL s (default 0.1) print `<unix time> <tid> <comm> <cpu> <cpu ticks in the interval>` for each thread of PID that used CPU in the interval
(/proc/PID/task/*/stat: comm, field 39 = the cpu the thread last ran on, utime+stime). A thread that the scheduler moves between the Cortex-X925 (cpu 5-9, 15-19)
and Cortex-A725 (0-4, 10-14) cores of a GB10 shows up as a change of the cpu column. usage: tsample.py PID [INTERVAL]"""
import os, sys, time
pid = int(sys.argv[1]); iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
prev = {}
def snap():
    out = {}
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            try:
                f = open(f"/proc/{pid}/task/{t}/stat").read()
            except FileNotFoundError:
                continue
            comm = f[f.index("(") + 1:f.rindex(")")]
            r = f[f.rindex(")") + 2:].split()
            out[t] = (comm, int(r[11]) + int(r[12]), int(r[36]))
    except (FileNotFoundError, ProcessLookupError):
        return None
    return out
while True:
    s = snap()
    if s is None:
        break
    now = time.time()
    for t, (comm, ticks, cpu) in s.items():
        d = ticks - prev.get(t, (0, ticks, 0))[1]
        if d > 0:
            print(f"{now:.3f} {t} {comm.replace(' ', '_')} {cpu} {d}", flush=True)
    prev = s
    time.sleep(iv)
