#!/usr/bin/env python3
"""Lane wg: diff the INTARSIA_HIP_TRACE kernel tables of two real_35b_batch_step_timing logs (knob-off vs option 1, graphs off, ROWS=1).
usage: chase_fit.py OFF_LOG OPT1_LOG        selftest: chase_fit.py --selftest
Table line:  name  launches  total_ms  avg_us  pct   (between the 'kernel ... launches' header and the 'hip trace gaps' lines).
Prints the trace totals (kernel GPU time, stream wall, idle between launches) and the per-kernel launch / total-time / avg differences."""
import re, sys

def parse(text):
    tot = re.search(r"hip trace: (\d+) batches, (\d+) launches, kernel GPU time ([\d.]+) ms, stream wall time ([\d.]+) ms, idle between launches ([\d.]+) ms", text)
    rows, on = {}, False
    for ln in text.splitlines():
        if re.match(r"kernel\s+launches\s+total ms", ln): on = True; continue
        if on:
            m = re.match(r"(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%", ln)
            if not m: on = False; continue
            rows[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    step = re.search(r"TIMING batch B= ?1: ([\d.]+) ms/step", text)
    return dict(tot=tuple(map(float, tot.groups())) if tot else None, rows=rows, step=float(step.group(1)) if step else None)

def report(a, b):
    print(f"B=1 step ms: off {a['step']} opt1 {b['step']}")
    for lab, i in (("launches", 1), ("kernel GPU ms", 2), ("stream wall ms", 3), ("idle between launches ms", 4)):
        print(f"  {lab:26} off {a['tot'][i]:12.1f}  opt1 {b['tot'][i]:12.1f}  delta {b['tot'][i] - a['tot'][i]:10.1f}")
    names = set(a["rows"]) | set(b["rows"])
    d = sorted(((b["rows"].get(n, (0, 0, 0))[1] - a["rows"].get(n, (0, 0, 0))[1], n) for n in names), key=lambda x: -abs(x[0]))
    print(f"\n{'kernel':44}{'d total ms':>11}{'launches off':>13}{'opt1':>7}{'avg us off':>11}{'opt1':>8}")
    for dt, n in d[:12]:
        ra = a["rows"].get(n, (0, 0, 0)); rb = b["rows"].get(n, (0, 0, 0))
        print(f"{n[:43]:44}{dt:11.2f}{ra[0]:13d}{rb[0]:7d}{ra[2]:11.1f}{rb[2]:8.1f}")

def selftest():
    t = "TIMING batch B= 1: 6.07 ms/step\nhip trace: 16 batches, 100 launches, kernel GPU time 10.0 ms, stream wall time 11.0 ms, idle between launches 1.0 ms (9% of wall)\nkernel launches total ms avg us % kern\nfoo 40 6.00 150.0 60.0%\nbar 60 4.00 66.7 40.0%\nhip trace gaps after launches of < 0.1 ms: n 1"
    p = parse(t)
    assert p["rows"] == {"foo": (40, 6.0, 150.0), "bar": (60, 4.0, 66.7)} and p["tot"][1] == 100 and p["step"] == 6.07
    print("selftest ok")

if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]: selftest()
    elif len(sys.argv) == 3: report(parse(open(sys.argv[1], errors="replace").read()), parse(open(sys.argv[2], errors="replace").read()))
    else: sys.exit(__doc__)
