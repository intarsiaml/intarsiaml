#!/usr/bin/env python3
"""Round 10: which dispatches run >= 1 ms, what the dispatch right after each one (+1) is, and what the gfx1201 stall
should cost there (kernels/hip_checks/gap_repro on tools/rocm-gap-repro: unprofiled, the +1 dispatch after a kernel of
>= ~1.05 ms cannot finish before ~W us after it starts, so it costs max(0, W - dur(+1)); +2 and later are unaffected).

  longnext.py <rocprofv3 kernel_trace.csv> [--min-ms 1.0] [--w-us 260] [--ms <unprofiled prefill/decode ms>] [--min-n 2] [--list]

--ms gives the denominator of the percentage (the unprofiled prefill or decode time; default the profiled span, which
includes model load). --min-n drops groups with fewer launches (default 2: one-off first-launch costs, rocclr fills).

Durations come from the profiler's own Start/End (kernel durations are right under rocprofv3; only the gap moves: under
the profiler the window shows as a ~277 us gap BEFORE +1, printed as `pgap`). Groups by (long kernel, +1 kernel).
"""
import csv
import re
import statistics as st
import sys


def short(name: str) -> str:
    n = name.replace("(anonymous namespace)::", "").replace("void ", "").replace("__amd_rocclr_", "rocclr:")
    n = re.sub(r"\(.*$", "", n)  # drop the argument list
    return n[:70]


def main() -> None:
    args = sys.argv[1:]
    path = args[0]
    min_ms = float(args[args.index("--min-ms") + 1]) if "--min-ms" in args else 1.0
    w_us = float(args[args.index("--w-us") + 1]) if "--w-us" in args else 260.0
    ref_ms = float(args[args.index("--ms") + 1]) if "--ms" in args else None
    min_n = int(args[args.index("--min-n") + 1]) if "--min-n" in args else 2
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    span_ms = (rows[-1][1] - rows[0][0]) / 1e6
    ktime_ms = sum(e - s for s, e, _ in rows) / 1e6
    groups: dict = {}
    listing = []
    for i, (s, e, n) in enumerate(rows[:-1]):
        d = (e - s) / 1e3
        if d < min_ms * 1e3:
            continue
        s1, e1, n1 = rows[i + 1]
        d1 = (e1 - s1) / 1e3
        pgap = (s1 - e) / 1e3
        cost = max(0.0, w_us - d1) if d >= 1050 else 0.0
        band = ">=1.05" if d >= 1050 else "1.00-1.05"
        key = (band, short(n), short(n1))
        g = groups.setdefault(key, [])
        g.append((d, d1, pgap, cost))
        listing.append((i, d, short(n), d1, short(n1), pgap, cost))
    print(f"# {path}\n# {len(rows)} dispatches, span {span_ms:.1f} ms, kernel time {ktime_ms:.1f} ms, W = {w_us:.0f} us")
    print(f"{'band':9} {'n':>4} {'long ms (med)':>13} {'+1 us (med)':>11} {'pgap us (med)':>13} {'cost ms':>8}  long -> +1")
    tot = 0.0
    dropped = 0
    for key, g in sorted(groups.items(), key=lambda kv: -sum(x[3] for x in kv[1])):
        if len(g) < min_n:
            dropped += len(g)
            continue
        c = sum(x[3] for x in g) / 1e3
        tot += c
        print(f"{key[0]:9} {len(g):4d} {st.median(x[0] for x in g) / 1e3:13.3f} {st.median(x[1] for x in g):11.1f} "
              f"{st.median(x[2] for x in g):13.1f} {c:8.2f}  {key[1]} -> {key[2]}")
    den, what = (ref_ms, "of the given unprofiled time") if ref_ms else (span_ms, "of the profiled span (includes load)")
    print(f"# {dropped} launches in groups of fewer than {min_n} not counted (one-off first launches, rocclr fills)")
    print(f"# predicted stall cost, unprofiled: {tot:.2f} ms = {100 * tot / den:.2f} % {what}")
    if "--list" in args:
        for row in listing:
            print("%6d %9.1f %-50s %9.1f %-40s pgap %7.1f cost %6.1f" % row)


if __name__ == "__main__":
    main()
