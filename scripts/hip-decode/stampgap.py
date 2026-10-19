#!/usr/bin/env python3
"""Round 9: is the ~275 us stall after a >= 1 ms dispatch there without rocprofv3?

stampgap.py <unprofiled stamps.txt> <profiled stamps.txt> <profiled run_kernel_trace.csv>

INTARSIA_HIP_STAMP (kernels/hip/ik_runtime.hip) brackets every long prefill dispatch (MoE gate_up / down and its pieces, the
GDN delta rule and its pieces, flash attention) with 1-thread wall-clock stamp kernels. Between two consecutive stamps of
one op runs exactly one dispatch, so

    interval = stamp(after) - stamp(before) = dispatch duration + the dead time on both sides of it.

The profiled run's kernel trace gives each bracketed dispatch's duration (and shows the stamp kernels around it); the
unprofiled run has the same stamps in the same order. dead time = interval - duration, in both runs, bucketed by the
dispatch's duration (< 1 ms, >= 1 ms). If rocprofv3 alone causes the stall, the >= 1 ms bucket's dead time is ~275 us
more than the < 1 ms bucket's in the profiled run and about the same in the unprofiled run.
"""
import csv
import statistics
import sys


def stamps(path):
    with open(path) as f:
        head = f.readline().split()
        khz = float(head[2])
        rows = [l.split() for l in f if l.strip()]
    return [(int(r[0]) / (khz / 1000.0), r[1], int(r[2])) for r in rows]  # (us, tag, piece)


def intervals(st):
    """(index, tag, piece, interval us) for every stamp that closes a dispatch (piece >= 0)"""
    out = []
    for i in range(1, len(st)):
        if st[i][2] >= 0 and st[i - 1][1] == st[i][1] and st[i - 1][2] == st[i][2] - 1:
            out.append((i, st[i][1], st[i][2], st[i][0] - st[i - 1][0]))
    return out


def durations(csv_path):
    """durations (us) of the dispatches between consecutive ik_stamp_kernel launches, in order"""
    K = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(csv_path)))
    out, prev, between = [], None, []
    for s, e, n in K:
        if "ik_stamp_kernel" in n:
            if prev is not None:
                out.append(between)
            prev, between = (s, e), []
        else:
            between.append((s, e, n))
    return out  # out[j] = dispatches between stamp j and stamp j+1


def main():
    un, pr, tr = stamps(sys.argv[1]), stamps(sys.argv[2]), durations(sys.argv[3])
    iu, ip = intervals(un), intervals(pr)
    if len(un) != len(pr):
        print(f"WARNING: {len(un)} unprofiled stamps vs {len(pr)} profiled")
    rows = []
    for (i, tag, piece, a), (j, tag2, piece2, b) in zip(iu, ip):
        if tag != tag2 or piece != piece2:
            print(f"order mismatch at {i}: {tag}/{piece} vs {tag2}/{piece2}")
            break
        between = tr[j - 1] if j - 1 < len(tr) else []
        if len(between) != 1:
            continue
        s, e, name = between[0]
        rows.append((tag, (e - s) / 1e3, a, b))
    print(f"{len(rows)} bracketed dispatches matched ({len(iu)} unprofiled / {len(ip)} profiled intervals)")
    print(f"{'bucket':<12} {'tag':<10} {'n':>4} {'dur us':>9} {'dead unprof us (median, p10..p90)':>36} {'dead prof us (median, p10..p90)':>34}")

    def q(v, f):
        v = sorted(v)
        return v[int((len(v) - 1) * f)]

    for lo, hi, lab in ((0, 1000, "< 1 ms"), (1000, 1e9, ">= 1 ms")):
        for tag in sorted({r[0] for r in rows}) + ["all"]:
            sel = [r for r in rows if lo <= r[1] < hi and (tag == "all" or r[0] == tag)]
            if not sel:
                continue
            du = [r[2] - r[1] for r in sel]
            dp = [r[3] - r[1] for r in sel]
            print(f"{lab:<12} {tag:<10} {len(sel):>4} {statistics.median(r[1] for r in sel):>9.1f} "
                  f"{statistics.median(du):>12.1f} ({q(du, .1):>7.1f} .. {q(du, .9):>7.1f}) {statistics.median(dp):>12.1f} ({q(dp, .1):>7.1f} .. {q(dp, .9):>7.1f})")


if __name__ == "__main__":
    main()
