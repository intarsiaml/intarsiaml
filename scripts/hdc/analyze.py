#!/usr/bin/env python3
"""Lane HDC: per-step attribution of an rocprofv3 --kernel-trace CSV of a tp=2 MTP-off decode (generate --spec off).

usage: analyze.py k_kernel_trace.csv [--anchor SUBSTR] [--first N] [--last M]

Steps are cut at the start of the anchor kernel (default: the bf16 logits GEMV, one per step per rank). Per rank and category it prints
median per-step microseconds and launches; `idle` is the step period minus the union of kernel intervals on that rank (so launch gaps,
waits on the other rank and host time between graph replays all land in `idle`).
"""
import csv, re, sys, statistics as st, collections

CATS = [  # first match wins; patterns on the demangled kernel name
    ("allreduce", r"ar2_kernel|allreduce|all_reduce|ar_norm|ar2_"),
    ("gemv-mxfp4", r"mxfp4_a8|mx_a8_quant|mxfp4_.*gemv|mxfp4_pair"),
    ("gemv-logits", r"gemv_kernel"),
    ("gdn", r"gdn|conv_delta"),
    ("attention", r"attn|attention|splitkv|flash|gate_split_qk|sigmoid_mul"),
    ("norm", r"rmsnorm|norm_kernel"),
    ("act/other", r"silu|glu|argmax|embedding|add|copy|fill|scale|cast"),
]

SHORT = re.compile(r'ik_bf16|HIP_vector_type<[^>]*>|const|\(anonymous namespace\)::')

def cat(name):
    for c, p in CATS:
        if re.search(p, name):
            return c
    return "other"

def main():
    a = sys.argv[1:]
    path = a[0]
    anchor = "gemv_kernel"
    lo, hi = 10, 10
    i = 1
    while i < len(a):
        if a[i] == "--anchor": anchor = a[i + 1]; i += 2
        elif a[i] == "--first": lo = int(a[i + 1]); i += 2
        elif a[i] == "--last": hi = int(a[i + 1]); i += 2
        else: i += 1
    rows = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    out = {}
    for ag, ks in sorted(rows.items()):
        ks.sort()
        idx = [j for j, k in enumerate(ks) if anchor in k[2]]
        if len(idx) < lo + hi + 5:
            print(f"{ag}: only {len(idx)} anchor kernels, skipped"); continue
        per = []
        for s in range(lo, len(idx) - hi - 1):
            part = ks[idx[s]:idx[s + 1]]
            period = ks[idx[s + 1]][0] - ks[idx[s]][0]
            # union of intervals
            busy = 0; cur_e = None; cur_s = None
            for (b, e, _) in part:
                if cur_e is None or b > cur_e:
                    if cur_e is not None: busy += cur_e - cur_s
                    cur_s, cur_e = b, e
                else:
                    cur_e = max(cur_e, e)
            if cur_e is not None: busy += cur_e - cur_s
            c = collections.defaultdict(lambda: [0, 0])
            for (b, e, n) in part:
                x = c[cat(n)]; x[0] += e - b; x[1] += 1
            per.append((period, busy, c, part))
        med = lambda xs: st.median(xs)
        period = med([p[0] for p in per]) / 1e3
        busy = med([p[1] for p in per]) / 1e3
        print(f"\n== {ag}: {len(per)} steps, median period {period:.0f} us, kernel-busy (union) {busy:.0f} us ({100*busy/period:.1f}%), idle {period-busy:.0f} us, launches/step {med([len(p[3]) for p in per]):.0f}")
        cats = sorted({k for p in per for k in p[2]})
        tot = 0
        for k in cats:
            t = med([p[2][k][0] if k in p[2] else 0 for p in per]) / 1e3
            n = med([p[2][k][1] if k in p[2] else 0 for p in per])
            tot += t
            print(f"   {k:14s} {t:8.0f} us  {100*t/period:5.1f}%  {n:5.0f} launches")
        print(f"   {'sum of kernels':14s} {tot:8.0f} us (overlap/sum check vs busy {busy:.0f})")
        # top kernels
        byn = collections.defaultdict(lambda: [0, 0])
        for p in per:
            for (b, e, n) in p[3]:
                x = byn[n]; x[0] += e - b; x[1] += 1
        print("   top kernels (us/step, calls/step, avg us):")
        for n, (t, c) in sorted(byn.items(), key=lambda x: -x[1][0])[:10]:
            short = SHORT.sub('', n)[:90]
            print(f"     {t/len(per)/1e3:8.0f}  {c/len(per):6.1f}  {t/c/1e3:8.1f}  {short}")
        out[ag] = (period, busy)
    return out

if __name__ == "__main__":
    main()
