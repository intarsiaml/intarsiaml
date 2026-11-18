#!/usr/bin/env python3
"""postchain2 step 3: compare nsys_an.py summaries of servers (MID1 graph replays).
usage: nsys_cmp.py DIR [TOP]   (reads DIR/<group><i>.json or .json.gz; group `b` = base, every other group (n = n20, e = one eager step, ...) against it)
Per variant (captured graph) and server: median span / busy of a launch (ms). Then per node (ordinal in capture order) the median duration per
server; effect of group G = mean over G's servers - mean over the base servers; nulls = differences of consecutive pairs of servers of the same
group (b2 - b1, n2 - n1, ...), same ordinal. Aggregated per kernel name (sum over that name's nodes of the median differences, us per launch)."""
import json, sys, glob, os, re, statistics as st, collections
D = sys.argv[1]; TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 15
import gzip
J = {os.path.basename(f).split(".")[0]: json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f)) for f in sorted(glob.glob(f"{D}/*.json") + glob.glob(f"{D}/*.json.gz"))}
grp = collections.defaultdict(list)
for k in J:
    grp[re.match(r"[a-z]+", k).group(0)].append(k)
B = grp["b"]; others = [g for g in grp if g != "b"]
med = st.median
common = set.intersection(*(set(v) for v in J.values()))
for var in sorted(common, key=lambda v: -J[B[0]][v]["n_all"]):
    names = J[B[0]][var]["names"]
    print(f"\n== {var}: {len(names)} nodes; launches used per server " + " ".join(f"{k}:{J[k][var]['n']}" for k in J))
    print("  median span ms: " + "  ".join(f"{k} {med(J[k][var]['span']):.4f}" for k in J))
    print("  median busy ms: " + "  ".join(f"{k} {med(J[k][var]['busy']):.4f}" for k in J))
    sp = {k: med(J[k][var]["span"]) for k in J}
    print("  span, group mean - base mean (ms): " + "  ".join(f"{g} {st.mean(sp[k] for k in grp[g]) - st.mean(sp[k] for k in B):+.4f}" for g in others)
          + " | nulls " + " ".join(f"{b}-{a} {sp[b] - sp[a]:+.4f}" for g in grp for a, b in zip(grp[g][0::2], grp[g][1::2])))
    md = {k: [med(x) for x in J[k][var]["dur"]] for k in J}
    nulls = [(f"{b}-{a}", [md[b][i] - md[a][i] for i in range(len(names))]) for g in grp for a, b in zip(grp[g][0::2], grp[g][1::2])]
    for g in others:
        eff = [st.mean(md[k][i] for k in grp[g]) - st.mean(md[b][i] for b in B) for i in range(len(names))]
        print(f"  [{g} - b] sum of per-node median differences, us per launch: {sum(eff):+.1f}; nulls " + " ".join(f"{l} {sum(x):+.1f}" for l, x in nulls))
        agg = collections.defaultdict(lambda: [0.0, 0, [0.0] * len(nulls), 0.0])
        for i, nm in enumerate(names):
            a = agg[nm]; a[0] += eff[i]; a[1] += 1; a[3] += st.mean(md[b][i] for b in B)
            for j, (_, x) in enumerate(nulls):
                a[2][j] += x[i]
        print(f"  [{g} - b] per kernel name (nodes per launch, base us per launch, effect us, nulls us), top {TOP} by |effect|:")
        for nm, (e, cnt, nl, base) in sorted(agg.items(), key=lambda x: -abs(x[1][0]))[:TOP]:
            print(f"    {nm[:60]:60s} x{cnt:3d} base {base:8.1f}  eff {e:+7.1f}  nulls " + " ".join(f"{x:+6.1f}" for x in nl))
