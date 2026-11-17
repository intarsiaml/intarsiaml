#!/usr/bin/env python3
"""mcdiag: per-kernel time of captured-graph steps, difference-in-differences between a probe run (one chain pass) and a base run of the same
greedy request, from nsys sqlite exports (`nsys profile --cuda-graph-trace=node`) and the runs' INTARSIA_SPEC_TRACE logs.

A replay = the kernel nodes that share one correlation id of a graph launch. The Nth `spec: mtp graph step` trace line (it carries the position `len`
and the queued head rows `n`) is matched to the replay with the same index counted from the END of the run (the run may hold a replay or two before
the first traced step). Only replays of the dominant recording (most replays; same kernel-name sequence) are used, so every node index means the same
kernel. Steps at positions below the probe chain (minus MARGIN) are PRE, steps after it (plus MARGIN) are POST. For every node index:
  DiD = (probe POST - probe PRE) - (base POST - base PRE)     mean ns per node
which cancels the growth of the attention with the position (base has the same positions). Prints the span of the replays the same way, the
node durations summed by kernel name, and the largest single nodes.
usage: nsys.py --probe P.sqlite P.err --base B.sqlite B.err [--margin 15] [--top 15]"""
import sqlite3, re, sys, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("--probe", nargs=2, required=True)
ap.add_argument("--base", nargs=2, required=True)
ap.add_argument("--margin", type=int, default=15)
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--skip-pos", type=int, default=100)
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+)")


def trace(fn):
    steps, chain = [], None
    for l in open(fn, errors="replace"):
        m = G.match(l)
        if m:
            steps.append((int(m.group(4)), int(m.group(3)), float(m.group(1))))
        m = C.match(l)
        if m and chain is None:
            chain = int(m.group(1))
    return steps, chain


def replays(db):
    con = sqlite3.connect(db)
    launch = {cid for (cid, nm) in con.execute("SELECT correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId") if "GraphLaunch" in nm}
    cols = [r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    gcol = "k.graphId" if "graphId" in cols else "NULL"
    rows = con.execute(f"SELECT k.start,k.end,k.correlationId,{gcol},s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName ORDER BY k.start").fetchall()
    reps, cur = [], None
    for k in rows:
        if k[3] not in (None, 0) or k[2] in launch:
            if cur is not None and cur[0] == k[2]:
                cur[1].append(k)
            else:
                cur = [k[2], [k]]
                reps.append(cur)
        else:
            cur = None
    return [r[1] for r in reps]


def load(pair):
    steps, chain = trace(pair[1])
    reps = replays(pair[0])
    off = len(reps) - len(steps)
    return [(steps[i], reps[i + off]) for i in range(len(steps)) if i + off >= 0], chain, len(reps), len(steps)


P, chain, nrp, nsp = load(a.probe)
B, _, nrb, nsb = load(a.base)
print(f"probe: {nrp} replays, {nsp} traced steps; base: {nrb} replays, {nsb} traced steps; chain at pos {chain}")
sig = lambda r: tuple(k[4] for k in r)
dom = collections.Counter(sig(r) for _, r in P + B).most_common(1)[0][0]
nk = len(dom)
print("dominant recording:", nk, "nodes")


def sets(lst):
    pre, post = [], []
    for (pos, n, ms), r in lst:
        if pos < a.skip_pos or sig(r) != dom:
            continue
        if pos < chain - a.margin:
            pre.append(r)
        elif pos > chain + a.margin:
            post.append(r)
    return pre, post


pp, pq = sets(P)
bp, bq = sets(B)
print(f"replays used: probe pre {len(pp)} post {len(pq)}; base pre {len(bp)} post {len(bq)}")
dur = lambda r, j: r[j][1] - r[j][0]
mean = lambda x: sum(x) / len(x)
nd = lambda S, j: mean([dur(r, j) for r in S])
span = lambda r: r[-1][1] - r[0][0]
busy = lambda r: sum(k[1] - k[0] for k in r)
for name, f in (("span", span), ("kernel time sum", busy)):
    d_p = mean([f(r) for r in pq]) - mean([f(r) for r in pp])
    d_b = mean([f(r) for r in bq]) - mean([f(r) for r in bp])
    print(f"{name}: probe post-pre {d_p / 1e3:+.1f} us, base post-pre {d_b / 1e3:+.1f} us, DiD {(d_p - d_b) / 1e3:+.1f} us per replay")
did = []
for j in range(nk):
    did.append(((nd(pq, j) - nd(pp, j)) - (nd(bq, j) - nd(bp, j)), j))
byname = collections.defaultdict(float)
for d, j in did:
    byname[dom[j]] += d
print("DiD by kernel name (ns per replay, sum over nodes):")
for nm, d in sorted(byname.items(), key=lambda kv: -abs(kv[1]))[: a.top]:
    print(f"   {nm[:70]:70s} {d:+9.0f}  ({sum(1 for x in dom if x == nm)} nodes, base mean {sum(nd(bp, j) for j in range(nk) if dom[j] == nm) / 1e3:.1f} us per replay)")
print("largest single nodes:")
for d, j in sorted(did, key=lambda x: -abs(x[0]))[: a.top]:
    print(f"   node {j:4d} {dom[j][:60]:60s} DiD {d:+8.0f} ns (probe pre {nd(pp, j):.0f} post {nd(pq, j):.0f}; base pre {nd(bp, j):.0f} post {nd(bq, j):.0f})")
print("total DiD over nodes: %+.1f us" % (sum(d for d, _ in did) / 1e3))
