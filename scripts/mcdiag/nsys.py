#!/usr/bin/env python3
"""mcdiag: per-kernel time of captured-graph steps before and after a chain pass, from an nsys sqlite export of a `generate` run
(`nsys profile --cuda-graph-trace=node`). A replay = the kernel nodes sharing one correlation id of a graph launch; replays with the same node-name
sequence are one recording (n queued head rows). The chain pass = the first run of >= EAGER_MIN eager (non-graph) kernels between two replays
after the warm-up. Prints, per recording: replays before / after, mean span (first node start to last node end) and the sum of node durations,
then the kernels whose mean duration moved most, and the gaps between nodes. Also prints how the time between replays splits.
usage: nsys.py FILE.sqlite [--skip N] [--top N] [--csv OUT.csv]"""
import sqlite3, sys, argparse, collections, hashlib, csv

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--skip", type=int, default=6, help="replays dropped at the start (capture, warm-up)")
ap.add_argument("--top", type=int, default=14)
ap.add_argument("--csv")
ap.add_argument("--eager-min", type=int, default=300)
a = ap.parse_args()
con = sqlite3.connect(a.db)
cols = [r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
print("kernel table columns:", " ".join(cols))
gcol = "graphId" if "graphId" in cols else None
gn = "graphNodeId" if "graphNodeId" in cols else None
q = f"SELECT k.start,k.end,k.correlationId,{gcol or 'NULL'},{gn or 'NULL'},s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.sharedMemoryExecuted,k.dynamicSharedMemory,k.staticSharedMemory,k.registersPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName ORDER BY k.start"
rows = con.execute(q).fetchall()
print("kernels:", len(rows))
# runtime launches of graphs: correlation id -> launch time
launch = {}
for cid, st, nm in con.execute("SELECT correlationId,start,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId"):
    if "GraphLaunch" in nm:
        launch[cid] = st
print("graph launches:", len(launch))
# split into graph replays and eager runs, in time order
seq = []  # ("g", corr, [kernels]) or ("e", None, [kernels])
for k in rows:
    isg = (k[3] not in (None, 0)) or (k[2] in launch)
    if isg:
        if seq and seq[-1][0] == "g" and seq[-1][1] == k[2]:
            seq[-1][2].append(k)
        else:
            seq.append(["g", k[2], [k]])
    else:
        if seq and seq[-1][0] == "e":
            seq[-1][2].append(k)
        else:
            seq.append(["e", None, [k]])
reps = [(i, s) for i, s in enumerate(seq) if s[0] == "g"]
print("replays:", len(reps), "eager runs:", sum(1 for s in seq if s[0] == "e"), "eager sizes >=%d:" % a.eager_min, [len(s[2]) for s in seq if s[0] == "e" and len(s[2]) >= a.eager_min][:10])
# first chain pass after the warm-up
chain_at = None
for ri, (i, s) in enumerate(reps):
    if ri < a.skip:
        continue
    nxt = seq[i + 1] if i + 1 < len(seq) else None
    if nxt and nxt[0] == "e" and len(nxt[2]) >= a.eager_min:
        chain_at = ri
        break
print("first chain pass after replay #", chain_at, "(eager kernels:", len(seq[reps[chain_at][0] + 1][2]) if chain_at is not None else None, ")")
sig_of = lambda s: hashlib.md5("|".join(k[5] for k in s[2]).encode()).hexdigest()[:8]
groups = collections.defaultdict(lambda: ([], []))
for ri, (i, s) in enumerate(reps):
    if ri < a.skip:
        continue
    pre = chain_at is None or ri <= chain_at
    if not pre and ri <= chain_at + 3:
        continue  # the replays right after the chain pass (n, catch-up) are their own thing
    groups[sig_of(s)][0 if pre else 1].append(s)
out = []
for sig, (pre, post) in sorted(groups.items(), key=lambda kv: -len(kv[1][0]) - len(kv[1][1])):
    nk = len((pre or post)[0][2])
    if len(pre) < 3 or len(post) < 3:
        print(f"recording {sig}: {nk} nodes, replays pre {len(pre)} post {len(post)} (too few to compare)")
        continue
    span = lambda r: (r[2][-1][1] - r[2][0][0]) / 1e3
    busy = lambda r: sum(k[1] - k[0] for k in r[2]) / 1e3
    m = lambda f, x: sum(f(r) for r in x) / len(x)
    print(f"recording {sig}: {nk} nodes, replays pre {len(pre)} post {len(post)}: span {m(span, pre):.1f} -> {m(span, post):.1f} us ({m(span, post) - m(span, pre):+.1f}), kernel time sum {m(busy, pre):.1f} -> {m(busy, post):.1f} us ({m(busy, post) - m(busy, pre):+.1f}), gaps {m(span, pre) - m(busy, pre):.1f} -> {m(span, post) - m(busy, post):.1f}")
    delta = []
    for j in range(nk):
        dp = sum(r[2][j][1] - r[2][j][0] for r in pre) / len(pre)
        dq = sum(r[2][j][1] - r[2][j][0] for r in post) / len(post)
        delta.append((dq - dp, j, pre[0][2][j][5], dp, dq, pre[0][2][j][6:9], pre[0][2][j][9], post[0][2][j][11], pre[0][2][j][11], post[0][2][j][13], pre[0][2][j][13]))
        out.append((sig, j, pre[0][2][j][5], dp, dq, dq - dp))
    delta.sort(key=lambda d: -abs(d[0]))
    for d in delta[: a.top]:
        print(f"   node {d[1]:4d} {d[2][:60]:60s} {d[3]:9.0f} -> {d[4]:9.0f} ns ({d[0]:+7.0f}) grid {d[5]} blk {d[6]} dynsmem {d[8]}->{d[7]} regs {d[10]}->{d[9]}")
    # by kernel name
    byname = collections.defaultdict(lambda: [0, 0, 0])
    for d in delta:
        b = byname[d[2]]
        b[0] += d[3]; b[1] += d[4]; b[2] += 1
    tot = sorted(byname.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:8]
    print("   by kernel name (sum over nodes, per replay):")
    for nm, b in tot:
        print(f"      {nm[:60]:60s} x{b[2]:3d}  {b[0] / 1e3:8.1f} -> {b[1] / 1e3:8.1f} us ({(b[1] - b[0]) / 1e3:+7.1f})")
if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "node", "kernel", "pre_ns", "post_ns", "delta_ns"])
        w.writerows(out)
