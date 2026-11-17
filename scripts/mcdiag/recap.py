import sqlite3, sys, re, collections
db, err = sys.argv[1], sys.argv[2]
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+) captures (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+)")
steps, chain = [], None
for l in open(err, errors="replace"):
    m = G.match(l)
    if m: steps.append((int(m.group(4)), int(m.group(3)), int(m.group(5))))
    m = C.match(l)
    if m and chain is None: chain = int(m.group(1))
con = sqlite3.connect(db)
launch = {r[0]: r[1] for r in con.execute("SELECT correlationId,start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%'")}
rows = con.execute("SELECT start,correlationId,graphNodeId,gridX,blockX,dynamicSharedMemory,registersPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
reps = collections.OrderedDict(); eager = []
for r in rows:
    if r[1] in launch: reps.setdefault(r[1], []).append(r)
    else: eager.append(r)
reps = list(reps.values())
off = len(reps) - len(steps)
pre, post = [], []
for i, s in enumerate(steps):
    (pre if chain and s[0] < chain else post).append(reps[i + off])
sig = lambda r: tuple(sorted((k[2], k[3], k[4], k[5], k[6]) for k in r))   # node id + launch config
print("chain at pos", chain, "| trace captures field first/last:", steps[0][2], steps[-1][2], "| graph launches:", len(launch), "replays:", len(reps))
sp, sq = collections.Counter(sig(r) for r in pre), collections.Counter(sig(r) for r in post)
print("distinct recordings (node ids + grid/block/dynsmem/regs): pre", len(sp), "post", len(sq), "| post recordings not seen pre:", len(set(sq) - set(sp)), "(counts pre", sorted(sp.values()), "post", sorted(sq.values()), ")")
t_chain = min((k[0] for k in eager if k[0] > reps[0][0][0]), default=None)
# eager kernels between replays: count per gap
gaps = collections.Counter()
ends = [(max(k[0] for k in r), r[0][1]) for r in reps]
starts = [min(k[0] for k in r) for r in reps]
big = []
for i in range(len(reps) - 1):
    n = sum(1 for k in eager if starts[i] <= k[0] < starts[i + 1])
    if n: big.append((i, n))
print("eager kernels in gaps between replays (replay index, count):", big[:12], "... total gaps with eager kernels", len(big))
