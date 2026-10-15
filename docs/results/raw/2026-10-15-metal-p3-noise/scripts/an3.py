import sys, json, re, glob, statistics as st, collections
from datetime import datetime
D = sys.argv[1]
mm = []
for l in open(f"{D}/macmon.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_power"]))
mm.sort()
rows = []
for f in sorted(glob.glob(f"{D}/p*-server.err")):
    label = re.search(r"(p\d+-c\d+)-server", f)[1]; chunk = int(label.split("-c")[1])
    for l in open(f):
        m = re.match(r"([\d.]+) \[prefill-trace\] plen (\d+) pos0 (\d+) wall (\d+) ms", l)
        if not m: continue
        t, plen, pos0, wall = float(m[1]), int(m[2]), int(m[3]), int(m[4])
        n = plen - pos0
        if n < 3900 or n > 4300: continue
        # last chunk size
        k = (n - 1) // chunk   # loop chunks
        last = n - k * chunk
        s = [x for x in mm if t - wall/1000 < x[0] <= t + 0.1]
        mhz = st.mean(x[1] for x in s)
        rows.append(dict(chunk=chunk, n=n, last=last, wall=wall, mhz=mhz, norm=wall*mhz/1600, label=label))
def bucket(r):
    l = r["last"]
    if r["chunk"] == 256:
        return "last<16" if l < 16 else ("last16-63" if l < 64 else ("last64-199" if l < 200 else "last>=200"))
    return "last<16" if l < 16 else ("last16-63" if l < 64 else ("last64-1999" if l < 2000 else "last>=2000"))
g = collections.defaultdict(list)
for r in rows: g[(r["chunk"], bucket(r))].append(r)
print("chunk bucket  n  | wall ms median | clock-normalised ms (wall*MHz/1600) median, min, max | MHz median")
for k in sorted(g):
    rs = g[k]
    print(k, len(rs), f"{st.median(r['wall'] for r in rs):.0f}", f"norm {st.median(r['norm'] for r in rs):.0f} ({min(r['norm'] for r in rs):.0f}-{max(r['norm'] for r in rs):.0f})", f"{st.median(r['mhz'] for r in rs):.0f}")
print()
print("per request, sorted by chunk, last:")
for r in sorted(rows, key=lambda r: (r["chunk"], r["last"])): print(r["chunk"], r["n"], "last", r["last"], r["wall"], f"{r['mhz']:.0f}MHz", f"norm {r['norm']:.0f}")
