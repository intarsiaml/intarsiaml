#!/usr/bin/env python3
"""Lane PF27c: per-rank kernel table of a rocprofv3 kernel trace of one tp2 prefill (generate --max-tokens 1): the busiest burst per
agent (gaps > 50 ms split bursts), grouped by short kernel name + template args + grid, sorted by time.
usage: knames.py trace.csv[.gz] [--agent 'Agent 1'] [--top 40]"""
import argparse, collections, csv, gzip, re
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--agent", default=None); ap.add_argument("--top", type=int, default=40)
ap.add_argument("--nogrid", action="store_true")
a = ap.parse_args()
op = gzip.open if a.csv.endswith(".gz") else open
rows = collections.defaultdict(list)
with op(a.csv, "rt") as f:
    for r in csv.DictReader(f):
        rows[r["Agent_Id"]].append(r)
def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"\(.*$", "", n)  # drop args
    n = re.sub(r"^void ", "", n)
    return n[:110]
for ag in sorted(rows):
    if a.agent and ag != a.agent: continue
    ks = sorted(rows[ag], key=lambda r: int(r["Start_Timestamp"]))
    bursts, cur = [], [ks[0]]
    for p, r in zip(ks, ks[1:]):
        if int(r["Start_Timestamp"]) - int(p["End_Timestamp"]) > 50e6: bursts.append(cur); cur = []
        cur.append(r)
    bursts.append(cur)
    b = max(bursts, key=lambda bb: sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in bb))
    span = (int(b[-1]["End_Timestamp"]) - int(b[0]["Start_Timestamp"])) / 1e6
    g = collections.defaultdict(lambda: [0, 0.0])
    for r in b:
        k = short(r["Kernel_Name"])
        if not a.nogrid: k += f" g={r['Grid_Size_X']}x{r['Grid_Size_Y']} wg={r['Workgroup_Size_X']}"
        g[k][0] += 1; g[k][1] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
    tot = sum(v[1] for v in g.values())
    print(f"== {ag}: span {span:.1f} ms, kernel {tot:.1f} ms, idle {span - tot:.1f} ms")
    for k, (n, ms) in sorted(g.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"{ms:8.1f} ms {n:5d} x {1000 * ms / n:8.1f} us  {k}")
