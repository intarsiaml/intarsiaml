import json, sys, datetime, statistics as st, os
R = sys.argv[1]
mm = []
for l in open(f"{R}/macmon.jsonl"):
    try:
        j = json.loads(l); mm.append((datetime.datetime.fromisoformat(j["timestamp"]).timestamp(), j["gpu_usage"][0], j["gpu_usage"][1], j["gpu_power"], j["temp"]["gpu_temp_avg"]))
    except Exception: pass
for l in open(f"{R}/runs.tsv"):
    a, b, stem, res = l.rstrip("\n").split("\t")
    f = f"{R}/{stem}.jsonl"
    if not os.path.exists(f): continue
    rs = [json.loads(x) for x in open(f)]
    pre = []
    for r in rs:
        if "gemv" in r["op"]: break
        pre.append(r)
    gaps = sum(max(0.0, b2["s"] - a2["e"]) for a2, b2 in zip(pre, pre[1:]))
    busy = sum(r["e"] - r["s"] for r in pre)
    # skip the first-chunk residency gaps: busy time only
    w = [x for x in mm if pre and pre[0]["s"] - 0 <= x[0] and float(a) <= x[0] <= float(b) and x[2] > 0.5]
    med = st.median(x[1] for x in w) if w else 0
    print(f"{stem:18} {res:32} prefill GPU busy {busy*1e3:7.1f} ms  MHz {med:4.0f}  busy*MHz/1617 {busy*1e3*med/1617:7.1f}  gaps {gaps*1e3:6.1f}  temp {max((x[4] for x in w), default=0):.0f}")
