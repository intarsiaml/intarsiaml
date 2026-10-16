import json,sys,glob,statistics as st
from datetime import datetime
d=sys.argv[1]
for seg in "ABCD":
    for f in sorted(glob.glob(f"{d}/{seg}-*.macmon.jsonl")):
        rows=[json.loads(l) for l in open(f) if l.strip()]
        t0=datetime.fromisoformat(rows[0]["timestamp"])
        xs=[r for r in rows if (datetime.fromisoformat(r["timestamp"])-t0).total_seconds()>8]
        g=[r["gpu_power"]*1000 for r in xs]; a=[r["all_power"]*1000 for r in xs]; u=[r["gpu_usage"][1]*100 for r in xs]
        print(f"{f.split('/')[-1]:28s} n={len(xs)} gpu_power mW mean {st.mean(g):.1f} median {st.median(g):.1f} max {max(g):.0f}  all_power mean {st.mean(a):.1f}  gpu_usage% mean {st.mean(u):.2f}")
