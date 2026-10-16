#!/usr/bin/env python3
"""Cut a cb-mode trace into requests (t0/t1 from the cbgap-bench request log) and print busy / span / idle per request,
plus where the idle is: python3 reqgaps.py <name>.trace.jsonl <name>.req.jsonl"""
import json, sys
tr = [json.loads(l) for l in open(sys.argv[1])]
rq = [json.loads(l) for l in open(sys.argv[2])]
for q in rq:
    rs = [r for r in tr if r["s"] >= q["t0"] - 1e-3 and r["e"] <= q["t1"] + 1e-3]
    if not rs: print(q["rep"], "no records"); continue
    busy = sum(r["e"] - r["s"] for r in rs) * 1000
    span = (rs[-1]["e"] - rs[0]["s"]) * 1000
    gaps = [(rs[i]["s"] - rs[i - 1]["e"]) * 1000 for i in range(1, len(rs))]
    lead = (rs[0]["s"] - q["t0"]) * 1000   # request start to first GPU work
    tail = (q["t1"] - rs[-1]["e"]) * 1000  # last GPU work to request end
    big = sorted(((round(g, 1), i + 1) for i, g in enumerate(gaps) if g > 1.0), reverse=True)
    print(f"len {q['len']} rep {q['rep']}: wall {q['wall_ms']:.1f} prefill {q['prefill_ms']:.1f} | cbs {len(rs)} busy {busy:.1f} span {span:.1f} "
          f"idle-in-span {span - busy:.1f} (gaps>1ms {big[:6]}; sum of gaps<=1ms {sum(g for g in gaps if g <= 1.0):.1f}) lead {lead:.1f} tail {tail:.1f}")
    if "ho" in rs[0] and "-v" in sys.argv:
        for i, r in enumerate(rs):
            prev_e = rs[i - 1]["e"] if i else q["t0"]
            gap = (r["s"] - prev_e) * 1000
            if gap > 1.0 or i < 4:
                print(f"    cb {i:3d} t {r['t']:5d} nd {r['nd']:3d} gpu {(r['e'] - r['s']) * 1000:7.2f} ms | gap before {gap:7.2f} | encode started {(r['ho'] - prev_e) * 1000:8.2f} ms after prev GPU end, "
                      f"encode {(r['hc'] - r['ho']) * 1000:6.2f} ms, commit() {(r['hr'] - r['hc']) * 1000:5.2f} ms, committed {(r['s'] - r['hr']) * 1000:7.2f} ms before GPU start")
