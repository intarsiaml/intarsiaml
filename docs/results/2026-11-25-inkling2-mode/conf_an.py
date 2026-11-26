#!/usr/bin/env python3
"""ink-mode slot 2: cold (C) vs hot (H) processes, GPU high-power seconds, battery current. usage: conf_an.py RUN_DIR"""
import glob, json, re, statistics as st, sys
from datetime import datetime
d = sys.argv[1]
mm = []
for l in open(f"{d}/macmon.jsonl"):
    try:
        j = json.loads(l)
    except Exception:
        continue
    mm.append((datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp(), j["gpu_usage"][0], j["gpu_power"], j.get("sys_power", 0)))
bt = []
for l in open(f"{d}/batt.txt"):
    p = l.split()
    if len(p) == 2:
        v = int(p[1])
        bt.append((float(p[0]), v - (1 << 64) if v >= 1 << 63 else v))
dt = st.median(b[0] - a[0] for a, b in zip(mm, mm[1:]))
rows = []
for f in sorted(glob.glob(f"{d}/[0-9][0-9][0-9]-?.err")):
    tag = f[len(d) + 1:-4]
    t = open(f).read()
    p = dict(re.findall(r"^prefill\[(\d)\]: \d+ tokens, (\d+) ms", t, re.M))
    if "2" not in p:
        continue
    t0, t1 = map(float, open(f[:-4] + ".t").read().split())
    w = [x for x in mm if t0 <= x[0] <= t1]
    hi = [x for x in w if x[2] > 65]
    b_run = [a for ts, a in bt if t0 <= ts <= t1]
    b_hi = [a for ts, a in bt if any(abs(ts - x[0]) < 0.2 for x in hi)]
    b_pre = [a for ts, a in bt if t0 - 10 <= ts < t0]
    rows.append(dict(tag=tag, arm=tag[-1], p4=int(p["1"]), p16=int(p["2"]), boost=len(hi) * dt,
                     sys_hi=st.median([x[3] for x in hi]) if hi else None, amp_min=min(b_run) if b_run else None,
                     amp_hi=st.median(b_hi) if b_hi else None, amp_pre=st.median(b_pre) if b_pre else None))
print("tag\tp4k\tp16k\tboost_s\tsysW_hi\tamp_min\tamp_hi\tamp_pre_idle")
for r in rows:
    print("\t".join(str(round(v, 1)) if isinstance(v, float) else str(v) for v in (r["tag"], r["p4"], r["p16"], r["boost"], r["sys_hi"], r["amp_min"], r["amp_hi"], r["amp_pre"])))
neg = [(ts, a) for ts, a in bt if a < 0]
print(f"battery samples {len(bt)}, distinct readings {len(set(a for _, a in bt))} (ioreg updates about once a minute), negative (discharging) readings: "
      + ", ".join(sorted(set(str(a) for _, a in neg))) + f" ({len(neg)} samples)")
for arm in "CH":
    rs = [r for r in rows if r["arm"] == arm and not r["tag"].startswith("warm")]
    if not rs:
        continue
    print(f"arm {arm}: n {len(rs)}, 4k median {st.median(r['p4'] for r in rs)}, 16k median {st.median(r['p16'] for r in rs)} "
          f"[{min(r['p16'] for r in rs)}-{max(r['p16'] for r in rs)}], 16k < 6150: {sum(r['p16'] < 6150 for r in rs)}, "
          f"boost s median {st.median(r['boost'] for r in rs):.2f}, battery min mA median {st.median([r['amp_min'] for r in rs if r['amp_min'] is not None]) if rs else '-'}, "
          f"pre-idle mA median {st.median([r['amp_pre'] for r in rs if r['amp_pre'] is not None])}")
