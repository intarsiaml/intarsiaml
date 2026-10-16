#!/usr/bin/env python3
"""Per request: did it decode through a drafting-off phase and then see drafting allowed again before it finished?
usage: transitions.py SERVER.log[.gz] START_MS END_MS   (epoch-ms window of the runs to include)"""
import gzip, re, sys
p, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
op = gzip.open if p.endswith(".gz") else open
adm, fin, off_dec, on_dec, phases = {}, {}, {}, {}, []
with op(p, "rt", errors="replace") as f:
    for ln in f:
        m = re.match(r"cb-trace t=(\d+) (\w+) \{ (.*) \}", ln)
        if not m or not (t0 <= int(m[1]) <= t1):
            continue
        kind, body = m[2], m[3]
        if kind == "Admitted":
            d = dict(re.findall(r"(\w+): (\d+)", body)); adm[int(d["id"])] = int(d["iter"])
        elif kind == "Finished":
            d = dict(re.findall(r"(\w+): (\d+)", body)); fin[int(d["id"])] = int(d["iter"])
        elif kind == "DecodePhase":
            d = dict(re.findall(r"(\w+): (\w+)", body)); phases.append((int(d["iter"]), d["spec_allowed"] == "true"))
        elif kind == "Decode":
            it = int(re.search(r"iter: (\d+)", body)[1]); ids = [int(x) for x in re.search(r"ids: \[([^\]]*)\]", body)[1].split(",") if x.strip()]
            allowed = body.rstrip().endswith("spec_allowed: true")
            for i in ids:
                (on_dec if allowed else off_dec).setdefault(i, []).append(it)
allowed_iters = sorted(i for i, a in phases if a)
done = [i for i in adm if i in fin]
through = []
for i in done:
    if i in off_dec:
        last_off = max(off_dec[i])
        if any(last_off < it <= fin[i] for it in allowed_iters):
            through.append(i)
print(f"requests admitted+finished in window: {len(done)}; decoded in a drafting-off phase: {sum(1 for i in done if i in off_dec)}; "
      f"of those, drafting allowed again before they finished (off -> on while alive): {len(through)}")
