#!/usr/bin/env python3
"""Per-process accumulated GPU time (ns) from ioreg AGXDeviceUserClient AppUsage. Prints JSON {"pid name": ns}."""
import json, re, subprocess, sys
out = subprocess.run(["ioreg", "-l", "-w0", "-r", "-c", "AGXDeviceUserClient"], capture_output=True, text=True).stdout
tot = {}
for blk in out.split("+-o AGXDeviceUserClient")[1:]:
    m = re.search(r'"IOUserClientCreator" = "pid (\d+), (.*)"', blk)
    if not m:
        continue
    k = f"{m.group(1)} {m.group(2)}"
    tot[k] = tot.get(k, 0) + sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', blk))
if len(sys.argv) > 1:  # diff mode: gputime.py before.json after.json [exclude-pid]
    a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); ex = sys.argv[3] if len(sys.argv) > 3 else None
    d = {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0) > 0 and (ex is None or not k.startswith(ex + " "))}
    print(json.dumps({k: round(v / 1e6, 1) for k, v in sorted(d.items(), key=lambda x: -x[1])}))
else:
    print(json.dumps(tot))
