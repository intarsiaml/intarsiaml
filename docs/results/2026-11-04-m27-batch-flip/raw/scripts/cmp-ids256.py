#!/usr/bin/env python3
import json
d = json.load(open(".scratch/logs/gates/ids.json"))
keys = sorted({k.split("/")[1] for k in d}, key=int)
exposed = lambda p: 2 <= p % 256 <= 9 and p > 256 and p % 2048 >= 256
print("plen tail exposed | main solo==behind (first diff) | peel solo==behind | peel solo==main solo (first diff) | peel behind==main behind")
for k in keys:
    p = int(k); g = lambda t, m: d[f"{t}/{k}/{m}"]["ids"]
    fd = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"{p:5d} {p % 256:3d} {str(exposed(p)):5} | {str(g('main', 'solo') == g('main', 'behind')):5} {str(fd(g('main', 'solo'), g('main', 'behind'))):4} | {str(g('peel', 'solo') == g('peel', 'behind')):5} | {str(g('peel', 'solo') == g('main', 'solo')):5} {str(fd(g('peel', 'solo'), g('main', 'solo'))):4} | {g('peel', 'behind') == g('main', 'behind')}  ids {len(g('peel', 'solo'))}")
