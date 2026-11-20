#!/usr/bin/env python3
"""memgrow: parse the '[memstat]' blocks of a serve log (libmemstat.c: glibc malloc_info() XML every N s and around one malloc_trim(0)).
usage: memstat.py <serve-log> <load-start-epoch> [pid]   (default: the pid with the most heaps)
One row per block: minutes since the load start, kind, arenas, arena address space / read-write part (resident upper bound), free chunks in the arenas
(fastbins + the rest), live bytes in the arenas (read-write part minus free), directly mmap'd chunks (count, MB). All MB."""
import re, sys, collections

log, t0 = sys.argv[1], int(sys.argv[2])
want = int(sys.argv[3]) if len(sys.argv) > 3 else None
text = open(log, errors="replace").read()
parts = re.split(r"(?m)^\[memstat\] ", text)[1:]
rows = collections.defaultdict(list)
last_t = {}
for part in parts:
    m = re.match(r"pid=(\d+) t=(\d+) (.*)", part)
    m2 = re.match(r"pid=(\d+) malloc_trim\(0\)=(\d+)", part)
    if m:
        pid, t, kind = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        last_t[pid] = t
    elif m2:
        pid, t, kind = int(m2.group(1)), last_t.get(int(m2.group(1)), 0), "after trim"
    else:
        continue
    heaps = re.findall(r"<heap nr=\"(\d+)\">(.*?)</heap>", part, re.S)
    if not heaps:
        continue
    tot = {"aspace": 0, "mprot": 0, "free": 0, "sys": 0}
    for _, body in heaps:
        for typ in ("fast", "rest"):
            mm = re.search(rf'<total type="{typ}" count="\d+" size="(\d+)"', body)
            tot["free"] += int(mm.group(1)) if mm else 0
        for key, pat in (("aspace", r'<aspace type="total" size="(\d+)"'), ("mprot", r'<aspace type="mprotect" size="(\d+)"'), ("sys", r'<system type="current" size="(\d+)"')):
            mm = re.search(pat, body)
            tot[key] += int(mm.group(1)) if mm else 0
    tail = part[part.rfind("</heap>"):]
    mm = re.search(r'<total type="mmap" count="(\d+)" size="(\d+)"', tail)
    mmap_n, mmap_b = (int(mm.group(1)), int(mm.group(2))) if mm else (0, 0)
    rows[pid].append((t, kind, len(heaps), tot, mmap_n, mmap_b))
if not rows:
    sys.exit("no malloc_info blocks")
pid = want or max(rows, key=lambda p: max(r[2] for r in rows[p]))
MB = 1 << 20
print(f"pid {pid}; {len(rows[pid])} blocks (other pids: {', '.join(str(p) for p in rows if p != pid) or 'none'})")
print("min\tkind\tarenas\taspace\trw_part\tfree\tlive(rw-free)\tmmap_n\tmmap_MB")
for t, kind, n, tot, mn, mb in rows[pid]:
    print(f"{(t - t0) / 60:.1f}\t{kind}\t{n}\t{tot['aspace'] / MB:.0f}\t{tot['mprot'] / MB:.0f}\t{tot['free'] / MB:.0f}\t{(tot['mprot'] - tot['free']) / MB:.0f}\t{mn}\t{mb / MB:.0f}")
