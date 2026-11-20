#!/usr/bin/env python3
"""memgrow: categorise a process's resident anonymous memory from /proc/<pid>/smaps: [heap] (glibc main arena, brk), glibc thread-arena heaps
(64 MiB-aligned start, size <= 64 MiB, rw-p), other anonymous mappings >= 1 MiB (direct mmap: big malloc chunks, CUDA pinned host memory, stacks), and the rest.
usage: smaps_cat.py <pid>   prints one line: time, total anon MB, then MB (and count) per category."""
import sys, time
pid = sys.argv[1]
cat = {"heap": [0, 0], "arena": [0, 0], "big": [0, 0], "small": [0, 0], "file": [0, 0]}
cur = None
for line in open(f"/proc/{pid}/smaps"):
    p = line.split()
    if "-" in p[0] and len(p) >= 5 and not line.startswith(("Rss", "Pss", "Size")):
        lo, hi = [int(x, 16) for x in p[0].split("-")]
        name = p[5] if len(p) > 5 else ""
        cur = (lo, hi, p[1], name)
    elif line.startswith("Rss:") and cur:
        kb = int(p[1]); lo, hi, perm, name = cur
        if name and name != "[heap]":
            cat["file"][0] += kb; cat["file"][1] += 1
        elif name == "[heap]":
            cat["heap"][0] += kb; cat["heap"][1] += 1
        elif lo % (64 << 20) == 0 and (hi - lo) <= (64 << 20) and perm == "rw-p":
            cat["arena"][0] += kb; cat["arena"][1] += 1
        elif kb >= 1024:
            cat["big"][0] += kb; cat["big"][1] += 1
        else:
            cat["small"][0] += kb; cat["small"][1] += 1
anon = sum(v[0] for k, v in cat.items() if k != "file")
print(time.strftime("%H:%M:%S"), f"anon_mb={anon // 1024}", " ".join(f"{k}={v[0] // 1024}MB/{v[1]}" for k, v in cat.items()))
