#!/usr/bin/env python3
"""Sum the engine's per-request speculation lines (`dsv41: spec N tokens in R rounds (..), A/D drafts accepted`).

    specsum.py LINES.txt LABEL [name,name,...]

With names (the requests in order, e.g. the golden prompts), prints each request and sums per category: code (name
contains "code"), chat ("chat"), prose ("story", "science", "prose"), other.
"""
import re
import sys

pat = re.compile(r"spec (\d+) tokens in (\d+) rounds .*?, (\d+)/(\d+) drafts accepted")
rows = [tuple(int(x) for x in m.groups()) for m in (pat.search(l) for l in open(sys.argv[1])) if m]
label = sys.argv[2]
names = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []


def line(what, rs):
    t, r, a, d = (sum(x[i] for x in rs) for i in range(4))
    return f"spec {label} {what}: {len(rs)} requests, {t} tokens in {r} rounds = {t / max(r, 1):.3f} a round; drafts accepted {a}/{d} = {100 * a / max(d, 1):.1f} %"


def cat(n):
    if "code" in n:
        return "code"
    if "chat" in n:
        return "chat"
    if any(k in n for k in ("story", "science", "prose")):
        return "prose"
    return "other"


if names and len(names) == len(rows):
    by = {}
    for n, r in zip(names, rows):
        print(f"spec {label} {n}: {r[0]} tokens in {r[1]} rounds = {r[0] / max(r[1], 1):.2f} a round")
        by.setdefault(cat(n), []).append(r)
    for c in ("code", "prose", "chat", "other"):
        if c in by:
            print(line(c, by[c]))
elif names:
    print(f"spec {label}: {len(rows)} lines for {len(names)} names; per-request split skipped")
print(line("all", rows))
