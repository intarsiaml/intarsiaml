#!/usr/bin/env python3
"""Lane mdraft: compare the drafting of the gate arms from their server logs (serve-ids-<arm>.log.gz, INTARSIA_SPEC_TRACE=1 INTARSIA_CB_TRACE=1).

usage: spec_cmp.py DIR REF_ARM ARM...
Every verify pass prints `spec: token N pos P drafted D accepted A emits E`. A sequence's (pos, token, D, A) series is a function of its own
drafts (drafts are the head's greedy chain, bitwise per sequence), so the multiset of these lines over a whole arm must equal the reference
arm's: the same drafts were proposed and accepted at the same positions, whichever iteration a request joined. Also counts the batched
drafts (`spec: mtp draft batch N sequences`). Exits 1 when an arm's multiset differs from the reference's."""
import collections, glob, gzip, os, re, sys

d, ref, *arms = sys.argv[1:]


def load(arm):
    for suffix in (".log.gz", ".log"):
        f = f"{d}/serve-ids-{arm}{suffix}"
        if os.path.exists(f):
            return (gzip.open(f, "rt", errors="replace") if suffix.endswith("gz") else open(f, errors="replace")).read().splitlines()
    sys.exit(f"no server log for {arm} in {d}")


pat = re.compile(r"spec: token (\d+) pos (\d+) drafted (\d+) accepted (\d+) emits (\d+)")
R = collections.Counter()
bad = 0
for arm in [ref] + arms:
    lines = load(arm)
    c = collections.Counter(m.groups() for m in map(pat.search, lines) if m)
    batches = [int(m.group(1)) for m in (re.search(r"spec: mtp draft batch (\d+) sequences", l) for l in lines) if m]
    acc = sum(int(k[3]) * v for k, v in c.items()); dr = sum(int(k[2]) * v for k, v in c.items())
    if arm == ref:
        R = c
    diff = (c - R) + (R - c)
    print(f"{arm}: {sum(c.values())} verify passes, drafted {dr}, accepted {acc}; batched drafts {len(batches)}; multiset differences vs {ref}: {sum(diff.values())}")
    if diff:
        bad += 1
        for k, v in sorted((c - R).items())[:5]:
            print(f"   only in {arm}: {k} x{v}")
        for k, v in sorted((R - c).items())[:5]:
            print(f"   only in {ref}: {k} x{v}")
sys.exit(1 if bad else 0)
