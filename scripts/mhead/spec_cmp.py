#!/usr/bin/env python3
"""Lane mhead: compare the drafting of the gate arms from their server logs (serve-ids-<arm>.log.gz, INTARSIA_SPEC_TRACE=1 INTARSIA_CB_TRACE=1).

usage: spec_cmp.py DIR REF_ARM ARM...
Every verify pass prints `spec: token N pos P drafted D accepted A emits E`. A sequence's (pos, token, D, A) series is a function of its own
drafts (drafts are the head's greedy chain, bitwise per sequence), so the multiset of these lines over a whole arm must equal the reference
arm's: the same drafts were proposed and accepted at the same positions, whichever iteration a request joined. Lane mhead: the lines carry the drafted ids too, and arms are joined on (token, pos) (a mismatch there fails). Also counts the batched
drafts (`spec: mtp draft batch N sequences`). Exits 1 when an arm's multiset differs from the reference's."""
import collections, glob, gzip, os, re, sys

d, ref, *arms = sys.argv[1:]


def load(arm):
    for suffix in (".log.gz", ".log"):
        f = f"{d}/serve-ids-{arm}{suffix}"
        if os.path.exists(f):
            return (gzip.open(f, "rt", errors="replace") if suffix.endswith("gz") else open(f, errors="replace")).read().splitlines()
    sys.exit(f"no server log for {arm} in {d}")


# `drafts [..]` (the drafted ids) is printed since lane mhead; older logs lack it
pat = re.compile(r"spec: token (\d+) pos (\d+) drafted (\d+) accepted (\d+) emits (\d+)(?: drafts \[([\d, ]*)\])?")
R = collections.Counter()
RJ = {}
bad = 0
for arm in [ref] + arms:
    lines = load(arm)
    c = collections.Counter(m.groups() for m in map(pat.search, lines) if m)
    batches = [int(m.group(1)) for m in (re.search(r"spec: mtp draft batch (\d+) sequences", l) for l in lines) if m]
    acc = sum(int(k[3]) * v for k, v in c.items()); dr = sum(int(k[2]) * v for k, v in c.items())
    # join on (token, pos): where both arms ran a pass at that point of a request, the drafted ids and the accepted count must be equal
    # (arrival timing moves which passes exist, never what a pass at a given point drafts)
    j = collections.defaultdict(collections.Counter)
    for k, v in c.items():
        j[k[:2]][(k[2], k[3], k[5])] += v
    if arm == ref:
        R, RJ = c, j
    both = [k for k in j if k in RJ]
    jbad = [k for k in both if j[k] != RJ[k]]
    print(f"   {arm}: joined on (token, pos): {len(both)} points in both arms, {len(jbad)} with different drafted ids / accepted; {len(set(j) - set(RJ))} only in {arm}, {len(set(RJ) - set(j))} only in {ref}")
    for k in sorted(jbad)[:5]:
        print(f"   JOIN MISMATCH {k}: {arm} {dict(j[k])} vs {ref} {dict(RJ[k])}")
    bad += bool(jbad)
    diff = (c - R) + (R - c)
    print(f"{arm}: {sum(c.values())} verify passes, drafted {dr}, accepted {acc}; batched drafts {len(batches)}; multiset differences vs {ref}: {sum(diff.values())}")
    if diff:
        bad += 1
        for k, v in sorted((c - R).items())[:5]:
            print(f"   only in {arm}: {k} x{v}")
        for k, v in sorted((R - c).items())[:5]:
            print(f"   only in {ref}: {k} x{v}")
sys.exit(1 if bad else 0)
