#!/usr/bin/env python3
"""Lane det: compare the runs of `intarsia generate --repeat N --show-ids [--trace-margins]` (stderr `<tag>.err`) and the
layer hashes of INTARSIA_DET_TAP (`<tag>.tap`) in one or more raw directories.

    tools/det/detcmp.py docs/results/2026-10-28-det-tp2-nondeterminism/raw/w1 [more dirs]

Per arm (file stem): the distinct id lists and where each run first parts from the reference (the first run of the first
file of the arm's group); with margins, the steps whose logits row hash differs and the margin at the first token that
flipped; with a tap, rank 0 against rank 1 inside every pass (they must be equal) and run against run (the first pass and
layer whose hash differs). Arms named `<x>1`, `<x>2`, ... are one group (separate processes, same command).
"""
import collections
import pathlib
import re
import sys


def runs_from_err(p):
    ids, mar = {}, {}
    for line in p.read_text(errors="replace").splitlines():
        m = re.match(r"^ids(?:\[(\d+)\])?: \[(.*)\]$", line)
        if m:
            ids[int(m.group(1) or 0)] = [int(x) for x in m.group(2).split(",") if x.strip()]
        m = re.match(r"^margins(?:\[(\d+)\])?: (.*)$", line)
        if m:
            mar[int(m.group(1) or 0)] = [tuple(x.split(":")) for x in m.group(2).split()]
    return [ids[k] for k in sorted(ids)], [mar[k] for k in sorted(mar)]


def runs_from_tap(p):
    """[run][pass] -> {rank: [hash per layer]}, (pos, t); a run starts at each pass with pos=0."""
    runs = []
    for line in p.read_text().splitlines():
        f = line.split()
        pos, t, rank = (int(f[i].split("=")[1]) for i in range(3))
        if pos == 0 and rank == 0:
            runs.append([])
        if rank == 0:
            runs[-1].append(((pos, t), {}))
        runs[-1][-1][1][rank] = f[3:]
    return runs


def first_diff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


def main(dirs):
    groups = collections.OrderedDict()
    for d in map(pathlib.Path, dirs):
        for err in sorted(d.glob("*.err")):
            g = re.sub(r"\d+$", "", err.stem) if re.search(r"(tap|ids)\d+$", err.stem) else err.stem
            groups.setdefault(g, []).append(err)
    for g, errs in groups.items():
        allids, allmar, alltap, names = [], [], [], []
        for err in errs:
            ids, mar = runs_from_err(err)
            tap = err.with_suffix(".tap")
            taps = runs_from_tap(tap) if tap.exists() else []
            for i, r in enumerate(ids):
                allids.append(r)
                allmar.append(mar[i] if i < len(mar) else None)
                alltap.append(taps[i] if i < len(taps) else None)
                names.append(f"{err.stem}[{i}]")
        if not allids:
            continue
        ref = allids[0]
        distinct = collections.Counter(tuple(r) for r in allids)
        print(f"== {g}: {len(allids)} runs, {len(distinct)} distinct id lists {sorted(distinct.values(), reverse=True)}")
        for n, r, m in zip(names, allids, allmar):
            d = first_diff(ref, r)
            line = f"  {n}: ids {'same' if d is None else f'part at token {d} ({ref[d] if d < len(ref) else None} -> {r[d] if d < len(r) else None})'}"
            if m and allmar[0]:
                hd = [i for i in range(min(len(m), len(allmar[0]))) if m[i][2] != allmar[0][i][2]]
                line += f"; logits rows differing {len(hd)}/{len(m)}"
                if hd:
                    line += f" (first step {hd[0]})"
                if d is not None and d < len(allmar[0]):
                    line += f"; ref margin at the flip {allmar[0][d][1]}"
            print(line)
        # tap: rank 0 vs rank 1 inside each pass, and run vs run (rank 0)
        taps = [t for t in alltap if t]
        if not taps:
            continue
        mism, total = [], 0
        for n, t in zip(names, alltap):
            for (pt, by) in t or []:
                if 1 in by:
                    total += 1
                    li = first_diff(by[0], by[1])
                    if li is not None:
                        mism.append((n, pt, li))
        print(f"  tap rank0 vs rank1: {len(mism)} of {total} passes differ" + (f"; first {mism[:5]}" if mism else ""))
        t0 = alltap[0]
        for n, t in zip(names[1:], alltap[1:]):
            if not t or not t0:
                continue
            fd = None
            for k, ((pt, by), (_, by0)) in enumerate(zip(t, t0)):
                li = first_diff(by0[0], by[0])
                if li is not None:
                    fd = (k, pt, li)
                    break
            nd = sum(1 for (_, by), (_, by0) in zip(t, t0) if by[0] != by0[0])
            print(f"  tap {n} vs {names[0]}: " + ("all passes equal" if fd is None else f"{nd} passes differ, first pass #{fd[0]} (pos,t)={fd[1]} at layer {fd[2]}"))


if __name__ == "__main__":
    main(sys.argv[1:])
