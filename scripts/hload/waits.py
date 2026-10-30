#!/usr/bin/env python3
"""Lane hload: load-wait audit of a gfx12 assembly file (hipcc -S, scripts/hload/isa.sh).

For every kernel (or the ones matching --match) prints, in program order, the vector-memory loads and the
`s_wait_loadcnt N` waits, with a straight-line count of loads issued since the last wait ("pending"), the VALU work
between consecutive waits, and the loop structure (a backward branch closes a loop). A wait `0x0` with many loads
pending right before a run of math (fma / dot2 / cvt) is the load-then-wait serialisation this lane looks for.

  waits.py file.s [--match REGEX] [--summary] [--drain N] [--trace]

A "serial" wait retires the last few loads in flight (wait <= 1, nothing left to overlap) before math: the per-use load pattern
(load, wait, compute, next load). A "drain" is a wait that retires >= N loads at once (pending - N, straight-line count) and is followed by math before the next
wait: the compute waits for a whole window. Inside a loop it repeats per iteration.

The pending count is a straight-line model: it ignores control flow (loads under an exec branch are counted as issued),
so it is an upper bound inside branchy code; the trace shows the branches.
"""
import argparse
import re
import sys

LOAD = re.compile(r"^\s+(global_load|buffer_load|scratch_load|flat_load)\w*")
WAIT = re.compile(r"^\s+s_wait_loadcnt\s+(0x[0-9a-fA-F]+|\d+)")
MATH = re.compile(r"^\s+v_(fma|fmac|dot2|pk_fma|cvt|mul|add|perm|wmma|dot4)\w*")
VALU = re.compile(r"^\s+v_\w+")
LABEL = re.compile(r"^(\.?[\w.$]+):")
BRANCH = re.compile(r"^\s+s_(cbranch_\w+|branch)\s+(\S+)")


def kernels(lines):
    name, body = None, []
    for ln in lines:
        m = re.match(r"^([A-Za-z_][\w.$]*):\s*(;.*)?$", ln)
        if m and not ln.startswith(".") and name is None:
            name, body = m.group(1), []
            continue
        if name is not None:
            if ln.startswith(".Lfunc_end"):
                yield name, body
                name = None
            else:
                body.append(ln)


def analyse(name, body, trace):
    labels = {}
    for i, ln in enumerate(body):
        m = LABEL.match(ln)
        if m:
            labels[m.group(1)] = i
    loops = []  # (start, end) of backward branches
    for i, ln in enumerate(body):
        m = BRANCH.match(ln)
        if m and m.group(2) in labels and labels[m.group(2)] < i:
            loops.append((labels[m.group(2)], i))
    pending, nload, events = 0, 0, []
    valu = math = 0
    for i, ln in enumerate(body):
        if LOAD.match(ln):
            pending += 1
            nload += 1
        elif WAIT.match(ln):
            n = int(WAIT.match(ln).group(1), 0)
            inloop = any(s <= i <= e for s, e in loops)
            events.append((i, n, pending, valu, math, inloop))
            pending = min(pending, n)
            valu = math = 0
        elif VALU.match(ln):
            valu += 1
            if MATH.match(ln):
                math += 1
        if trace:
            if LOAD.match(ln) or WAIT.match(ln) or BRANCH.match(ln) or LABEL.match(ln):
                print(f"   {i:6d} {ln.strip()[:100]}")
    events.append((len(body), -1, pending, valu, math, False))  # sentinel: the work after the last wait
    return nload, loops, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asm")
    ap.add_argument("--match", default=".")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--summary", action="store_true", help="one line per kernel")
    ap.add_argument("--drain", type=int, default=4, help="loads waited for at once that count as a drain")
    a = ap.parse_args()
    lines = open(a.asm).read().splitlines()
    for name, body in kernels(lines):
        if not re.search(a.match, name):
            continue
        print(f"== {name}  ({len(body)} lines)")
        nload, loops, ev = analyse(name, body, a.trace)
        z = [e for e in ev[:-1] if e[1] == 0]
        drains = [k for k in range(len(ev) - 1) if ev[k][2] - ev[k][1] >= a.drain and ev[k + 1][4] > 0]
        serial = [k for k in range(len(ev) - 1) if 1 <= ev[k][2] - ev[k][1] < a.drain and ev[k][1] <= 1 and ev[k + 1][4] > 0]
        print(f"   loads {nload}, waits {len(ev) - 1}, wait 0x0: {len(z)}, drains (>= {a.drain} loads waited at once, math follows): "
              f"{len(drains)}, serial (the last 1..{a.drain - 1} loads in flight waited, math follows): {len(serial)}, loops {len(loops)}")
        if a.summary:
            continue
        for k, (i, n, p, v, m, inl) in enumerate(ev[:-1]):
            # the work between this wait and the next one is counted at the next event: printed here as "then"
            nv = ev[k + 1][3] if k + 1 < len(ev) else None
            nm = ev[k + 1][4] if k + 1 < len(ev) else None
            flag = " <== drain" if p - n >= a.drain and (nm or 0) > 0 else ""
            print(f"   @{i:6d} wait {n:2d} pending {p:3d} {'loop' if inl else '    '} then valu {nv} math {nm}{flag}")


if __name__ == "__main__":
    sys.exit(main())
