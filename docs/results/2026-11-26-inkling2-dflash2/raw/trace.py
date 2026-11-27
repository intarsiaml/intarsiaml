#!/usr/bin/env python3
"""Lane ink-dflash2: INTARSIA_SPEC_TRACE=1 `generate --prompt-ids` stderr -> per prompt: verify passes, tokens per pass, accepted
histogram, draft ms and verify (+ rollback) ms per pass (by verify rows), and for a DFlash2 k7 trace the predicted tok/s at
k = 1..7 (the same block truncated: accepted min(a, k); draft ms unchanged; verify ms at k + 1 rows from the row table).
usage: trace.py DFL_TRACE.err [DSPARK_TRACE.err]   (the second gives the verify ms of 2..6 rows)"""
import re, sys
from collections import defaultdict

acc_re = re.compile(r"^spec: token \d+ pos \d+ drafted (\d+) accepted (\d+) emits (\d+)")
tim_re = re.compile(r"^spec: mtp draft ([\d.]+) ms, verify \+ rollback ([\d.]+) ms")
d2_re = re.compile(r"^spec: dflash2 draft flush ([\d.]+) block ([\d.]+) head ([\d.]+) selproj ([\d.]+) dl ([\d.]+) select ([\d.]+) total ([\d.]+) ms")
dec_re = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms \(([\d.]+) tok/s\)")


def parse(path):
    out, cur, pend = {}, [], None
    for line in open(path, errors="replace"):
        if m := acc_re.match(line):
            pend = (int(m.group(1)), int(m.group(2)))
        elif (m := tim_re.match(line)) and pend:
            cur.append((pend[0], pend[1], float(m.group(1)), float(m.group(2))))
            pend = None
        elif m := dec_re.match(line):
            out[m.group(1)] = (cur, float(m.group(4)))
            cur = []
    return out


def rows_table(*traces):
    t = defaultdict(list)
    for tr in traces:
        for passes, _ in tr.values():
            for d, a, dm, vm in passes:
                t[d + 1].append(vm)
    return {r: sum(v) / len(v) for r, v in t.items() if v}


dfl = parse(sys.argv[1])
dsp = parse(sys.argv[2]) if len(sys.argv) > 2 else {}
rt = rows_table(dfl, dsp)
print("verify + rollback ms by rows:", {r: round(v, 2) for r, v in sorted(rt.items())})
for name, tr in [("DFlash2", dfl), ("DSpark", dsp)]:
    if not tr:
        continue
    print(f"\n{name}: prompt | passes | tokens/pass | accepted hist 0..7 | draft ms | verify ms | traced tok/s")
    for p, (passes, rate) in tr.items():
        if not passes:
            print(f"  {p}: no passes")
            continue
        n = len(passes)
        tpp = sum(a + 1 for _, a, _, _ in passes) / n
        h = [sum(1 for _, a, _, _ in passes if a == i) for i in range(8)]
        dm = sum(x[2] for x in passes) / n
        vm = sum(x[3] for x in passes) / n
        print(f"  {p}: {n} | {tpp:.2f} | {h} | {dm:.2f} | {vm:.2f} | {rate:.1f}")
if dfl:
    print("\nDFlash2 predicted (traced draft ms + row-table verify ms) tok/s by k:")
    for p, (passes, _) in dfl.items():
        if not passes:
            continue
        cells = []
        for k in range(1, 8):
            vm = rt.get(k + 1)
            if vm is None:
                lo, hi = max((r for r in rt if r < k + 1), default=None), min((r for r in rt if r > k + 1), default=None)
                if lo is None or hi is None:
                    cells.append(f"k{k} -")
                    continue
                vm = rt[lo] + (rt[hi] - rt[lo]) * (k + 1 - lo) / (hi - lo)
            tok = sum(min(a, k) + 1 for _, a, _, _ in passes)
            ms = sum(dm + vm for _, _, dm, _ in passes)
            cells.append(f"k{k} {1000 * tok / ms:.1f}")
        print(f"  {p}: " + "  ".join(cells))
