#!/usr/bin/env python3
"""Summarise p2prime-step0.sh `run` output (docs/results/2026-10-19-p2prime-step0.md): per binary, the runs, which tests
failed, and per stress test the launches that differ and the NaN-carrying ones, summed over runs.
usage: p2s0_summary.py <window log> [<window log> ...]"""
import re
import sys
from collections import defaultdict

runs = defaultdict(int)
res = defaultdict(lambda: defaultdict(int))  # name -> key -> count
fails = defaultdict(lambda: defaultdict(int))  # name -> test -> runs failed
cur = None
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        line = re.sub(r"^\[gpu1\] ", "", line.rstrip("\n"))
        m = re.match(r"--- (\S+) run (\d+) \[[^\]]*\] GPU (\d)", line)
        if m:
            cur = f"{m.group(1)}"
            runs[cur] += 1
            res[cur][f"gpu{m.group(3)} runs"] += 1
            continue
        if cur is None:
            continue
        m = re.match(r"test (\S+) \.\.\. FAILED", line)
        if m:
            fails[cur][m.group(1)] += 1
        m = re.search(r"(v3_sentinel_stress(?:_narrow)?)(?: \[([^\]]+)\])?: (\d+) launches, (\d+) differ", line)
        if m:
            key = m.group(1) + (f" [{m.group(2)}]" if m.group(2) else "")
            res[cur][key + " launches"] += int(m.group(3))
            res[cur][key + " differ"] += int(m.group(4))
        m = re.search(r"  \[([^\]]+)\] (t=\S+ e=\S+ k=\S+ h=\S+ i=\S+), \d+ rounds: (.*); stream 1: (\d+) differ \((\d+) NaN\)", line)
        if m:
            for sk, d, n in re.findall(r"skew (\d+): (\d+) differ \((\d+) NaN\)", m.group(3)):
                res[cur][f"  [{m.group(1)}] {m.group(2)} skew {sk}: differ"] += int(d)
                res[cur][f"  [{m.group(1)}] {m.group(2)} skew {sk}: NaN"] += int(n)
            res[cur][f"  [{m.group(1)}] stream 1: differ"] += int(m.group(4))
            res[cur][f"  [{m.group(1)}] stream 1: NaN"] += int(m.group(5))
        if line.startswith("exit="):
            res[cur][line] += 1
for name in runs:
    r = res[name]
    gpus = ", ".join(f"{k} {v}" for k, v in r.items() if k.endswith(" runs"))
    exits = ", ".join(f"{k} x{v}" for k, v in r.items() if k.startswith("exit="))
    print(f"== {name}: {runs[name]} runs ({gpus}; {exits})")
    for k, v in r.items():
        if k.endswith(" differ") and not k.startswith("  "):
            print(f"   {k[:-7]}: {v} of {r[k[:-7] + ' launches']} launches differ")
    rows = defaultdict(list)  # (config, shape) -> "skew s: d (n NaN)"
    for k, v in r.items():
        m = re.match(r"  \[([^\]]+)\] (.*) skew (\d+): differ", k)
        if m:
            rows[(m.group(1), m.group(2))].append(f"skew {m.group(3)}: {v} ({r[k[:-6] + 'NaN']} NaN)")
        m = re.match(r"  \[([^\]]+)\] stream 1: differ", k)
        if m:
            rows[(m.group(1), "stream 1 (t=2 e=256)")].append(f"{v} ({r[k[:-6] + 'NaN']} NaN)")
    for (cfg, shape), cells in rows.items():
        print(f"     [{cfg}] {shape}: {'; '.join(cells)}")
