#!/usr/bin/env python3
"""Aggregate a run dir (see run.sh): per-phase profile, base-vs-head A/B medians, identity diff, llama.cpp.
usage: aggregate.py RUN_DIR"""
import glob
import json
import re
import statistics as st
import sys

d = sys.argv[1]


def med(x):
    return st.median(x) if x else float("nan")


def rng(x):
    return f"{min(x):.2f}-{max(x):.2f}" if x else "n/a"


# 1. profile: cumulative counters printed every 100 constrained steps by the prof binary
last = None
for line in open(f"{d}/prof-server.err", errors="replace"):
    m = re.match(r"constrain-profile steps=(\d+) compute_ns=(\d+) wait_ns=(\d+) spawn_ns=(\d+) commit_ns=(\d+) apply_ns=(\d+) sample_ns=(\d+)", line)
    if m:
        last = [int(x) for x in m.groups()]
if last:
    n = last[0]
    names = ["mask compute (worker thread, overlapped)", "main-thread wait for mask", "thread spawn + channel", "commit (grammar advance)",
             "mask apply", "sampler (argmax over 151936)"]
    print(f"## profile: {n} constrained steps (prof binary)")
    for nm, v in zip(names, last[1:]):
        print(f"- {nm}: {v / n / 1000:.1f} us/step")
    serial = sum(last[2:])
    print(f"- serial main-thread cost (wait+spawn+commit+apply+sample): {serial / n / 1000:.1f} us/step")

# 2. A/B
rows = [json.loads(l) for l in open(f"{d}/ab-speed.jsonl")] if glob.glob(f"{d}/ab-speed.jsonl") else []
for arm in ("cons", "greedy", "sampled"):
    for eng in ("base", "head"):
        v = [r[f"{arm}_tok_s"] for r in rows if r["label"].startswith(eng)]
        print(f"A/B {arm:8s} {eng}: median {med(v):.2f} tok/s (min-max {rng(v)}, n={len(v)})")
toks = {r["tokens"] for r in rows}
print("tokens per request:", sorted(toks))

# 3. identity
ids = {}
for f in sorted(glob.glob(f"{d}/identity-*.json")):
    ids[f.split("identity-")[1][:-5]] = json.load(open(f))
base = [k for k in ids if k.startswith("base")]
head = [k for k in ids if k.startswith("head")]
if base and head:
    ref = ids[base[0]]
    for k in base[1:] + head:
        diff = [c for c in ref if ids[k].get(c) != ref[c]]
        errs = [c for c in ids[k] if str(ids[k][c]).startswith("ERROR")]
        print(f"identity {k} vs {base[0]}: {len(ref)} cases, {len(diff)} differ {diff}, {len(errs)} errors {errs}")

# 4. llama.cpp
lr = [json.loads(l) for l in open(f"{d}/llama-speed.jsonl")] if glob.glob(f"{d}/llama-speed.jsonl") else []
for arm in ("cons", "greedy", "sampled"):
    v = [r[f"{arm}_tok_s"] for r in lr]
    if v:
        print(f"llama.cpp {arm}: median {med(v):.2f} tok/s (min-max {rng(v)}, n={len(v)}), tokens {sorted({r['tokens'] for r in lr})}")
