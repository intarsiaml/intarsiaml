#!/usr/bin/env python3
"""Lane lb: barrier ISA before/after, from the isa.sh dumps (<isa dir>/{base,dev}/<file>.s).

usage: isacmp.py <isa dir> <file> [kernel-name-substring ...]
Per kernel (first match of each substring, or every kernel with a barrier): number of `s_barrier_signal`, how many of them have
`s_wait_loadcnt 0x0` / `s_wait_loadcnt_dscnt 0x0` in the 12 instructions before them (all outstanding global loads drained), how many have an
LDS wait (`s_wait_dscnt`) before them, and how many have `global_inv` in the 5 instructions after them.
"""
import collections
import re
import sys


def parse(path):
    res = collections.OrderedDict()
    cur = None
    for ln in open(path):
        m = re.match(r"^(_Z\w+):\s*(;.*)?$", ln)
        if m:
            cur = m.group(1)
            res[cur] = []
            continue
        if cur and ln.startswith("\t"):
            res[cur].append(ln.strip())
    out = {}
    for k, v in res.items():
        nb = lc = ds = gi = 0
        for i, l in enumerate(v):
            if l.startswith("s_barrier_signal"):
                nb += 1
                prev, nxt = v[max(0, i - 12):i], v[i + 1:i + 6]
                lc += any(re.match(r"s_wait_loadcnt(_dscnt)? 0x0", p) for p in prev)
                ds += any(re.match(r"s_wait_(dscnt|loadcnt_dscnt) ", p) for p in prev)
                gi += any(p.startswith("global_inv") for p in nxt)
        out[k] = (nb, lc, ds, gi)
    return out


def short(k):
    return re.sub(r"^_Z(N12_GLOBAL__N_)?\d*", "", k)[:64]


def main():
    d, f, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
    b, v = parse(f"{d}/base/{f}.s"), parse(f"{d}/dev/{f}.s")
    print("| file | kernel | barriers | loadcnt 0 before (base / dev) | dscnt wait before (base / dev) | global_inv after (base / dev) |")
    print("|---|---|---:|---|---|---|")
    keys = [k for k in b if b[k][0]]
    if pats:
        keys = [next(k for k in keys if p in k) for p in pats if any(p in k for k in keys)]
    for k in keys:
        x, y = b[k], v.get(k, (0, 0, 0, 0))
        print(f"| {f} | `{short(k)}` | {x[0]} | {x[1]} / {y[1]} | {x[2]} / {y[2]} | {x[3]} / {y[3]} |")


main()
