#!/usr/bin/env python3
"""Lane wire-probe: first index where two `ids:` dumps (generate --show-ids) differ. Args: pairs of files (base, change)."""
import re
import sys


def ids(path):
    text = open(path, errors="replace").read()
    return [int(x) for x in re.findall(r"-?\d+", text.split("ids:", 1)[1])] if "ids:" in text else []


for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    x, y = ids(a), ids(b)
    n = min(len(x), len(y))
    d = next((i for i in range(n) if x[i] != y[i]), None)
    name = a.rsplit("/", 1)[-1]
    if not x or not y:
        print(f"{name}: missing ids ({len(x)} / {len(y)})")
    elif d is None:
        print(f"{name}: identical over {n} ids")
    else:
        print(f"{name}: first difference at id {d} of {n}")
