#!/usr/bin/env python3
"""Lane bi8: the cold greedy reference {"<i>": ids} from the probe log's `COLD <i> plen <n> ids [...]` lines."""
import json, re, sys
ref = {}
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"COLD (\d+) plen \d+ ids (\[[\d, ]*\])", line)
    if m:
        ref[m.group(1)] = json.loads(m.group(2))
json.dump(ref, open(sys.argv[2], "w"))
print(f"{len(ref)} prompts")
