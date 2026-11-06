"""Lane dm: speed-spec rows (natural-v1 by prompt size) of every visit in a p1-bench steps.log: decode tok/s median and spec acceptance.
usage: python3 -I scripts/dm/p1_table.py <steps.log>"""
import json, os, re, sys

R = "/opt/code/deepeval"
for line in open(sys.argv[1]):
    m = re.search(r"visit (\d+) arm (\S+) suite speed-spec exit \d+ (results/\S+/result\.json)", line)
    if not m:
        continue
    v, arm, path = m.groups()
    d = json.load(open(os.path.join(R, path)))
    cells = []
    for r in d["speed"]["rows"]:
        dt = r.get("decode_tok_s") or {}
        cells.append(f"{r['target_prompt_tokens']}: {dt.get('median')} ({dt.get('min')}-{dt.get('max')})")
    ss = (d.get("workload") or {}).get("spec_stats") or {}
    acc = ss.get("acceptance_rate")
    print(f"v{v} {arm:5s} | " + " | ".join(cells) + f" | accept {acc}")
