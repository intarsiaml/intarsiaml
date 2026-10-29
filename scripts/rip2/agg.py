#!/usr/bin/env python3
"""Lane rip: the speed A/B table from localeval result.json files (labels rip-<model>-<cell>-<arm>-r<round>).
  agg.py /opt/code/deepeval/results [label-prefix]   prints, per model and cell, A / B medians per round and the median of rounds."""
import glob, json, os, re, statistics, sys

root = sys.argv[1]
pre = sys.argv[2] if len(sys.argv) > 2 else "rip-"
cells = {}  # (model, cell, size or streams) -> arm -> [values per round]
cached = {}
for f in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
    d = json.load(open(f))
    m = re.fullmatch(re.escape(pre) + r"(\w+)-(prefill|decode)-([AB])-r(\d+)", d.get("label") or "")
    if not m:
        continue
    model, cell, arm, _ = m.groups()
    s = d["speed"]
    if cell == "prefill":
        for r in s["rows"]:
            key = (model, "prefill tok/s", r["target_prompt_tokens"])
            cells.setdefault(key, {}).setdefault(arm, []).append(r["prefill_tok_s"]["median"])
            cached[key] = max(cached.get(key, 0), r.get("cached_tokens_max") or 0)
    else:
        for c in s.get("concurrency", []) or []:
            n = c.get("streams") or c.get("concurrency")
            v = c.get("aggregate_decode_tok_s") or c.get("aggregate_tok_s")
            cells.setdefault((model, "decode agg tok/s", n), {}).setdefault(arm, []).append(v)
        for r in s["rows"]:
            cells.setdefault((model, "decode tok/s 1 stream", r["target_prompt_tokens"]), {}).setdefault(arm, []).append(r["decode_tok_s"]["median"])
print("| model | metric | size / streams | A per round | B per round | A median | B median | B vs A | cached max |")
print("|---|---|---:|---|---|---:|---:|---:|---:|")
for key in sorted(cells, key=str):
    a, b = cells[key].get("A", []), cells[key].get("B", [])
    if not a or not b or None in a or None in b:
        print(f"| {key[0]} | {key[1]} | {key[2]} | {a} | {b} | | | | |")
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"| {key[0]} | {key[1]} | {key[2]} | {' / '.join(f'{x:.0f}' if x > 100 else f'{x:.1f}' for x in a)} | "
          f"{' / '.join(f'{x:.0f}' if x > 100 else f'{x:.1f}' for x in b)} | {ma:.1f} | {mb:.1f} | {100 * (mb - ma) / ma:+.1f}% | {cached.get(key, '')} |")
