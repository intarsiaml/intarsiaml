#!/usr/bin/env python3
"""lane interleave: before/after table from a window's raw dir (replay2 jsonl + server metrics diffs).
usage: table.py RAW_DIR ARM [ARM...]   (ARM: file prefix, e.g. E-main-interleaved-seed0-k2; metrics from serve-<arm name>.metrics[.before])"""
import json, statistics, sys, pathlib

def hist(path, name):
    out = {}
    for line in open(path):
        if line.startswith(name + "_bucket"):
            le = line.split('le="')[1].split('"')[0]
            out[le] = float(line.rsplit(" ", 1)[1])
        elif line.startswith(name + " ") or line.startswith(name + "_total "):
            out["value"] = float(line.rsplit(" ", 1)[1])
    return out

def diff(raw, arm, name):
    a, b = hist(raw / f"serve-{arm}.metrics", name), hist(raw / f"serve-{arm}.metrics.before", name)
    return {k: a[k] - b.get(k, 0.0) for k in a}

raw = pathlib.Path(sys.argv[1])
print("| arm | ITL p99 median / worst (client) | TTFT sum | decode/stream median | combined | wall | server gaps >0.15 / >0.25 / >0.5 s | paused steps | settles: whole / slices |")
print("|---|---|---|---|---|---|---|---|---|")
for pfx in sys.argv[2:]:
    rows = [json.loads(l) for l in open(raw / f"{pfx}.jsonl")]
    ok = [r for r in rows if r["ok"]]
    summ = json.load(open(raw / f"{pfx}.summary.json"))
    name = pfx.split("-interleaved")[0]
    p99 = [r["itl_ms_p99"] for r in ok if r["itl_ms_p99"] is not None]
    ttft = sum(r["ttft_s"] for r in ok)
    dec = statistics.median(r["decode_tok_s"] for r in ok)
    toks = sum(r["completion_tokens"] for r in ok)
    wall = summ["wall_s"]
    h = diff(raw, name, "intarsia_batch_itl_seconds")
    n = h.get("+Inf", 0)
    gt = lambda le: int(n - h.get(le, 0))
    paused = diff(raw, name, "intarsia_batch_paused_steps_total").get("value", "n/a")
    log = (raw / f"serve-{name}.log").read_text()
    whole = log.count("prefix store: settle (")
    slices = log.count("prefix store: settle slice (")
    print(f"| {name} | {statistics.median(p99):.0f} / {max(p99):.0f} ms | {ttft:.1f} s | {dec:.1f} tok/s | {toks / wall:.1f} tok/s | {wall:.1f} s "
          f"| {gt('0.15')} / {gt('0.25')} / {gt('0.5')} of {int(n)} | {paused if paused == 'n/a' else int(paused)} | {whole} / {slices} |")
