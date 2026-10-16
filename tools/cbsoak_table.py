#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""One markdown row per cbsoak report.json found under the given dirs (ramp / soak summaries for result docs).
  uv run tools/cbsoak_table.py DIR [DIR...]"""
import json
import pathlib
import sys


def f(x, nd=0):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    print("| run | planned rps | sent | ok/canc/shed/err | completed rps | out tok/s | TTFT p50/p95/p99 | ITL p50/p95/p99/max | ITL in 3k+ prefill p50/p95 (n) | queue max | mean rows | mem growth MB | dev mem growth MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in sys.argv[1:]:
        for p in sorted(pathlib.Path(d).rglob("report.json")):
            r = json.loads(p.read_text())
            c, rt, t, i, m, s = r["counts"], r["rates"], r["ttft_ms"], r["itl_ms"], r["memory"], r.get("server") or {}
            sg = rt["itl_during_long_prefill_ms"]
            q = s.get("queue_depth") or {}
            print(f"| {p.parent.relative_to(d) if p.parent != pathlib.Path(d) else '.'} | {r['meta']['rate']} | {c['sent']} | {c['ok']}/{c['cancelled']}/{c['shed']}/{c['error']} | {f(rt['completed_req_s'], 3)} | {f(rt['out_tok_s'], 1)} | "
                  f"{f(t['p50'])}/{f(t['p95'])}/{f(t['p99'])} | {f(i['p50'], 1)}/{f(i['p95'], 1)}/{f(i['p99'], 1)}/{f(i['max'])} | {f(sg['p50'], 1)}/{f(sg['p95'], 1)} ({sg['n']}) | "
                  f"{f(q.get('max') if isinstance(q, dict) else None)} | {f(s.get('batch_mean_rows'), 2)} | {f(m.get('growth_mb'))} | {f((m.get('device') or {}).get('growth_mb'))} |")


if __name__ == "__main__":
    main()
