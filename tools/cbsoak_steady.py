#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Steady-state ITL of a Poisson soak (Q13: steady-state ITL p95 <= ~60 ms).

  uv run tools/cbsoak_steady.py DIR [DIR ...] [--skip-s 360] [--max-p95 60]

Reads DIR/raw.json (or DIR/soak/raw.json, DIR/soak2/raw.json) written by cbsoak.py run. A decode gap is *steady* when no other
request is in its prefill phase while the gap elapses; the prefill phase of a request is [t_send, t_send + TTFT] as the client sees it
(it includes queue wait, so it can only remove too many gaps, never keep a prefill-stalled one). Every other gap is a prefill-overlap
gap. Gaps are also split by whether they overlap a >= 3000 token prompt. The first --skip-s seconds are excluded (warm-up).
Prints one table; exit 1 when steady p95 exceeds --max-p95.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cbsoak import chunk_times, percentile  # noqa: E402


def load(d):
    d = pathlib.Path(d)
    for c in (d / "raw.json", d / "soak" / "raw.json", d / "soak2" / "raw.json"):
        if c.exists():
            return json.loads(c.read_text())["requests"]
    raise SystemExit(f"no raw.json under {d}")


def split(reqs, skip_s):
    wins = []  # (i, t0, t1, long)
    for r in reqs:
        if r.get("ttft_ms") is None:
            continue
        pt = (r.get("usage") or {}).get("prompt_tokens") or r.get("target_tokens") or 0
        wins.append((r["i"], r["t_send"], r["t_send"] + r["ttft_ms"] / 1000.0, pt >= 3000))
    wins.sort(key=lambda w: w[1])
    steady, over, over_long = [], [], []
    for r in reqs:
        if r["status"] not in ("ok", "cancelled") or not r.get("itl_ms"):
            continue
        ts = chunk_times(r)
        for k, g in enumerate(r["itl_ms"]):
            a = ts[k + 1]
            b = a - g / 1000.0
            if b < skip_s:
                continue
            hit = [w for w in wins if w[0] != r["i"] and a > w[1] and b < w[2]]
            if not hit:
                steady.append(g)
            else:
                over.append(g)
                if any(w[3] for w in hit):
                    over_long.append(g)
    return steady, over, over_long


def row(name, xs):
    if not xs:
        return f"| {name} | 0 | | | | |"
    return f"| {name} | {len(xs)} | {percentile(xs, 50):.1f} | {percentile(xs, 95):.1f} | {percentile(xs, 99):.1f} | {max(xs):.1f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--skip-s", type=float, default=0.0)
    ap.add_argument("--max-p95", type=float, default=None)
    a = ap.parse_args()
    rc = 0
    for d in a.dirs:
        s, o, ol = split(load(d), a.skip_s)
        print(f"### {d} (skip first {a.skip_s:.0f} s)\n\n| gaps | n | p50 ms | p95 ms | p99 ms | max ms |\n|---|---|---|---|---|---|")
        print(row("steady (no other request prefilling)", s))
        print(row("overlapping any other request's prefill", o))
        print(row("overlapping a >=3000-token prefill", ol))
        if a.max_p95 is not None and s:
            ok = percentile(s, 95) <= a.max_p95
            print(f"\nsteady p95 {percentile(s, 95):.1f} ms vs limit {a.max_p95}: {'pass' if ok else 'FAIL'}\n")
            rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
