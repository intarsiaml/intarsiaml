#!/usr/bin/env python3
"""storettft: cold-prompt TTFT with an idle gap between requests (greedy, thinking off, nonce first, exactly --gen tokens).

localeval `speed` sends the next request the moment the previous stream ends, so a prefix-store settle that runs after the
previous request's Done (device -> host moves) is still running when the next request arrives. This script uses localeval's
streaming Client and its filler prompts (same TTFT definition as `localeval speed`) but sleeps --gap seconds after every
request, so the settle finishes in idle time. Gap 0 reproduces back-to-back. One warmup per size, then --reps, order by size.

  uv run --project /opt/code/deepeval python scripts/storettft/ttft_gap.py --url URL --model guidance --arm NAME --gap 5 --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

from localeval.client import Client, model_info, wait_until_idle
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sizes", default="4096,16384,32768")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off")
    info = model_info(target)
    if not info.get("reachable"):
        raise SystemExit(f"unreachable: {info}")
    client = Client(target)
    base = client.chat([{"role": "user", "content": INSTRUCTION.strip()}], max_tokens=1, nonce=True)
    cal = client.chat([{"role": "user", "content": filler(200, 0) + INSTRUCTION}], max_tokens=1, nonce=True)
    per = (cal.prompt_tokens - base.prompt_tokens) / 200
    rows = []
    for size in [int(s) for s in a.sizes.split(",")]:
        n = max(0, round((size - base.prompt_tokens) / per))
        prompt = filler(n, seed=size) + INSTRUCTION
        runs = []
        for rep in range(a.reps + 1):  # rep 0 is the warmup, discarded
            time.sleep(a.gap)
            wait_until_idle(target, 600, False, lambda *x: None)
            r = client.chat([{"role": "user", "content": prompt}], max_tokens=a.gen, force_tokens=True, nonce=True)
            print(f"[gap] {a.arm} gap={a.gap} {size} rep{rep}: " + (f"prompt={r.prompt_tokens} cached={r.cached_tokens or 0} ttft={r.ttft_s:.3f}s "
                  f"decode={r.decode_tok_s:.1f} out={r.completion_tokens}" if r.ok else f"ERROR {r.error}"), flush=True)
            if rep > 0 and r.ok:
                runs.append(r)
        t = [r.ttft_s for r in runs]
        rows.append({"size": size, "prompt_tokens": runs[0].prompt_tokens if runs else None, "ttft_each": [round(x, 3) for x in t],
                     "ttft_median": round(statistics.median(t), 3) if t else None, "cached_max": max([r.cached_tokens or 0 for r in runs], default=0)})
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-gap.json").write_text(json.dumps({"arm": a.arm, "gap_s": a.gap, "gen": a.gen, "rows": rows}, indent=1))
    for r in rows:
        print(f"| {r['size']} | {r['prompt_tokens']} | {r['ttft_median']} | {r['ttft_each']} | cached max {r['cached_max']} |", flush=True)


if __name__ == "__main__":
    main()
