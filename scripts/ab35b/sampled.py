#!/usr/bin/env python3
"""ab35b run 3a: sampled decode (temperature 0.7, top_p 0.95), single stream, reps per prompt size.

localeval's `speed` command is greedy only, so this uses localeval's streaming Client directly (same timing
definitions: TTFT, decode tok/s = (completion_tokens - 1) / decode window, peak 32-token stretch) and its filler
prompt builder (`localeval.speed.filler`). Each rep is a fresh nonce-prefixed prompt (no prefix-cache hit on either
engine; cached_tokens is recorded to show it), exactly 256 tokens forced with ignore_eos + min_tokens like `speed`.
For intarsiaml the /metrics spec counters are read before and after each size, giving the MTP acceptance at that size.

  uv run --project /opt/code/deepeval python scripts/ab35b/sampled.py --url URL --model guidance --arm A --out DIR [--metrics URL/metrics]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import urllib.request

from localeval.client import Client, model_info, wait_until_idle
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler


def scrape(url):
    if not url:
        return {}
    try:
        txt = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^(intarsia_spec_\w+|intarsia_tokens_cached_total|intarsia_batch_cached_tokens_total)\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def spread(v):
    v = [x for x in v if x is not None]
    return {"median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)} if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sizes", default="4096,32768,65536")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--metrics", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-idle", type=float, default=600)
    a = ap.parse_args()
    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off", extra_body={"top_p": a.top_p})
    info = model_info(target)
    if not info.get("reachable"):
        raise SystemExit(f"unreachable: {info}")
    ctx = info.get("max_model_len")
    client = Client(target)
    base = client.chat([{"role": "user", "content": INSTRUCTION.strip()}], max_tokens=1, nonce=True)
    cal = client.chat([{"role": "user", "content": filler(200, 0) + INSTRUCTION}], max_tokens=1, nonce=True)
    per = (cal.prompt_tokens - base.prompt_tokens) / 200
    print(f"[sampled] calibrated {per:.2f} tok/sentence base={base.prompt_tokens} ctx={ctx}", flush=True)
    rows = []
    for size in [int(s) for s in a.sizes.split(",")]:
        if ctx and size + a.gen + 64 > ctx:
            rows.append({"size": size, "skipped": f"exceeds context {ctx}"})
            continue
        n = max(0, round((size - base.prompt_tokens) / per))
        m0 = scrape(a.metrics)
        runs = []
        for rep in range(a.reps):
            wait_until_idle(target, a.wait_idle, False, lambda *x: None)
            prompt = filler(n, seed=size + 17 * rep) + INSTRUCTION
            r = client.chat([{"role": "user", "content": prompt}], max_tokens=a.gen, temperature=a.temperature,
                            force_tokens=True, nonce=True, thinking="off")
            runs.append(r)
            print(f"[sampled] {size} rep{rep + 1}: " + (f"prompt={r.prompt_tokens} cached={r.cached_tokens or 0} ttft={r.ttft_s:.2f}s "
                  f"decode={r.decode_tok_s:.1f} peak={r.peak_decode_tok_s} out={r.completion_tokens}" if r.ok else f"ERROR {r.error}"), flush=True)
        m1 = scrape(a.metrics)
        ok = [r for r in runs if r.ok]
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        prop, acc = d.get("intarsia_spec_proposed_tokens_total"), d.get("intarsia_spec_accepted_tokens_total")
        rows.append({"size": size, "prompt_tokens": int(statistics.median([r.prompt_tokens for r in ok])) if ok else None,
                     "ttft_s": spread([r.ttft_s for r in ok]), "decode_tok_s": spread([r.decode_tok_s for r in ok]),
                     "peak_decode_tok_s": spread([r.peak_decode_tok_s for r in ok]),
                     "completion_tokens": spread([r.completion_tokens for r in ok]),
                     "cached_tokens_max": max([r.cached_tokens or 0 for r in ok], default=0),
                     "decode_each": [round(r.decode_tok_s, 1) for r in ok],
                     "errors": [r.error for r in runs if not r.ok],
                     "mtp": {"delta": d, "accept_rate": round(acc / prop, 3) if prop else None,
                             "tokens_per_verify": round((acc + d["intarsia_spec_steps_total"]) / d["intarsia_spec_steps_total"], 3)
                             if d.get("intarsia_spec_steps_total") else None} if m1 else None})
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"arm": a.arm, "url": a.url, "temperature": a.temperature, "top_p": a.top_p, "gen": a.gen, "reps": a.reps, "rows": rows}
    (out / f"{a.arm}-sampled.json").write_text(json.dumps(res, indent=1))
    print("| size | prompt tok | TTFT s | decode tok/s median (min-max) | MTP accept |", flush=True)
    for r in rows:
        if "skipped" in r:
            print(f"| {r['size']} | skipped: {r['skipped']} |")
            continue
        m = r["mtp"]
        print(f"| {r['size']} | {r['prompt_tokens']} | {r['ttft_s']['median']} | {r['decode_tok_s']['median']} ({r['decode_tok_s']['min']}-{r['decode_tok_s']['max']}) | "
              f"{(m or {}).get('accept_rate')} |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
