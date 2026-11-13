#!/usr/bin/env python3
"""fp8-decode: sequential single-stream decode reps at one or more prompt sizes, with per-rep inter-token latency.

Same timing definitions as scripts/ab35b/sampled.py (localeval's streaming Client: decode tok/s = (completion_tokens - 1) /
decode window, peak 32-token stretch), plus the per-rep ITL p50 / p99 / cv so a uniformly slower rep (every step slower)
can be told from one with a few stalls. `--same-prompt` sends the same filler text every rep (the nonce prefix still
differs, so no prefix-cache hit); `--seed` pins the sampling seed so the reps also generate the same tokens.

  uv run --project /opt/code/deepeval python scripts/fp8-decode/reps.py --url URL --model guidance --arm X --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import re
import sys
import urllib.request

from localeval.client import Client, model_info, wait_until_idle
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler


def vllm(url):
    """Running requests and generation-token counter of the box's other server (live traffic would bias a rep)."""
    if not url:
        return None
    try:
        txt = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception:
        return None
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^vllm:(num_requests_running|generation_tokens_total|prompt_tokens_total)\{[^}]*\}\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sizes", default="65536")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--same-prompt", action="store_true")
    ap.add_argument("--prompt-seeds", default="", help="comma list of filler indices i (filler seed size + 17 i, as sampled.py's "
                    "rep i), one request each, instead of --reps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vllm-metrics", default="http://spark-4e77:8000/metrics")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra = {"top_p": a.top_p}
    if a.seed is not None:
        extra["seed"] = a.seed
    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off", extra_body=extra)
    info = model_info(target)
    if not info.get("reachable"):
        raise SystemExit(f"unreachable: {info}")
    client = Client(target)
    base = client.chat([{"role": "user", "content": INSTRUCTION.strip()}], max_tokens=1, nonce=True)
    cal = client.chat([{"role": "user", "content": filler(200, 0) + INSTRUCTION}], max_tokens=1, nonce=True)
    per = (cal.prompt_tokens - base.prompt_tokens) / 200
    print(f"[reps] calibrated {per:.2f} tok/sentence base={base.prompt_tokens}", flush=True)
    rows = []
    k = 0
    for size in [int(s) for s in a.sizes.split(",")]:
        n = max(0, round((size - base.prompt_tokens) / per))
        order = [int(x) for x in a.prompt_seeds.split(",")] if a.prompt_seeds else [0 if a.same_prompt else r for r in range(a.reps)]
        for rep, pi in enumerate(order):
            wait_until_idle(target, 600, False, lambda *x: None)
            prompt = filler(n, seed=size + 17 * pi) + INSTRUCTION
            v0 = vllm(a.vllm_metrics)
            r = client.chat([{"role": "user", "content": prompt}], max_tokens=a.gen, temperature=a.temperature,
                            force_tokens=True, nonce=True, thinking="off")
            k += 1
            v1 = vllm(a.vllm_metrics)
            vd = {kk: round(v1[kk] - v0.get(kk, 0)) for kk in v1 if kk != "num_requests_running"} if v0 and v1 else None
            if vd is not None:
                vd["running_after"] = v1.get("num_requests_running")
            row = {"req": k, "size": size, "rep": rep + 1, "prompt_idx": pi, "ok": r.ok, "error": r.error, "prompt_tokens": r.prompt_tokens,
                   "cached": r.cached_tokens, "ttft_s": r.ttft_s, "decode_tok_s": r.decode_tok_s,
                   "peak": r.peak_decode_tok_s, "itl_p50_ms": r.itl_ms_p50, "itl_p99_ms": r.itl_ms_p99, "itl_cv": r.itl_cv,
                   "out": r.completion_tokens, "text_sha": hashlib.sha1(r.text.encode()).hexdigest()[:10],
                   "text_head": r.text[:80], "t_start": r.t_start, "vllm": vd}
            rows.append(row)
            print(f"[reps] req{k} {size} rep{rep + 1} p{pi}: " + (
                f"prompt={r.prompt_tokens} ttft={r.ttft_s:.2f}s decode={r.decode_tok_s:.1f} peak={r.peak_decode_tok_s} "
                f"itl_p50={r.itl_ms_p50} p99={r.itl_ms_p99} cv={r.itl_cv} out={r.completion_tokens} sha={row['text_sha']} "
                f"start={r.t_start} vllm={vd}" if r.ok else f"ERROR {r.error}"), flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-reps.json").write_text(json.dumps({"arm": a.arm, "args": vars(a), "rows": rows}, indent=1))
    for size in sorted({r["size"] for r in rows}):
        v = [r["decode_tok_s"] for r in rows if r["size"] == size and r["ok"]]
        if v:
            print(f"| {a.arm} | {size} | median {statistics.median(v):.1f} | each {[round(x, 1) for x in v]} |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
