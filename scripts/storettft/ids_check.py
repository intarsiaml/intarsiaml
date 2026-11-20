#!/usr/bin/env python3
"""storettft gate (b), cold-prompt part: greedy replies for fixed prompts of 1k to 33k tokens, each sent twice (cold, then again so the
second one restores from the store), saved as text + completion tokens for comparison between arms (compare_ids.py).
Prompts are localeval's filler notes, the same text in every arm (no nonce), thinking off, temperature 0, 64 tokens, one request at a time.

  uv run --project /opt/code/deepeval python scripts/storettft/ids_check.py --url URL --model guidance --arm NAME --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from localeval.client import Client, model_info
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sizes", default="1500,3000,4300,6200,8300,10300,12500,16500,20500,24600,28700,33000")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off")
    if not model_info(target).get("reachable"):
        raise SystemExit("unreachable")
    client = Client(target)
    # fixed sentences-per-size (22.1 tokens per sentence, 52 base tokens: localeval's calibration on this tokenizer): the calibration
    # requests carry a random nonce whose token count shifts the sentence count by one between runs, which makes the prompts differ.
    per, base_tokens = 22.1, 52
    rows = []
    for size in [int(s) for s in a.sizes.split(",")]:
        n = max(0, round((size - base_tokens) / per))
        prompt = filler(n, seed=size) + INSTRUCTION
        for rep, kind in enumerate(("cold", "again")):
            r = client.chat([{"role": "user", "content": prompt}], max_tokens=a.gen, nonce=False)
            rows.append({"size": size, "kind": kind, "ok": r.ok, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens or 0,
                         "completion_tokens": r.completion_tokens, "text": r.text, "ttft_s": r.ttft_s})
            print(f"[ids] {a.arm} {size} {kind}: prompt={r.prompt_tokens} cached={r.cached_tokens or 0} out={r.completion_tokens} ttft={r.ttft_s:.2f}s", flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-ids.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
