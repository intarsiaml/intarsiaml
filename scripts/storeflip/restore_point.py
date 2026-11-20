#!/usr/bin/env python3
"""storeflip gate 3: restore-point granularity. One long cold single user message A (default 20k tokens), then prompts that share only the first
K tokens of A's message (K = --shared) and then diverge, then A again. Records cached tokens and TTFT of every send. Per share size K a separate A_k (own filler seed, the same text in every
arm, no nonce) and a prompt = filler(K tokens) (a prefix of A_k's text) + a different tail. Greedy, thinking off, one request at a time, a pause after
each request so the store's settle is over.

  uv run --project /opt/code/deepeval python scripts/storeflip/restore_point.py --url URL --model guidance --arm NAME --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from localeval.client import Client, model_info
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler

PER, BASE = 22.1, 52  # tokens per filler sentence / fixed tokens (localeval calibration on this tokenizer; fixed so prompts are identical across arms)
SEED = 4242


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--total", type=int, default=20000)
    ap.add_argument("--shared", default="3000,5000,9000,13000,17000")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--pause", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off")
    if not model_info(target).get("reachable"):
        raise SystemExit("unreachable")
    client = Client(target)
    n_total = round((a.total - BASE) / PER)
    rows = []

    def send(label, content, shared):
        time.sleep(a.pause)
        r = client.chat([{"role": "user", "content": content}], max_tokens=a.gen, nonce=False)
        row = {"label": label, "shared_tokens_target": shared, "ok": r.ok, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens or 0,
               "ttft_s": r.ttft_s, "text": r.text}
        rows.append(row)
        print(f"[restore] {a.arm} {label}: prompt={r.prompt_tokens} cached={r.cached_tokens or 0} ttft={r.ttft_s:.3f}s", flush=True)

    # one seed per share size, so a share prompt can only restore from ITS OWN earlier long message A_k (not from another share prompt's entries)
    for j, k in enumerate(int(x) for x in a.shared.split(",")):
        seed = SEED + j
        send(f"A{k}-cold", filler(n_total, seed) + INSTRUCTION, a.total)
        n = round((k - BASE) / PER)
        send(f"share{k}", filler(n, seed) + f" Now a different question about topic {k}: list ten harbours." + INSTRUCTION, k)
        if j == 0:
            send(f"A{k}-again", filler(n_total, seed) + INSTRUCTION, a.total)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-restore.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
