#!/usr/bin/env python3
"""kvflip: token ids and text of long-prompt requests from an intarsia server (/v1/completions, return_token_ids), greedy (T=0) and seeded sampled
(T=0.7, top_p 0.95, seed 11), so two server configurations (the KV-once knob unset / set) can be compared token for token.
  uv run --project /opt/code/deepeval python scripts/kvflip/ids.py --url URL --model guidance --arm a1 --out DIR --sizes 4000,32500,65300 --gen 700 --modes greedy,sampled
Prompt sizes cross the decode chunk changes at 4096 / 32768 / 65536 keys."""
import argparse, json, pathlib, sys, urllib.request
from localeval.speed import INSTRUCTION, filler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--sizes", default="4000,32500,65300"); ap.add_argument("--gen", type=int, default=700)
    ap.add_argument("--modes", default="greedy,sampled")
    a = ap.parse_args()
    res = []

    def call(body):
        req = urllib.request.Request(a.url.rstrip("/") + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=900))
        except urllib.error.HTTPError as e:
            print(f"[ids] HTTP {e.code}: {e.read().decode()[:600]}", flush=True)
            raise

    for size in [int(s) for s in a.sizes.split(",")]:
        n = max(1, round(size / 24))
        probe = call({"model": a.model, "prompt": filler(n, seed=size) + INSTRUCTION, "max_tokens": 1, "temperature": 0})["usage"]["prompt_tokens"]
        n = max(1, round(n * size / probe))
        prompt = filler(n, seed=size) + INSTRUCTION
        for mode in a.modes.split(","):
            body = {"model": a.model, "prompt": prompt, "max_tokens": a.gen, "ignore_eos": True, "return_token_ids": True}
            body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.7, "top_p": 0.95, "seed": 11})
            r = call(body); ch = r["choices"][0]
            res.append({"size": size, "mode": mode, "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                        "finish": ch["finish_reason"], "token_ids": ch.get("token_ids"), "text": ch.get("text")})
            print(f"[ids] size {size} {mode}: prompt {r['usage']['prompt_tokens']} out {r['usage']['completion_tokens']} {ch['finish_reason']} ids={len(ch.get('token_ids') or [])}", flush=True)
            out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
            (out / f"{a.arm}-ids.json").write_text(json.dumps(res))

if __name__ == "__main__":
    sys.exit(main())
