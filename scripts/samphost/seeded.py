#!/usr/bin/env python3
"""samp64: seeded sampled requests (T=0.7, top_p 0.95, fixed seed, fixed prompt, no nonce) at several prompt sizes; saves the exact reply text
and token counts so two server configurations (INTARSIA_SPEC_SAMPLED_CTX_K set / unset) can be compared byte for byte.
  uv run --project /opt/code/deepeval python scripts/samphost/seeded.py --url URL --model guidance --arm w8 --out DIR --sizes 4096,16000,40200,68000 --gen 700
A prompt of 32300 tokens with --gen 700 crosses a 32768-token schedule entry inside the decode; 68000 sits past it from the first token; 4096 never reaches it."""
import argparse, json, pathlib, sys, urllib.request
from localeval.speed import INSTRUCTION, filler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--sizes", default="4096,16000,40200,68000"); ap.add_argument("--gen", type=int, default=700)
    ap.add_argument("--seeds", default="11"); ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    res = []

    def call(body):
        req = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=900))
        except urllib.error.HTTPError as e:
            print(f"[seeded] HTTP {e.code}: {e.read().decode()[:600]}", flush=True)
            raise

    for size in [int(s) for s in a.sizes.split(",")]:
        # size = target prompt tokens: one probe (max_tokens 1) at a first guess of 24 tokens per sentence (measured: 23 to 24), then the sentence count is scaled once
        n = max(1, round(size / 24))
        probe = call({"model": a.model, "messages": [{"role": "user", "content": filler(n, seed=size) + INSTRUCTION}], "max_tokens": 1, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}})["usage"]["prompt_tokens"]
        n = max(1, round(n * size / probe))
        prompt = filler(n, seed=size) + INSTRUCTION
        for seed in [int(s) for s in a.seeds.split(",")]:
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.gen, "temperature": a.temperature,
                    "top_p": 0.95, "seed": seed, "ignore_eos": True, "min_tokens": a.gen, "chat_template_kwargs": {"enable_thinking": False}}
            r = call(body)
            ch = r["choices"][0]
            res.append({"size": size, "seed": seed, "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                        "finish": ch["finish_reason"], "text": ch["message"].get("content")})
            out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
            (out / f"{a.arm}-seeded.json").write_text(json.dumps(res, indent=1))   # partial results survive a later failure
            print(f"[seeded] size {size} seed {seed}: prompt {r['usage']['prompt_tokens']} out {r['usage']['completion_tokens']} {ch['finish_reason']} {len(res[-1]['text'] or '')} chars", flush=True)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-seeded.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    sys.exit(main())
