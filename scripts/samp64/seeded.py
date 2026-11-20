#!/usr/bin/env python3
"""samp64: seeded sampled requests (T=0.7, top_p 0.95, fixed seed, fixed prompt, no nonce) at several prompt sizes; saves the exact reply text
and token counts so two server configurations (INTARSIA_SPEC_SAMPLED_MAX_CTX set / unset) can be compared byte for byte.
  uv run --project /opt/code/deepeval python scripts/samp64/seeded.py --url URL --model guidance --arm w8 --out DIR --sizes 4096,32500,68000 --gen 700
A prompt of 32500 tokens with --gen 700 crosses a 32768 limit inside the decode; 68000 sits past it from the first token; 4096 never reaches it."""
import argparse, json, pathlib, sys, urllib.request
from localeval.speed import INSTRUCTION, filler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--sizes", default="4096,32500,68000"); ap.add_argument("--gen", type=int, default=700)
    ap.add_argument("--seeds", default="11,12"); ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    per = 12.0   # tokens per filler sentence, as calibrated by sampled.py (~12.4 on this prompt builder); the exact count is recorded
    res = []
    for size in [int(s) for s in a.sizes.split(",")]:
        n = max(0, round(size / per))
        prompt = filler(n, seed=size) + INSTRUCTION
        for seed in [int(s) for s in a.seeds.split(",")]:
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.gen, "temperature": a.temperature,
                    "top_p": 0.95, "seed": seed, "ignore_eos": True, "min_tokens": a.gen, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            ch = r["choices"][0]
            res.append({"size": size, "seed": seed, "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                        "finish": ch["finish_reason"], "text": ch["message"].get("content")})
            print(f"[seeded] size {size} seed {seed}: prompt {r['usage']['prompt_tokens']} out {r['usage']['completion_tokens']} {ch['finish_reason']} {len(res[-1]['text'] or '')} chars", flush=True)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-seeded.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    sys.exit(main())
