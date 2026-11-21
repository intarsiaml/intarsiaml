#!/usr/bin/env python3
"""samp2s: seeded sampled requests (T 0.7, top_p 0.95, fixed seeds, fixed prompts, no nonce) in three cells, replies saved for a byte-for-byte comparison between windows.
  cell a: top_k 20, TWO concurrent streams (seed 11 and seed 23, different prompts, started together)
  cell b: top_k 0 (whole rows), one stream, seed 11 then seed 23 (different prompts)
  cell c: top_k 100 (whole rows), one stream, seed 11 then seed 23
Every request has its own prompt (seed offset by cell and request), so each prefills fresh. Prompt size is set by one probe per size as in samphost's seeded.py.
  uv run --project /opt/code/deepeval python scripts/samp2s/seeded2.py --url URL --model guidance --arm a1 --out DIR [--sizes 4096,32768] [--gen 700] [--metrics URL]"""
import argparse, json, pathlib, sys
from localeval.speed import INSTRUCTION, filler
from multi import body_for, run_streams, scrape, calibrate

SEEDS = (11, 23)
CELLS = [("a-k20x2", 20, 2), ("b-k0x1", 0, 1), ("c-k100x1", 100, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="4096,32768"); ap.add_argument("--gen", type=int, default=700); ap.add_argument("--metrics", default="")
    ap.add_argument("--cells", default="a,b,c")
    a = ap.parse_args()
    res, counters = [], {}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for size in [int(s) for s in a.sizes.split(",")]:
        n = calibrate(a.url, a.model, size)
        for ci, (name, top_k, streams) in enumerate(CELLS):
            if name[0] not in a.cells.split(","):
                continue
            m0 = scrape(a.metrics)
            groups = [[0, 1]] if streams == 2 else [[0], [1]]   # request indexes started together
            for g in groups:
                bodies = []
                for k in g:
                    prompt = filler(n, seed=size + 1000 * k + 100000 * (ci + 1)) + INSTRUCTION
                    bodies.append(body_for(a.model, prompt, a.gen, 0.7, 0.95, top_k, SEEDS[k], True))
                rs = run_streams(a.url, bodies)
                for k, r in zip(g, rs):
                    if r["error"]:
                        print(f"[seeded2] ERROR {name} size {size} seed {SEEDS[k]}: {r['error']}", flush=True)
                    res.append({"cell": name, "size": size, "seed": SEEDS[k], "top_k": top_k, "streams": streams, "prompt_tokens": r["usage"].get("prompt_tokens"),
                                "completion_tokens": r["usage"].get("completion_tokens"), "finish": r["finish"], "error": r["error"], "text": "".join(r["text"])})
                    print(f"[seeded2] {name} size {size} seed {SEEDS[k]}: prompt {res[-1]['prompt_tokens']} out {res[-1]['completion_tokens']} {r['finish']} {len(res[-1]['text'])} chars", flush=True)
                (out / f"{a.arm}-seeded2.json").write_text(json.dumps(res, indent=1))
            m1 = scrape(a.metrics)
            counters[f"{name}@{size}"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in sorted(m1)}
    (out / f"{a.arm}-seeded2.json").write_text(json.dumps(res, indent=1))
    (out / f"{a.arm}-seeded2-counters.json").write_text(json.dumps(counters, indent=1))
    print("[seeded2] counters per cell: " + json.dumps(counters), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
