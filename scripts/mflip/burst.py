#!/usr/bin/env python3
"""mflip client: N requests sharing one token-id prefix, request i submitted at i*gap s, plus an optional unrelated short prompt at --short-at s; or the same prompts one at a time (--mode solo).
  burst.py --url U --mode burst --seed 1 --n 8 --gap 0 --short-len 1000 --short-at 2 --arm A --visit A1 --cell mix --out f.jsonl
TTFT of each request = from its own submit to the first streamed token. Last prompt index n = the short one (if any). Appends one JSON line."""
import argparse, importlib.util, json, os, random, threading, time
spec = importlib.util.spec_from_file_location("hb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../herd/herd_bench.py"))
hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb)
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", default="intarsia")
ap.add_argument("--mode", choices=["burst", "solo"], required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--shared", type=int, default=16448); ap.add_argument("--own", type=int, default=200)
ap.add_argument("--gap", type=float, default=0.0)
ap.add_argument("--short-len", type=int, default=0); ap.add_argument("--short-at", type=float, default=2.0)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--arm", default=""); ap.add_argument("--visit", default=""); ap.add_argument("--cell", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
prompts = hb.make_prompts("herd", a.seed, a.n, a.shared, a.own, 0)
offs = [i * a.gap for i in range(a.n)]
if a.short_len:
    r2 = random.Random(a.seed * 31 + 5)
    prompts.append([r2.randrange(1000, 100000) for _ in range(a.short_len)]); offs.append(a.short_at)
rows = [None] * len(prompts)
t0 = time.perf_counter()
if a.mode == "solo":
    for i, p in enumerate(prompts): rows[i] = hb.request(a.url, a.model, p, a.max_tokens, None)
else:
    def go(i):
        d = offs[i] - (time.perf_counter() - t0)
        if d > 0: time.sleep(d)
        rows[i] = hb.request(a.url, a.model, prompts[i], a.max_tokens, None)
    ths = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in ths]; [t.join() for t in ths]
ok = [r for r in rows if r["ok"]]
out = {"arm": a.arm, "visit": a.visit, "cell": a.cell, "mode": a.mode, "seed": a.seed, "n": a.n, "gap": a.gap, "short_len": a.short_len, "short_at": a.short_at,
       "ok": len(ok), "offs": offs, "ttft": [r["ttft_s"] and round(r["ttft_s"], 3) for r in rows],
       "end": [r["end_s"] and round(r["end_s"], 3) for r in rows], "cached": [r["cached_tokens"] for r in rows],
       "prompt_tokens": [r["prompt_tokens"] for r in rows], "ids": [r["ids"] for r in rows], "errors": [r["error"] for r in rows],
       "wall_s": round(max([o + (r["end_s"] or 0) for o, r in zip(offs, rows)]), 3) if ok and a.mode == "burst" else round(time.perf_counter() - t0, 3),
       "t": time.strftime("%H:%M:%S")}
open(a.out, "a").write(json.dumps(out) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("ids", "offs", "end", "prompt_tokens")}))
