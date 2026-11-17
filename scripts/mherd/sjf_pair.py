#!/usr/bin/env python3
"""mherd sjf cell: a long prompt at t=0 and a short one at t=delay (token ids, /v1/completions, greedy, ignore_eos), or each alone.
  sjf_pair.py --url U --mode pair --seed 1 --long 40000 --short 1000 --delay 2 --arm A --visit A1 --out f.jsonl
  sjf_pair.py --url U --mode solo --seed 1 ...      # the same two prompts, one after the other (ids reference)
Appends one JSON line: TTFT and end of each, ids of each."""
import argparse, importlib.util, json, os, random, threading, time

spec = importlib.util.spec_from_file_location("hb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../herd/herd_bench.py"))
hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb)

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", default="intarsia")
ap.add_argument("--mode", choices=["pair", "solo"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--long", type=int, default=40000); ap.add_argument("--short", type=int, default=1000)
ap.add_argument("--delay", type=float, default=2.0); ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--arm", default=""); ap.add_argument("--visit", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = random.Random(a.seed * 104729 + 7)
ids = lambda k: [rng.randrange(1000, 100000) for _ in range(k)]  # noqa: E731
P = {"long": ids(a.long), "short": ids(a.short)}
res = {}
t_wall = time.perf_counter()
if a.mode == "solo":
    for k in ("long", "short"):
        res[k] = hb.request(a.url, a.model, P[k], a.max_tokens, None)
else:
    def go(k, d):
        time.sleep(d)
        res[k] = hb.request(a.url, a.model, P[k], a.max_tokens, None)
    ths = [threading.Thread(target=go, args=("long", 0.0)), threading.Thread(target=go, args=("short", a.delay))]
    [t.start() for t in ths]; [t.join() for t in ths]
out = {"arm": a.arm, "visit": a.visit, "mode": a.mode, "seed": a.seed, "long": a.long, "short": a.short, "delay": a.delay,
       "wall_s": round(time.perf_counter() - t_wall, 3), "t": time.strftime("%H:%M:%S")}
for k in ("long", "short"):
    r = res[k]
    out[k] = {"ok": r["ok"], "ttft_s": r["ttft_s"] and round(r["ttft_s"], 3), "end_s": r["end_s"] and round(r["end_s"], 3),
              "prompt_tokens": r["prompt_tokens"], "ids": r["ids"], "error": r["error"]}
open(a.out, "a").write(json.dumps(out) + "\n")
print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "ids"}) for k, v in out.items()}))
