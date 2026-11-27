#!/usr/bin/env python3
"""Lane ink-metal-dflash2-ready: HTTP client for `intarsia serve` (/v1/completions, return_token_ids).
  client.py wait URL [secs]                       -> exit 0 when /v1/models answers
  client.py seeds URL OUT.json [N=100] [BUDGET_S] -> seeded sampled requests; OUT = {"T|k|p|seed": ids}
  client.py greedy URL OUT.json PROMPTS.json TOK [conc] -> greedy ids per prompt name (one by one, or all at once with conc)
  client.py cmp A.json B.json                     -> compares the keys both hold"""
import json, sys, threading, time, urllib.request

def get(url, path):
    with urllib.request.urlopen(url + path, timeout=30) as r: return json.load(r)
def post(url, body):
    req = urllib.request.Request(url + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r: return json.load(r)
def model(url): return get(url, "/models")["data"][0]["id"]

PROMPT = "Once upon a time, in a small village by the sea, there lived"
CFG = [(1.0, 0, 1.0, 100), (0.6, 0, 1.0, 100), (0.7, 20, 0.9, 30)]   # T, top_k, top_p, n seeds

cmd = sys.argv[1]
if cmd == "wait":
    t0 = time.time(); lim = float(sys.argv[3]) if len(sys.argv) > 3 else 180
    while time.time() - t0 < lim:
        try: model(sys.argv[2]); sys.exit(0)
        except Exception: time.sleep(0.5)
    sys.exit(1)
elif cmd == "seeds":
    url, out = sys.argv[2], sys.argv[3]
    n_all = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
    m = model(url); res = {}; t0 = time.time(); stop = False
    for seed in range(n_all):
        for T, k, p, n in CFG:
            if seed >= n: continue
            if time.time() - t0 > budget: stop = True; break
            body = {"model": m, "prompt": PROMPT, "max_tokens": 32, "ignore_eos": True, "temperature": T, "top_k": k, "top_p": p, "seed": seed, "return_token_ids": True}
            res[f"{T}|{k}|{p}|{seed}"] = post(url, body)["choices"][0]["token_ids"]
        if stop: break
    json.dump(res, open(out, "w"))
    print(f"seeds: {len(res)} requests in {time.time()-t0:.1f} s" + (" (BUDGET REACHED)" if stop else ""))
elif cmd == "greedy":
    url, out, pf, tok = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    conc = len(sys.argv) > 6 and sys.argv[6] == "conc"
    m = model(url); ps = json.load(open(pf))["results"]; res = {}
    def one(r):
        b = {"model": m, "prompt": r["prompt_ids"], "max_tokens": tok, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
        res[r["name"]] = post(url, b)["choices"][0]["token_ids"]
    t0 = time.time()
    if conc:
        ts = [threading.Thread(target=one, args=(r,)) for r in ps]; [t.start() for t in ts]; [t.join() for t in ts]
    else:
        for r in ps: one(r)
    json.dump(res, open(out, "w"))
    print(f"greedy {'concurrent' if conc else 'solo'}: {[(k, len(v)) for k, v in res.items()]} in {time.time()-t0:.1f} s")
elif cmd == "cmp":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    keys = sorted(set(a) & set(b)); bad = [k for k in keys if a[k] != b[k]]
    print(f"{sys.argv[2]} vs {sys.argv[3]}: common {len(keys)} (of {len(a)} / {len(b)}), differ {len(bad)}" + (f" first {bad[:5]}" if bad else ""))
    sys.exit(1 if bad else 0)
