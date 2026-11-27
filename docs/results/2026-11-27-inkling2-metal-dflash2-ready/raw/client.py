#!/usr/bin/env python3
"""Lane ink-metal-dflash2-ready: HTTP client for `intarsia serve` (/v1/completions, return_token_ids).
  client.py wait URL [secs]                       -> exit 0 when /v1/models answers
  client.py seeds URL OUT.json [N=100] [BUDGET_S] -> seeded sampled requests; OUT = {"T|k|p|seed": ids}
  client.py greedy URL OUT.json PROMPTS.json TOK [conc] -> greedy ids per prompt name (one by one, or all at once with conc)
  client.py cmp A.json B.json                     -> compares the keys both hold
  client.py chat URL OUT.json B4.json ABBA3.json  -> check 5: multi-turn chats (chat endpoint + token-id extension), per turn ids and cached_tokens
  client.py cmpchat A.json B.json                 -> check 5: per-turn ids equal between two servers"""
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

elif cmd == "chat":
    url, out, bf, af = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    m = model(url); res = {}
    def cpost(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r: return json.load(r)
    CHATS = [["Give me a one-paragraph summary of how a hash map handles collisions.", "Now show a short Python example of it.", "What is its worst-case lookup time, and why?"],
             ["Write two sentences about a lighthouse keeper who finds a bottle.", "Continue the story with two more sentences.", "Summarise the whole story in five words."]]
    for ci, users in enumerate(CHATS):
        msgs = []
        for ti, u in enumerate(users):
            msgs.append({"role": "user", "content": u})
            r = cpost("/chat/completions", {"model": m, "messages": msgs, "max_tokens": 128, "temperature": 0, "return_token_ids": True})
            ch = r["choices"][0]; msgs.append({"role": "assistant", "content": ch["message"].get("content") or ""})
            res[f"chat{ci}-t{ti+1}"] = {"ids": ch["token_ids"], "prompt_tokens": r["usage"]["prompt_tokens"], "cached_tokens": r["usage"]["prompt_tokens_details"]["cached_tokens"], "content_chars": len(ch["message"].get("content") or "")}
    starts = {r["name"]: r["prompt_ids"] for r in json.load(open(bf))["results"]}
    blocks = {r["name"]: r["prompt_ids"] for r in json.load(open(af))["results"]}
    for ci, (sn, b1, b2) in enumerate([("chat_reasoning_rnone", "count", "story"), ("long_prose_300", "story", "count")]):
        prompt = list(starts[sn])
        for ti in range(3):
            if ti: prompt = prompt + gen + list(blocks[b1 if ti == 1 else b2])
            r = cpost("/completions", {"model": m, "prompt": prompt, "max_tokens": 128, "temperature": 0, "ignore_eos": True, "return_token_ids": True})
            gen = r["choices"][0]["token_ids"]
            res[f"ids{ci}-t{ti+1}"] = {"ids": gen, "prompt_tokens": r["usage"]["prompt_tokens"], "cached_tokens": r["usage"]["prompt_tokens_details"]["cached_tokens"]}
    json.dump(res, open(out, "w"))
    print("chat:", {k: (len(v["ids"]), v["prompt_tokens"], v["cached_tokens"]) for k, v in res.items()})
elif cmd == "cmpchat":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    bad = [k for k in a if a[k]["ids"] != b.get(k, {}).get("ids")]
    print(f"cmpchat {sys.argv[2]} vs {sys.argv[3]}: turns {len(a)} / {len(b)}, differ {len(bad)} {bad}")
    print("cached_tokens a:", {k: v["cached_tokens"] for k, v in a.items()}); print("cached_tokens b:", {k: v["cached_tokens"] for k, v in b.items()})
    sys.exit(1 if bad or len(a) != len(b) else 0)
