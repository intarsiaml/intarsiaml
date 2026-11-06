#!/usr/bin/env python3
"""Serve smoke for the fp8-head default (lane lmhead-fp8-default). usage: msmoke.py BASE_URL MODEL OUT_DIR TAG
Four short greedy prompts, 64 tokens: one after the other (solo), then all at once (concurrent; with --max-batch > 1 they are batched),
then one sampled request (temperature 0.7, top_p 0.95). Prints one `SMOKE ...` line each; texts go to OUT_DIR/TAG-*.txt."""
import json, sys, threading, urllib.request, hashlib, os
base, model, out, tag = sys.argv[1:5]
PROMPTS = ["Write a haiku about autumn rain.", "Explain in two sentences why the sky is blue.", "List three prime numbers greater than 100.",
           "What is the capital of Australia? Answer in one sentence."]
def ask(p, temp=0.0, n=64):
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": temp, "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0: body.update(top_p=0.95)
    else: body.update(min_tokens=n, ignore_eos=True)
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    return j["choices"][0]["message"]["content"], j["usage"]["completion_tokens"]
os.makedirs(out, exist_ok=True)
solo = [ask(p) for p in PROMPTS]
res = [None] * len(PROMPTS)
def w(i): res[i] = ask(PROMPTS[i])
ts = [threading.Thread(target=w, args=(i,)) for i in range(len(PROMPTS))]
[t.start() for t in ts]; [t.join() for t in ts]
for name, rs in (("solo", solo), ("conc", res)):
    open(f"{out}/{tag}-{name}.txt", "w").write("\n=====\n".join(t for t, _ in rs))
    print(f"SMOKE {tag} {name}: " + " ".join(hashlib.md5(t.encode()).hexdigest()[:8] + f"/{n}" for t, n in rs))
print(f"SMOKE {tag} conc==solo: {all(a[0] == b[0] for a, b in zip(solo, res))}")
t, n = ask(PROMPTS[0], 0.7, 64)
print(f"SMOKE {tag} sampled: {n} tokens, {len(t)} chars")
