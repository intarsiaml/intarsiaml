#!/usr/bin/env python3
"""(lane lazyc copy: short, 4k-a, 4k-b only; model name intarsia) Lane mtt gate 1: greedy token ids of served requests (/v1/completions, return_token_ids) and the MTP counters around each.
usage: ids.py BASE_URL OUT.json PROMPT_4K [max_tokens=256]
Cases: short, 4k (two prompts: the file behind two fixed prefixes). Fixed text, so two servers must give the same ids."""
import json, re, sys, urllib.request
base, out, p4 = sys.argv[1:4]
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 256
t4 = open(p4).read()
cases = [("short", "Explain how a hash map handles collisions, with an example in Python."),
         ("4k-a", "[gate A] " + t4), ("4k-b", "Second document, read it carefully.\n" + t4)]
def metrics():
    try: txt = urllib.request.urlopen(base.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode()
    except Exception: return {}
    return {m.group(1): float(m.group(2)) for m in (re.match(r"^(intarsia_spec_\w+)\s+([-0-9.e+]+)$", l) for l in txt.splitlines()) if m}
res = {}
for name, prompt in cases:
    m0 = metrics()
    body = {"model": "intarsia", "prompt": prompt, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    with urllib.request.urlopen(urllib.request.Request(base + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        d = json.load(r)
    m1 = metrics()
    res[name] = {"ids": d["choices"][0]["token_ids"], "usage": d.get("usage"), "spec": {k: m1[k] - m0.get(k, 0) for k in sorted(m1) if m1[k] != m0.get(k, 0)}}
    print(name, "tokens", len(res[name]["ids"]), "prompt", (d.get("usage") or {}).get("prompt_tokens"), "spec", res[name]["spec"], flush=True)
json.dump(res, open(out, "w"))
