#!/usr/bin/env python3
"""Lane mdraft (2026-11-16) greedy-ids gate: token ids of served requests (/v1/completions, return_token_ids) at 1 and 2 streams.

usage: ids3.py BASE_URL OUT.json PROMPT_4K [max_tokens=256] [model=m]
Cases: short, 4k-a, 4k-b (two prompts: the file behind two fixed prefixes), each alone ("s1") and then in the three concurrent pairs
short|4k-a, 4k-a|4k-b, 4k-b|short ("s2", both requests of a pair in flight together). Greedy, forced length (ignore_eos), so every arm
must return the same ids for the same prompt whatever the drafting or the concurrency. The /metrics spec counters around each group
tell whether drafting ran. Output: {"s1:short": ids, "s2:short|4k-a:0": ids, ...}.
"""
import json, re, sys, threading, urllib.request

base, out, p4 = sys.argv[1:4]
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 256
model = sys.argv[5] if len(sys.argv) > 5 else "m"
t4 = open(p4).read()
cases = {"short": "Explain how a hash map handles collisions, with an example in Python.",
         "4k-a": "[gate A] " + t4, "4k-b": "Second document, read it carefully.\n" + t4}
pairs = [("short", "4k-a"), ("4k-a", "4k-b"), ("4k-b", "short")]


def metrics():
    try:
        txt = urllib.request.urlopen(base.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    return {m.group(1): float(m.group(2)) for m in (re.match(r"^(intarsia_spec_\w+)\s+([-0-9.e+]+)$", l) for l in txt.splitlines()) if m}


def one(prompt, res, key):
    body = {"model": model, "prompt": prompt, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    with urllib.request.urlopen(urllib.request.Request(base + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        d = json.load(r)
    res[key] = d["choices"][0]["token_ids"]


res = {}
for name, prompt in cases.items():
    m0 = metrics(); one(prompt, res, f"s1:{name}"); m1 = metrics()
    print(f"s1:{name} tokens {len(res[f's1:{name}'])} spec {({k: m1[k] - m0.get(k, 0) for k in sorted(m1) if m1[k] != m0.get(k, 0)})}", flush=True)
for a, b in pairs:
    m0 = metrics()
    ts = [threading.Thread(target=one, args=(cases[n], res, f"s2:{a}|{b}:{i}")) for i, n in enumerate((a, b))]
    [t.start() for t in ts]; [t.join() for t in ts]
    m1 = metrics()
    print(f"s2:{a}|{b} tokens {[len(res[f's2:{a}|{b}:{i}']) for i in (0, 1)]} spec {({k: m1[k] - m0.get(k, 0) for k in sorted(m1) if m1[k] != m0.get(k, 0)})}", flush=True)
json.dump(res, open(out, "w"))
