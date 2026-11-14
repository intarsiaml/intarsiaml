#!/usr/bin/env python3
"""Lane mtp-sampled: token ids of seeded requests from an intarsia server (/v1/completions, return_token_ids).

ids.py <base_url> <model> <out.json> <metrics_url>

Cases: temperature 0.6 and 1.0 (top_k 20, top_p 0.95: the 35B's generation_config) at 1, 2 and 4 concurrent requests,
temperature 1.0 over the whole vocabulary (top_k 0, top_p 1) at 2, and a mixed batch (4 greedy + 4 sampled) at 4. Every
request has a fixed seed, so a case's ids must not depend on the server's drafting or on the concurrency. Half of the
concurrent groups start staggered, so the batch a request decodes in differs between groups. The prompts are short (one
prefill chunk), so the chunk schedule cannot differ between runs. The /metrics spec counters around each case give the
MTP acceptance.
"""
import json, re, sys, threading, time, urllib.request

BASE, MODEL, OUT, METRICS = sys.argv[1:5]
PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the first twenty prime numbers and explain the sieve of Eratosthenes.",
    "Summarise the causes of the French Revolution in five paragraphs.",
    "def quicksort(xs):\n    # complete this function and add tests\n",
    "Translate into German: The weather is nice today, so we will go for a long walk by the river and then eat ice cream.",
    "Question: A train leaves at 9:40 and arrives at 13:05. How long is the trip? Think step by step.\nAnswer:",
    "The following is a README for a command line tool that renames files in bulk.\n\n# bulkmv\n",
]
MAX = [160, 144, 160, 128, 160, 120, 136, 152]


def scrape():
    out = {}
    try:
        txt = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"^(intarsia_spec_\w+|intarsia_batch_decode_steps_total|intarsia_batch_decode_rows_total)\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def one(i, samp, out, delay=0.0):
    time.sleep(delay)
    body = {"model": MODEL, "prompt": PROMPTS[i], "max_tokens": MAX[i], "ignore_eos": True, "return_token_ids": True, "seed": 1000 + i}
    body.update(samp)
    req = urllib.request.Request(BASE + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out[i] = json.load(r)["choices"][0]["token_ids"]


def run(samp_of, conc):
    out = [None] * len(PROMPTS)
    m0 = scrape()
    t0 = time.time()
    for g, lo in enumerate(range(0, len(PROMPTS), conc)):
        # every other group arrives staggered (the later requests join a batch that already decodes)
        ts = [threading.Thread(target=one, args=(i, samp_of(i), out, 0.35 * (i - lo) if g % 2 else 0.0)) for i in range(lo, min(lo + conc, len(PROMPTS)))]
        [t.start() for t in ts]
        [t.join() for t in ts]
    m1 = scrape()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    return {"ids": out, "wall_s": round(time.time() - t0, 2), "metrics": d}


S06 = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
S10 = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
FULL = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
GREEDY = {"temperature": 0}
res = {}
for name, samp in (("t0.6", S06), ("t1.0", S10)):
    for conc in (1, 2, 4):
        res[f"{name}/b{conc}"] = run(lambda i: samp, conc)
res["t1.0-fullvocab/b2"] = run(lambda i: FULL, 2)
res["mixed-greedy-t1.0/b4"] = run(lambda i: GREEDY if i % 2 == 0 else S10, 4)
json.dump(res, open(OUT, "w"))
for k, v in res.items():
    d = v["metrics"]
    p, a, s = d.get("intarsia_spec_proposed_tokens_total", 0), d.get("intarsia_spec_accepted_tokens_total", 0), d.get("intarsia_spec_steps_total", 0)
    n = sum(len(x or []) for x in v["ids"])
    print(f"{k}: {n} tokens in {v['wall_s']} s; verify passes {s:.0f}, drafted {p:.0f}, accepted {a:.0f}" + (f" ({a / p:.3f}; {(a + s) / s:.2f} tokens/pass)" if p else ""), flush=True)
