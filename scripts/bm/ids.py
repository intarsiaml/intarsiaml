#!/usr/bin/env python3
"""Lane bm: greedy token ids of a set of prompts from an OpenAI-compatible intarsia server (/v1/completions, return_token_ids).
ids.py <base_url> <model> <out.json> <concurrency>: all prompts at once when concurrency > 1, one by one otherwise."""
import json, sys, threading, urllib.request

BASE, MODEL, OUT, CONC = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SEEDS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle. " * 6,
    "List the first twenty prime numbers and explain the sieve of Eratosthenes.",
    "Summarise the causes of the French Revolution. " * 20,
    "def quicksort(xs):\n    # complete this function and add tests\n",
    "Translate into German: The weather is nice today, so we will go for a long walk by the river and then eat ice cream. " * 3,
]
MAX = [256, 192, 256, 160, 224, 200]


def one(i, out):
    body = json.dumps({"model": MODEL, "prompt": SEEDS[i], "max_tokens": MAX[i], "temperature": 0, "ignore_eos": True,
                       "return_token_ids": True}).encode()
    req = urllib.request.Request(BASE + "/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out[i] = json.load(r)["choices"][0]["token_ids"]


out = [None] * len(SEEDS)
if CONC > 1:
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(len(SEEDS))]
    [t.start() for t in ts]
    [t.join() for t in ts]
else:
    for i in range(len(SEEDS)):
        one(i, out)
json.dump(out, open(OUT, "w"))
print(OUT, [len(x or []) for x in out])
