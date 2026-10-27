#!/usr/bin/env python3
"""Lane htk: seeded sampled (and greedy) requests against a running `intarsia serve`, N streams at once, token ids saved.
Run against a server with INTARSIA_DEVICE_SAMPLING=0 and again with =1 (same binary, same flags): `compare` the two files, every cell must be equal.
  equal.py run  --url http://tatooine:8296/v1 --out off.json
  equal.py compare off.json on.json
Cells: five temperature / top_k / top_p settings the device path takes (1 <= top_k <= 64), a top_p-only one (host path), greedy; each at 1, 4 and 16 streams.
"""
import argparse, json, sys, threading, urllib.request

SETTINGS = [  # name, temperature, top_k, top_p
    ("t0.7-k20-p1.0", 0.7, 20, 1.0),
    ("t1.0-k40-p0.9", 1.0, 40, 0.9),
    ("t0.8-k64-p0.5", 0.8, 64, 0.5),
    ("t1.3-k8-p0.95", 1.3, 8, 0.95),
    ("t0.6-k1-p1.0", 0.6, 1, 1.0),
    ("t0.8-p0.9-nok", 0.8, 0, 0.9),
    ("greedy", 0.0, 0, 1.0),
]
STREAMS = [1, 4, 16]
PROMPTS = [
    "The history of the printing press begins", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which", "In 1969, the Apollo 11 mission", "Explain, step by step, why the sky is blue. First,",
    "The three laws of thermodynamics state that", "A recipe for sourdough bread needs", "The capital of Australia is",
    "Once upon a time, in a small village by the sea,", "To install the package, run", "The Pythagorean theorem says",
    "In machine learning, overfitting happens when", "The French Revolution started because", "A haiku about autumn:",
    "The difference between TCP and UDP is", "Quantum entanglement is",
]


def post(url, body):
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(url, model, out, gen):
    res = {}
    for name, t, k, p in SETTINGS:
        for n in STREAMS:
            got = [None] * n

            def one(i):
                body = {"model": model, "prompt": PROMPTS[i % len(PROMPTS)] + ("" if i < len(PROMPTS) else f" ({i})"), "max_tokens": gen, "ignore_eos": True,
                        "temperature": t, "top_p": p, "seed": 100 + i, "return_token_ids": True}
                if k:
                    body["top_k"] = k
                ch = post(url, body)["choices"][0]
                got[i] = {"ids": ch.get("token_ids"), "text": ch["text"]}

            ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [x.start() for x in ths]
            [x.join() for x in ths]
            res[f"{name}/n{n}"] = got
            print(f"{name} n={n}: {sum(len(g['ids'] or []) for g in got)} tokens", flush=True)
    json.dump(res, open(out, "w"))


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    for key in A:
        same = A[key] == B.get(key)
        bad += not same
        print(("EQUAL " if same else "DIFFER"), key)
    n_ids = sum(len(g["ids"] or []) for v in A.values() for g in v)
    print(f"{len(A) - bad}/{len(A)} cells identical, {n_ids} token ids compared")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--model", default="intarsia"); r.add_argument("--out", required=True); r.add_argument("--gen", type=int, default=64)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    run(a.url, a.model, a.out, a.gen) if a.cmd == "run" else compare(a.a, a.b)
