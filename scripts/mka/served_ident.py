#!/usr/bin/env python3
"""Lane mka gate 4 (served): greedy token ids of N concurrent streams against the same prompts run alone, on a running
`intarsia serve --max-batch 8`. Prompts are slices of a long text (about 400 and 2500 tokens: one and many KV splits).
  served_ident.py run --url http://127.0.0.1:18631/v1 --text prompt-16384.txt --out arm.json
  served_ident.py compare a.json b.json      # every cell equal between two arms
`run` also checks, inside one arm, that every stream of the 4- and 8-stream cells equals its solo run."""
import argparse, json, sys, threading, urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def run(url, model, text, out, gen):
    t = open(text).read()
    prompts = []
    for i in range(8):
        n = 1600 if i % 2 == 0 else 10000  # characters
        off = 3000 * i
        prompts.append(t[off:off + n])
    res = {}
    for n in (1, 4, 8):
        got = [None] * 8

        def one(i):
            ch = post(url, {"model": model, "prompt": prompts[i], "max_tokens": gen, "ignore_eos": True, "temperature": 0.0, "return_token_ids": True})["choices"][0]
            got[i] = ch.get("token_ids") or ch["text"]

        if n == 1:
            for i in range(8):
                one(i)
        else:
            for lo in range(0, 8, n):
                ths = [threading.Thread(target=one, args=(i,)) for i in range(lo, lo + n)]
                [x.start() for x in ths]
                [x.join() for x in ths]
        res[f"n{n}"] = got
        print(f"n={n}: {sum(len(g) for g in got)} tokens", flush=True)
    bad = sum(res[f"n{n}"][i] != res["n1"][i] for n in (4, 8) for i in range(8))
    print(f"streams vs solo inside this arm: {16 - bad} of 16 stream cells equal their solo run")
    json.dump(res, open(out, "w"))
    return bad


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = sum(A[k][i] != B[k][i] for k in A for i in range(len(A[k])))
    tot = sum(len(A[k]) for k in A)
    print(f"{a} vs {b}: {tot - bad} of {tot} cells equal")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url")
    ap.add_argument("--model", default="intarsia")
    ap.add_argument("--text")
    ap.add_argument("--out")
    ap.add_argument("--gen", type=int, default=96)
    a = ap.parse_args()
    sys.exit(1 if (run(a.url, a.model, a.text, a.out, a.gen) if a.cmd == "run" else compare(*a.files)) else 0)
