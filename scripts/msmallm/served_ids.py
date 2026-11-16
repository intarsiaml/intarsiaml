#!/usr/bin/env python3
"""Lane msmallm: greedy token ids of 1 / 4 / 9 / 16 concurrent /v1/completions requests on a running `intarsia serve`.
Stream i's prompt is a slice of a long text: i % 3 == 0 about 100 tokens (one in-window NAX prefill call per projection),
1 about 2150 tokens (2048 + an in-window remainder), 2 about 4100 tokens (no in-window call).
  served_ids.py run --url http://127.0.0.1:18671/v1 --text prompt-16384.txt --out arm.json [--gen 256]
  served_ids.py compare a.json b.json      # every (streams, i) cell equal between two arms; also each arm's streams vs its solo run"""
import argparse, json, sys, threading, urllib.request

CHARS = [430, 9300, 17800]


def post(url, body):
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def run(url, model, text, out, gen):
    t = open(text).read()
    prompts = [t[1500 * i: 1500 * i + CHARS[i % 3]] for i in range(16)]
    res = {}
    for n in (1, 4, 9, 16):
        got = [None] * n

        def one(i):
            ch = post(url, {"model": model, "prompt": prompts[i], "max_tokens": gen, "ignore_eos": True, "temperature": 0.0, "return_token_ids": True})["choices"][0]
            got[i] = ch.get("token_ids") or ch["text"]

        th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        [x.start() for x in th]
        [x.join() for x in th]
        res[str(n)] = got
        print(f"streams {n}: {[len(g) if isinstance(g, list) else 'text' for g in got]}", flush=True)
    json.dump(res, open(out, "w"))
    # inside the arm: a stream's ids equal the 1-stream (i = 0) and the first 4-stream run of the same prompt
    ref = {i: res["4"][i] for i in range(4)}
    ref[0] = res["1"][0]
    bad = [(n, i) for n in ("4", "9", "16") for i in range(int(n)) if i in ref and res[n][i] != ref[i]]
    print(f"within arm: streams vs fewer streams, same prompt: {'SAME' if not bad else 'DIFFER ' + str(bad)}")


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = [(n, i) for n in A for i in range(len(A[n])) if A[n][i] != B.get(n, [None] * 99)[i]]
    cells = sum(len(v) for v in A.values())
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: {'SAME' if not bad else 'DIFFER'} ({cells - len(bad)}/{cells} cells equal){'' if not bad else ' ' + str(bad)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url")
    ap.add_argument("--model", default="intarsia")
    ap.add_argument("--text")
    ap.add_argument("--out")
    ap.add_argument("--gen", type=int, default=256)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.url, a.model, a.text, a.out, a.gen)
    else:
        compare(*a.files)
        sys.exit(0)
