#!/usr/bin/env python3
"""Lane srv16: served-token identity with long prompts (equal.py's prompts are < 16 tokens, too short to pack). N concurrent
requests, distinct corpus slices of 300..1100 tokens (so several prompts prefill together and pack), greedy and one seeded
sampled setting, 64 tokens each, token ids saved.
  ident_long.py run --url http://host:port/v1 --corpus tools/prompts/corpus_docs.txt --out a.json
  ident_long.py compare a.json b.json"""
import argparse, json, sys, threading, urllib.request

LENS = [4400, 1300, 3000, 2200, 3900, 1700, 4300, 2600, 3500, 1200, 4100, 2900, 3700, 2000, 4200, 3300]  # chars, ~0.27 tok/char
CELLS = [("greedy", 0.0, 0, 1.0), ("t0.8-k40-p0.95", 0.8, 40, 0.95)]
# --mix: 1k-token prompts beside 3k..7k-token ones (several chunks, so a non-final 2048-row chunk shares a pass), each cell run twice
# with the same prompts (the second run resumes from a reused prefix at a chunk boundary, so a packed segment starts past 0)
LENS_MIX = [26000, 4000, 12000, 3800, 19000, 4200, 15000, 3900]
CELLS_MIX = [("mix-greedy", 0.0, 0, 1.0), ("mix-greedy-again", 0.0, 0, 1.0), ("mix-t0.8-k40-p0.95", 0.8, 40, 0.95)]


def post(url, body):
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(url, model, corpus, out, gen, mix=False):
    text = open(corpus).read()
    res = {}
    lens, cells = (LENS_MIX, CELLS_MIX) if mix else (LENS, CELLS)
    span = max(lens) + 100
    for name, t, k, p in cells:
        for n in ((4, 8) if mix else (4, 16)):
            got = [None] * n

            def one(i):
                lo = (i * 3911) % max(1, len(text) - span)
                tag = name.replace("-again", "") if mix else name
                body = {"model": model, "prompt": f"[{tag}-{n}-{i}] " + text[lo: lo + lens[i % len(lens)]], "max_tokens": gen, "ignore_eos": True,
                        "temperature": t, "top_p": p, "seed": 200 + i, "return_token_ids": True}
                if k:
                    body["top_k"] = k
                r = post(url, body)
                got[i] = {"ids": r["choices"][0].get("token_ids"), "prompt_tokens": r.get("usage", {}).get("prompt_tokens")}

            ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            [x.start() for x in ths]
            [x.join() for x in ths]
            res[f"{name}/n{n}"] = got
            print(f"{name} n={n}: {sum(len(g['ids'] or []) for g in got)} tokens, prompts {[g['prompt_tokens'] for g in got]}", flush=True)
    json.dump(res, open(out, "w"))


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    for key in A:
        same = [g["ids"] for g in A[key]] == [g["ids"] for g in B.get(key, [])]
        bad += not same
        print(("EQUAL " if same else "DIFFER"), key)
    n_ids = sum(len(g["ids"] or []) for v in A.values() for g in v)
    print(f"{len(A) - bad}/{len(A)} long-prompt cells identical, {n_ids} token ids compared")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--model", default="intarsia"); r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True); r.add_argument("--gen", type=int, default=64); r.add_argument("--mix", action="store_true")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    run(a.url, a.model, a.corpus, a.out, a.gen, a.mix) if a.cmd == "run" else compare(a.a, a.b)
