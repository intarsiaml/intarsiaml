#!/usr/bin/env python3
"""Lane fp8kv: greedy-token agreement on long prompts against a running `intarsia serve` (one stream, temperature 0, 256 tokens,
ignore_eos, token ids saved). Prompts: scripts/cmtp/ids-prompts/{4k,16k}-{0,1}.txt, and 1k prompts cut from the 4k ones.
  greedy.py run --url http://spark-fa00:8190/v1 --out fp8.json
  greedy.py compare bf16.json fp8.json   (per prompt: identical?, first differing position, positional agreement)
"""
import argparse, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "cmtp", "ids-prompts")
ASK = "\n\nSummarise the text above in detail, section by section."


def prompts():
    out = {}
    for i in (0, 1):
        t4 = open(os.path.join(P, f"4k-{i}.txt")).read()
        out[f"1k-{i}"] = t4[:3600] + ASK  # about 1k tokens of the 4k document
        out[f"4k-{i}"] = t4 + ASK
        out[f"16k-{i}"] = open(os.path.join(P, f"16k-{i}.txt")).read() + ASK
    return out


def run(url, model, out, gen):
    res = {}
    for name, text in prompts().items():
        body = {"model": model, "prompt": text, "max_tokens": gen, "ignore_eos": True, "temperature": 0.0, "return_token_ids": True}
        req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            ch = json.load(r)["choices"][0]
        res[name] = ch.get("token_ids")
        print(f"{name}: {len(res[name] or [])} tokens", flush=True)
    json.dump(res, open(out, "w"))


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same_all = True
    for k in A:
        x, y = A[k] or [], B.get(k) or []
        n = min(len(x), len(y))
        first = next((i for i in range(n) if x[i] != y[i]), None)
        agree = sum(x[i] == y[i] for i in range(n)) / max(n, 1)
        same = x == y and n > 0
        same_all &= same
        print(f"{k}: {'IDENTICAL' if same else 'differ'} first_diff={first if first is not None else '-'} positional_agreement={agree:.3f} n={n}")
    sys.exit(0 if same_all else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--model", default="intarsia"); r.add_argument("--out", required=True); r.add_argument("--gen", type=int, default=256)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    run(a.url, a.model, a.out, a.gen) if a.cmd == "run" else compare(a.a, a.b)
