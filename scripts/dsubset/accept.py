#!/usr/bin/env python3
"""Lane dsubset: MTP acceptance per script on an intarsia server, one stream (gate 3).

accept.py <base_url> <model> <arm> <out_dir> <metrics_url> [--sets en,code,...] [--max-tokens 384]

For every held-out EVAL set (prompts.py; none of them is a collection prompt) the 8 prompts run one after another as greedy chat
completions, thinking off (the reply is the language itself; a think block would be English), natural stops, the tool set with
the `tools` array. The /metrics spec counters around a set give its acceptance (accepted / proposed drafts) and tokens per verify
pass; the token ids (`return_token_ids`) give the share of the reply's tokens that the arm's draft head can hold at all, which the
caller computes against a subset file. Writes <out_dir>/accept-<arm>.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prompts import EVAL, TOOLS_ALL  # noqa: E402


def scrape(url):
    out = {}
    try:
        txt = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"^(intarsia_spec_\w+)\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def chat(base, model, prompt, max_tokens, tools):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = TOOLS_ALL
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    ch = r["choices"][0]
    return ch.get("token_ids") or [], (r.get("usage") or {}).get("completion_tokens", 0), dt, ch.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("arm")
    ap.add_argument("out")
    ap.add_argument("metrics")
    ap.add_argument("--sets", default="en,code,zh,ru,ar,kj,tool")
    ap.add_argument("--max-tokens", type=int, default=int(__import__("os").environ.get("ACC_MAX_TOKENS", "384")))
    a = ap.parse_args()
    chat(a.base, a.model, EVAL["en"][0], 32, False)  # warm-up
    res = {"arm": a.arm, "sets": {}}
    for s in a.sets.split(","):
        m0 = scrape(a.metrics)
        reps, toks, secs = [], 0, 0.0
        for p in EVAL[s]:
            ids, n, dt, fin = chat(a.base, a.model, p, a.max_tokens, s == "tool")
            reps.append({"ids": ids, "finish": fin})
            toks += n
            secs += dt
        m1 = scrape(a.metrics)
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        prop, acc, steps = d.get("intarsia_spec_proposed_tokens_total", 0), d.get("intarsia_spec_accepted_tokens_total", 0), d.get("intarsia_spec_steps_total", 0)
        row = {"tokens": toks, "secs": round(secs, 2), "tok_s": round(toks / secs, 2) if secs else None, "proposed": prop, "accepted": acc,
               "verify_passes": steps, "accept_rate": round(acc / prop, 4) if prop else None,
               "tokens_per_pass": round((acc + steps) / steps, 3) if steps else None, "replies": reps}
        res["sets"][s] = row
        print(f"[accept] {a.arm} {s}: {toks} tokens, accept {row['accept_rate']}, tokens/pass {row['tokens_per_pass']}, {row['tok_s']} tok/s", flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"accept-{a.arm}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
