#!/usr/bin/env python3
"""Lane dsubset: generate the COLLECT prompts (prompts.py) on an intarsia server and keep the generated token ids.

collect.py <base_url> <model> <out.jsonl> [--conc 8] [--max-tokens 1024]

Chat completions with `return_token_ids` (the intarsia extension), temperature 0.6 / top_p 0.95 / top_k 20 (the 35B's
generation_config) with a fixed seed per prompt, thinking alternating on / off by prompt index (tool prompts run in both modes),
natural stops (no ignore_eos). One JSON line per request: category, index, thinking, the generated ids and the text.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__file__))
from prompts import COLLECT, TOOLS_ALL  # noqa: E402


def jobs():
    for cat, ps in COLLECT.items():
        for i, p in enumerate(ps):
            modes = [True, False] if cat == "tool" else [i % 2 == 0]
            for think in modes:
                yield cat, i, think, p


def one(base, model, cat, i, think, p, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": p}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 7000 + 97 * i + (1 if think else 0),
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if cat == "tool":
        body["tools"] = TOOLS_ALL
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    ch = r["choices"][0]
    ids = ch.get("token_ids") or ch.get("message", {}).get("token_ids")
    msg = ch.get("message", {})
    text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "") + "\n" + json.dumps(msg.get("tool_calls") or [], ensure_ascii=False)
    return {"cat": cat, "i": i, "thinking": think, "ids": ids, "text": text, "finish": ch.get("finish_reason"),
            "usage": r.get("usage"), "secs": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    todo = list(jobs())
    lock = threading.Lock()
    out = open(a.out, "w")
    n_tok = [0, 0]
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                j = todo.pop(0)
            try:
                rec = one(a.base, a.model, *j, a.max_tokens)
            except Exception as e:  # keep going; the record says what failed
                rec = {"cat": j[0], "i": j[1], "thinking": j[2], "error": repr(e)}
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_tok[0] += len(rec.get("ids") or [])
                n_tok[1] += 1
                if n_tok[1] % 20 == 0:
                    print(f"{n_tok[1]} requests, {n_tok[0]} ids, {time.time() - t0:.0f} s", flush=True)

    th = [threading.Thread(target=worker) for _ in range(a.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f"done: {n_tok[1]} requests, {n_tok[0]} ids, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
