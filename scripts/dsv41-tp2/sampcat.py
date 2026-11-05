#!/usr/bin/env python3
"""Lane dsv41/sampled-spec: decode tok/s per text kind at a given temperature, in the shape `catspeed.py` reads.

    sampcat.py URL OUT_DIR TEMPERATURE [--tokens 256] [--seed0 1000] [--dataset FILE ...]

localeval's `quality` sends temperature 0 (deepeval suite.py), so the sampled path never shows in the category runs. This sends the
same 32 requests (localeval's categories-rcp.jsonl, 24 recall / code / prose, plus chat-rcp.jsonl, 8 multi-turn chats) as streamed
chat requests with the given temperature, thinking off, `max_tokens` tokens, and a fixed seed per request (seed0 + index), so two arms
draw from the same random stream. Writes OUT_DIR/result.json (cases with speed.decode_tps and speed.completion_tokens) for
`catspeed.py LABEL=OUT_DIR`. Decode tok/s is (tokens - 1) / (last token's time - first token's time) of the stream.
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["/opt/code/deepeval/datasets/categories-rcp.jsonl", os.path.join(HERE, "chat-rcp.jsonl")]


def items(files):
    for f in files:
        for line in open(f):
            if line.startswith("{"):
                yield json.loads(line)


def one(conn, path, body):
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    t0 = time.time()
    first = last = None
    ntok = 0
    usage = None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            usage = d["usage"]
        ch = d.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"):
            now = time.time()
            first = first or now
            last = now
            ntok += 1
    n = (usage or {}).get("completion_tokens") or ntok
    return {"completion_tokens": n, "ttft_s": (first or t0) - t0,
            "decode_tps": ((n - 1) / (last - first)) if first and last and last > first and n > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("temperature", type=float)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--dataset", action="append")
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    os.makedirs(a.out, exist_ok=True)
    cases = []
    for i, it in enumerate(items(a.dataset or DATASETS)):
        msgs = it.get("messages") or [{"role": "user", "content": it["input"]}]
        body = {"model": "dsv41-intarsia", "messages": msgs, "temperature": a.temperature, "seed": a.seed0 + i, "max_tokens": a.tokens,
                "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        s = one(conn, u.path.rstrip("/") + "/chat/completions", body)
        cases.append({"name": it["name"], "speed": s})
        print(f"{it['name']:28s} T {a.temperature} {s['completion_tokens']:4d} tokens {s['decode_tps'] or 0:6.1f} tok/s ttft {s['ttft_s']:.2f} s", flush=True)
    json.dump({"temperature": a.temperature, "tokens": a.tokens, "seed0": a.seed0, "datasets": [{"name": "categories-rcp+chat-rcp", "cases": cases}]},
              open(os.path.join(a.out, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
