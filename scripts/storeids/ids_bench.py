#!/usr/bin/env python3
"""storeids: greedy ids of cold token-id prompts against one `intarsia serve` (stdlib only).

Prompts: random ids in [1000, 100000) from a seed, exactly as flipev's `herd_bench.py --mode distinct` (same seed = same prompts: seeds 71 / 72 at 8192 tokens
are flipev's pack-phase prompts). A set is `LEN:SEED,SEED:LPSEED`; each seed gives 8 prompts. burst and seq use the first seed list, lp its own seed (so no prompt is sent twice to one server: every reply is a cold prefill). Modes: `burst` (the 8 prompts of a seed released together, as flipev),
`seq` (one at a time), `lp` (non-streaming, `logprobs` 5, 4 tokens, first 4 prompts of LPSEED, one at a time).
One JSON line per request is appended to --out: visit, set, mode, seed, idx, ids, ttft, prompt_tokens, cached_tokens.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import threading
import time
import urllib.parse


def prompts(length: int, seed: int, n: int = 8):
    rng = random.Random(seed * 7919 + 13)
    return [[rng.randrange(1000, 100000) for _ in range(length)] for _ in range(n)]


def stream_req(url, model, prompt, max_tokens, barrier=None):
    u = urllib.parse.urlparse(url)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True})
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    res = {"ok": False, "ids": [], "ttft_s": None, "prompt_tokens": None, "cached_tokens": None, "error": None}
    try:
        conn.connect()
        if barrier is not None:
            barrier.wait()
        t0 = time.perf_counter()
        conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            res["error"] = f"HTTP {r.status}: {r.read()[:300]!r}"
            return res
        buf = b""
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                for line in ev.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    d = line[5:].strip()
                    if d == b"[DONE]":
                        continue
                    j = json.loads(d)
                    for ch in j.get("choices", []):
                        tid = ch.get("token_ids") or (ch.get("delta") or {}).get("token_ids")
                        if tid:
                            if res["ttft_s"] is None:
                                res["ttft_s"] = time.perf_counter() - t0
                            res["ids"].extend(tid)
                    u_ = j.get("usage")
                    if u_:
                        res["prompt_tokens"] = u_.get("prompt_tokens")
                        res["cached_tokens"] = (u_.get("prompt_tokens_details") or {}).get("cached_tokens")
        res["ok"] = res["ttft_s"] is not None
        if not res["ok"]:
            res["error"] = "no token streamed"
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        conn.close()
    return res


def lp_req(url, model, prompt, max_tokens):
    u = urllib.parse.urlparse(url)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "logprobs": 5,
                       "return_token_ids": True})
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    try:
        conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        r = conn.getresponse()
        raw = r.read()
        if r.status != 200:
            return {"ok": False, "error": f"HTTP {r.status}: {raw[:300]!r}"}
        j = json.loads(raw)
        ch = j["choices"][0]
        lp = ch.get("logprobs") or {}
        return {"ok": True, "ids": ch.get("token_ids"), "token_logprobs": lp.get("token_logprobs"),
                "top_logprobs_ids": lp.get("top_logprobs_ids"), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="guidance")
    ap.add_argument("--visit", required=True)
    ap.add_argument("--sets", default="8192:71,72:75;8193:81,82:85")
    ap.add_argument("--modes", default="burst,seq,lp")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    modes = a.modes.split(",")
    out = open(a.out, "a")

    def emit(**kw):
        out.write(json.dumps({"visit": a.visit, **kw}) + "\n")
        out.flush()

    # warm-up (not recorded): two short prompts compile / allocate whatever is lazy
    for p in prompts(512, 900, 2):
        stream_req(a.url, a.model, p, 8)
    for spec in a.sets.split(";"):
        length, seeds, lpseed = spec.split(":")
        length = int(length)
        seeds = [int(s) for s in seeds.split(",")]
        for mode in modes:
            for seed in ([int(lpseed)] if mode == "lp" else seeds):
                ps = prompts(length, seed)
                if mode == "burst":
                    bar = threading.Barrier(len(ps))
                    rs = [None] * len(ps)

                    def run(i):
                        rs[i] = stream_req(a.url, a.model, ps[i], a.max_tokens, bar)

                    th = [threading.Thread(target=run, args=(i,)) for i in range(len(ps))]
                    [t.start() for t in th]
                    [t.join() for t in th]
                elif mode == "seq":
                    rs = [stream_req(a.url, a.model, p, a.max_tokens) for p in ps]
                elif mode == "lp":
                    rs = [lp_req(a.url, a.model, p, 4) for p in ps[:4]]
                else:
                    raise SystemExit(f"unknown mode {mode}")
                for i, r in enumerate(rs):
                    emit(set=length, mode=mode, seed=seed, idx=i, **r)
                bad = sum(1 for r in rs if not r.get("ok"))
                print(f"[{a.visit}] len {length} {mode} seed {seed}: {len(rs)} requests, {bad} failed" + (f" ({rs[0].get('error')})" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
