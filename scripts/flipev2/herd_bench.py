#!/usr/bin/env python3
"""flipev (from scripts/herd/herd_bench.py; adds --stagger, --prompts mix): concurrent same-prefix requests against `intarsia serve --max-batch 8 --prefix-store on` (stdlib only).

Prompts are token ids (`/v1/completions` with a list of ids, so the shared prefix is exactly the same ids in every request and the
length is exact), random ids in [1000, 100000) from a seed: a new seed is a new prefix, never seen by the server's store.

  herd_bench.py --url http://spark-fa00:8193 --arm A --mode herd8    --seed 1   # 8 at once: one shared prefix + 8 distinct suffixes
  herd_bench.py ...                                  --mode distinct --seed 11  # 8 at once: 8 unrelated prompts (no-regression cell)
  herd_bench.py ...                                  --mode solo     --seed 1   # the herd8 prompts of that seed, one at a time

Every request streams (`stream`, `return_token_ids`, `stream_options.include_usage`): TTFT is the client's time from the barrier to the
first streamed token, `cached_tokens` is the usage chunk's. One JSON line per cell is appended to --out (default herd-results.jsonl).
Temperature 0, `ignore_eos`, a fixed `max_tokens`: the reply ids of one prompt must not depend on the arm.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import threading
import time
import urllib.parse


def make_prompts(mode: str, seed: int, n: int, shared: int, own: int, distinct_len: int, short_len: int = 1000):
    rng = random.Random(seed * 7919 + 13)
    ids = lambda k: [rng.randrange(1000, 100000) for _ in range(k)]  # noqa: E731
    if mode == "distinct":
        return [ids(distinct_len) for _ in range(n)]
    if mode == "mix":  # n shared-prefix requests + 2 unrelated distinct_len prompts, arrival order S U S S U S ...
        head = ids(shared)
        sh = [head + ids(own) for _ in range(n)]
        un = [ids(distinct_len) for _ in range(2)]
        return [sh[0], un[0]] + sh[1:3] + [un[1]] + sh[3:]
    if mode == "hs":  # n shared-prefix requests + one unrelated short_len prompt LAST (released at --short-at)
        head = ids(shared)
        sh = [head + ids(own) for _ in range(n)]
        return sh + [ids(short_len)]
    head = ids(shared)
    return [head + ids(own) for _ in range(n)]


def request(url: str, model: str, prompt: list[int], max_tokens: int, barrier: threading.Barrier | None, delay: float = 0.0):
    u = urllib.parse.urlparse(url)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True})
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    res = {"send_s": 0.0, "ok": False, "ids": [], "ttft_s": None, "end_s": None, "prompt_tokens": None, "cached_tokens": None, "error": None}
    try:
        conn.connect()
        if barrier is not None:
            barrier.wait()
        tb = time.perf_counter()  # the barrier release: every request of a cell is released within about a millisecond
        if delay:
            time.sleep(delay)
        t0 = time.perf_counter()
        res["send_s"] = t0 - tb
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
        res["end_s"] = time.perf_counter() - t0
        res["ok"] = res["ttft_s"] is not None
        if not res["ok"]:
            res["error"] = "no token streamed"
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        conn.close()
    return res


def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="guidance")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--visit", default="")
    ap.add_argument("--mode", choices=["herd8", "distinct", "solo", "mix", "hs"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--shared", type=int, default=16448, help="shared prefix tokens (16384 + 64: just past a store grid point)")
    ap.add_argument("--own", type=int, default=200)
    ap.add_argument("--distinct-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--stagger", type=float, default=0.0, help="request i is released i*stagger s after the barrier")
    ap.add_argument("--prompts", choices=["herd", "mix", "hs"], default="herd", help="solo mode: which prompt set to replay one at a time")
    ap.add_argument("--arrival", default="", help='comma list of request indices in arrival order, released i*stagger apart (e.g. "1,4,0,2,3,5": mix with the unrelated pair first); default natural order')
    ap.add_argument("--short-len", type=int, default=1000)
    ap.add_argument("--short-at", type=float, default=2.0, help="mode hs: release time of the last (short, unrelated) prompt")
    ap.add_argument("--out", default="herd-results.jsonl")
    a = ap.parse_args()

    kind = "distinct" if a.mode == "distinct" else "mix" if a.mode == "mix" or (a.mode == "solo" and a.prompts == "mix") else "hs" if a.mode == "hs" or (a.mode == "solo" and a.prompts == "hs") else "herd"
    prompts = make_prompts(kind, a.seed, a.n, a.shared, a.own, a.distinct_len, a.short_len)
    arr = [int(x) for x in a.arrival.split(",")] if a.arrival else []
    t_wall = time.perf_counter()
    if a.mode == "solo":
        rows = [request(a.url, a.model, p, a.max_tokens, None) for p in prompts]
    else:
        barrier = threading.Barrier(len(prompts))
        rows = [None] * len(prompts)

        def work(i):
            pos = arr.index(i) if arr else i
            dl = a.short_at if (a.mode == "hs" and i == len(prompts) - 1) else pos * a.stagger
            rows[i] = request(a.url, a.model, prompts[i], a.max_tokens, barrier, dl)

        ths = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    wall = time.perf_counter() - t_wall
    ok = [r for r in rows if r["ok"]]
    ttfts = [r["ttft_s"] for r in ok]
    out = {
        "arm": a.arm, "visit": a.visit, "mode": a.mode, "seed": a.seed, "n": len(prompts), "ok": len(ok),
        "prompt_tokens": [r["prompt_tokens"] for r in rows], "cached_tokens": [r["cached_tokens"] for r in rows],
        "ttft_s": [None if r["ttft_s"] is None else round(r["ttft_s"], 3) for r in rows],
        "end_s": [None if r["end_s"] is None else round(r["end_s"], 3) for r in rows],
        "send_s": [round(r["send_s"], 3) for r in rows],
        "abs_end_s": [None if r["end_s"] is None else round(r["send_s"] + r["end_s"], 3) for r in rows],
        "stagger": a.stagger, "prompts": a.prompts, "arrival": a.arrival,
        "ttft_sum_s": round(sum(ttfts), 3) if ttfts else None,
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_last_s": round(max(ttfts), 3) if ttfts else None,
        "wall_s": round(max(r["send_s"] + r["end_s"] for r in ok), 3) if ok and a.mode != "solo" else round(wall, 3),
        "completion_tokens": [len(r["ids"]) for r in rows],
        "ids": [r["ids"] for r in rows], "errors": [r["error"] for r in rows],
        "args": {"shared": a.shared, "own": a.own, "distinct_len": a.distinct_len, "max_tokens": a.max_tokens},
        "t": time.strftime("%H:%M:%S"),
    }
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    short = {k: out[k] for k in ("arm", "visit", "mode", "seed", "ok", "ttft_sum_s", "ttft_p50_s", "ttft_last_s", "wall_s", "cached_tokens")}
    print(json.dumps(short))


if __name__ == "__main__":
    main()
