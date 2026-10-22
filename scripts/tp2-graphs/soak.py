#!/usr/bin/env python3
"""Lane G1c follow-up: a mixed-request soak of an intarsia `serve` (stdlib only; runs on the Spark next to the server).

    soak.py --url http://127.0.0.1:8090 --seconds 540 --corpus ~/intarsia-tp2/prompts/corpus_docs.txt

Two client threads, each looping until the deadline over a weighted mix of:
  short    a one-line question, max_tokens 1-64
  long     2k-54k characters of the corpus at a random offset (about 0.5k-16k tokens), max_tokens 8-128
  reuse    one fixed 9k-character prefix plus a varying question (prefix-reuse hits: usage.cached_tokens > 0)
  turn     a two-turn chat continuing a previous answer
  stream   a short or long request with "stream": true (SSE; the [DONE] marker is required)
Every request is greedy (temperature 0). A fixed probe request runs at the start and again at the end: its text must be
identical. At the end /health must answer and one last short request must succeed.
Any HTTP error, exception, timeout, empty answer, missing usage, or a failed final check is an error. Prints one JSON
summary line (SOAK {...}) and exits 1 on any error.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--seconds", type=int, default=540)
ap.add_argument("--corpus", required=True)
ap.add_argument("--model", default="intarsia-35b")
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
corpus = open(os.path.expanduser(a.corpus)).read()
REUSE = corpus[1000:10000]
QUESTIONS = ["Summarise the text in one line.", "Name three key terms from it.", "What is the main claim?",
             "Give a title for it.", "List two open questions it raises."]
lock = threading.Lock()
stats = {"requests": 0, "errors": 0, "by_kind": {}, "cached_hits": 0, "prompt_tokens_max": 0, "error_samples": []}


def post(body, timeout=300):
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {raw[:200]}")
        return raw


def chat(messages, max_tokens, stream=False):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        raw = post(body)
        text, usage, done = "", None, False
        for line in raw.splitlines():
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d.strip() == "[DONE]":
                done = True
                continue
            j = json.loads(d)
            for c in j.get("choices") or []:
                text += (c.get("delta") or {}).get("content") or ""
            usage = j.get("usage") or usage
        if not done:
            raise RuntimeError("stream without [DONE]")
        return text, usage
    j = json.loads(post(body))
    return j["choices"][0]["message"]["content"] or "", j.get("usage")


def one(rng, kind):
    if kind == "short":
        return chat([{"role": "user", "content": rng.choice(["Name a colour.", "What is 7*8?", "Say hello.",
                                                              "Name a planet.", "Give a synonym for fast."])}],
                    rng.randint(1, 64))
    if kind in ("long", "stream"):
        n = rng.choice([2000, 6000, 15000, 30000, 54000]) if kind == "long" else rng.choice([200, 4000])
        off = rng.randint(0, max(0, len(corpus) - n - 1))
        return chat([{"role": "user", "content": corpus[off:off + n] + "\n" + rng.choice(QUESTIONS)}],
                    rng.randint(8, 128), stream=(kind == "stream"))
    if kind == "reuse":
        return chat([{"role": "user", "content": REUSE + "\n" + rng.choice(QUESTIONS)}], rng.randint(4, 48))
    if kind == "turn":
        q = [{"role": "user", "content": corpus[20000:24000] + "\nWhat is this about?"}]
        t1, _ = chat(q, 24)
        q += [{"role": "assistant", "content": t1}, {"role": "user", "content": rng.choice(QUESTIONS)}]
        return chat(q, rng.randint(8, 64))
    raise ValueError(kind)


KINDS = ["short"] * 4 + ["long"] * 2 + ["reuse"] * 3 + ["turn"] + ["stream"] * 2


def worker(seed, deadline):
    rng = random.Random(seed)
    while time.time() < deadline:
        kind = rng.choice(KINDS)
        t0 = time.time()
        try:
            text, usage = one(rng, kind)
            if not text.strip():
                raise RuntimeError("empty answer")
            if not usage or not usage.get("completion_tokens"):
                raise RuntimeError(f"no usage: {usage}")
            ok = True
        except Exception as e:  # noqa: BLE001 - any failure is a soak error
            ok, usage = False, None
            err = f"{kind}: {type(e).__name__}: {str(e)[:200]}"
        with lock:
            stats["requests"] += 1
            k = stats["by_kind"].setdefault(kind, {"n": 0, "err": 0, "sec": 0.0})
            k["n"] += 1
            k["sec"] += time.time() - t0
            if ok:
                cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                stats["cached_hits"] += cached > 0
                stats["prompt_tokens_max"] = max(stats["prompt_tokens_max"], usage.get("prompt_tokens") or 0)
            else:
                stats["errors"] += 1
                k["err"] += 1
                if len(stats["error_samples"]) < 10:
                    stats["error_samples"].append(err)


PROBE = [{"role": "user", "content": corpus[:3000] + "\nSummarise in one line."}]
start = time.time()
probe0, _ = chat(PROBE, 32)
deadline = start + a.seconds
ts = [threading.Thread(target=worker, args=(i, deadline)) for i in range(a.threads)]
for t in ts:
    t.start()
for t in ts:
    t.join()
final = {}
try:
    with urllib.request.urlopen(a.url + "/health", timeout=30) as r:
        final["health"] = r.status
    final["last_short"] = chat([{"role": "user", "content": "Name a colour."}], 4)[0]
    probe1, _ = chat(PROBE, 32)
    final["probe_identical"] = probe0 == probe1
except Exception as e:  # noqa: BLE001
    final["error"] = f"{type(e).__name__}: {e}"
stats["final"] = final
stats["seconds"] = round(time.time() - start)
bad = stats["errors"] > 0 or final.get("health") != 200 or not final.get("last_short") or not final.get("probe_identical")
stats["verdict"] = "FAIL" if bad else "PASS"
print("SOAK " + json.dumps(stats), flush=True)
raise SystemExit(1 if bad else 0)
