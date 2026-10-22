#!/usr/bin/env python3
"""Lane fx12: a fixed, seeded, serial soak of an intarsia `serve` (stdlib only; runs on the Spark next to the server).

    soak100.py --url http://127.0.0.1:8090 --corpus ~/intarsia-tp2/prompts/corpus_docs.txt --out soak-dev.jsonl

Exactly --n requests (default 100), one at a time, all greedy (temperature 0), the same sequence for a given --seed, so two
builds can be compared answer by answer. The mix is built to drive the tensor-parallel Truncate / Snapshot commands:
  reuse    one fixed 9k-character prefix plus a varying question (a prefix hit: usage.cached_tokens > 0)
  repeat   the exact request of an earlier one (the whole prompt is cached bar its last token)
  turn     a two-turn chat that continues an earlier answer
  long     2k-6k characters of the corpus at a seeded offset
  short    a one-line question
Each answer is written to --out as one JSON line (kind, prompt tokens, cached tokens, text). Any HTTP error, empty
answer or missing usage is an error. Prints one JSON summary line (SOAK {...}) and exits 1 on any error.
"""
import argparse
import json
import os
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=12)
ap.add_argument("--model", default="intarsia-35b")
a = ap.parse_args()
corpus = open(os.path.expanduser(a.corpus)).read()
REUSE = corpus[1000:10000]
QUESTIONS = ["Summarise the text in one line.", "Name three key terms from it.", "What is the main claim?",
             "Give a title for it.", "List two open questions it raises."]
SHORT = ["What is the capital of France?", "Name a prime number above 100.", "Say hello in German.",
         "What colour is the sky on a clear day?", "Give one word for happy."]


def chat(messages, max_tokens):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.loads(r.read().decode())
    text = j["choices"][0]["message"]["content"] or ""
    usage = j.get("usage") or {}
    if not text or not usage.get("completion_tokens"):
        raise RuntimeError(f"empty answer or no usage: {j}")
    return text, usage


rng = random.Random(a.seed)
history = []  # (messages, max_tokens) of earlier requests, for repeats
last = None  # (messages, text) of the last answer, for turns
stats = {"requests": 0, "errors": 0, "by_kind": {}, "cached_hits": 0, "prompt_tokens_max": 0, "error_samples": []}
t0 = time.time()
with open(a.out, "w") as out:
    for i in range(a.n):
        kind = ["short", "reuse", "reuse", "repeat", "turn", "long", "reuse", "short", "repeat", "reuse"][i % 10]
        if kind == "repeat" and not history:
            kind = "short"
        if kind == "turn" and last is None:
            kind = "short"
        if kind == "short":
            msgs, mt = [{"role": "user", "content": rng.choice(SHORT)}], rng.randint(8, 48)
        elif kind == "reuse":
            msgs, mt = [{"role": "user", "content": REUSE + "\n\n" + rng.choice(QUESTIONS)}], rng.randint(16, 64)
        elif kind == "long":
            off = rng.randint(0, max(0, len(corpus) - 7000))
            msgs, mt = [{"role": "user", "content": corpus[off:off + rng.randint(2000, 6000)] + "\n\nOne-line summary?"}], 48
        elif kind == "repeat":
            msgs, mt = rng.choice(history)
        else:  # turn
            prev, text = last
            msgs, mt = prev + [{"role": "assistant", "content": text}, {"role": "user", "content": "Shorter, please."}], 32
        try:
            text, usage = chat(msgs, mt)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            stats["cached_hits"] += cached > 0
            stats["prompt_tokens_max"] = max(stats["prompt_tokens_max"], usage.get("prompt_tokens") or 0)
            out.write(json.dumps({"i": i, "kind": kind, "prompt_tokens": usage.get("prompt_tokens"), "cached": cached,
                                  "text": text}) + "\n")
            out.flush()
            if kind != "repeat":
                history.append((msgs, mt))
            if kind in ("short", "reuse", "long"):
                last = (msgs, text)
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            stats["error_samples"].append(f"#{i} {kind}: {e}"[:300])
        stats["requests"] += 1
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
stats["seconds"] = round(time.time() - t0, 1)
print("SOAK " + json.dumps(stats))
raise SystemExit(1 if stats["errors"] else 0)
