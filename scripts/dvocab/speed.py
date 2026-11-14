#!/usr/bin/env python3
"""Lane dvocab (copied from lane mtp-sampled, plus `--set cjk`): decode speed of an intarsia server at 1, 2, 4 concurrent streams.

localeval's `speed` command is greedy only, so this uses localeval's streaming Client directly (as scripts/ab35b/sampled.py
does): the same timing definitions (TTFT, decode tok/s = (completion_tokens - 1) / decode window, itl_ms_p50 / p99). It also
keeps the chunk arrival times localeval measures, for the raw gap between chunks (a verified run arrives as a burst, which
localeval's per-token ITL spreads over the burst's tokens).

Every request is a chat completion at the given temperature (top_k / top_p: the server's generation_config defaults), a fixed
seed per prompt, thinking left at the server default, exactly --gen tokens (ignore_eos + min_tokens, like `speed`). With a
fixed seed the text is the same on every arm, so arms are compared on identical token streams; the sha256 of each reply is
recorded to check that. The /metrics spec counters around each stream count give the MTP acceptance.

  uv run --project /opt/code/deepeval python scripts/mtp-sampled/speed.py --url URL --model guidance --arm NAME --streams 1,2 --out DIR --metrics URL/metrics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import sys
import threading
import urllib.request

import localeval.client as lc
from localeval.client import Client, model_info
from localeval.config import Target

PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the sieve of Eratosthenes and why it is faster than trial division.",
    "Summarise the causes of the French Revolution in five paragraphs.",
    "Write a Rust function that merges two sorted vectors, with unit tests.",
    "Translate into German and explain the grammar: The weather is nice today, so we will go for a long walk by the river.",
    "A train leaves at 9:40 and arrives at 13:05, with a 25 minute stop on the way. What is its average speed over 310 km?",
    "Draft a polite email declining a meeting invitation and proposing two other times.",
]
# lane dvocab: a Chinese set (8 prompts; the first two are lane mtp2s's), a code set and a mixed set, to see what a reduced draft
# head (`--mtp-draft-vocab`) does to acceptance outside English: the 35B's vocabulary has no Chinese token below id ~85k (see the results doc)
CJK = [
    "用中文详细解释哈希表如何处理冲突，并用Python举一个例子。",
    "写一个关于灯塔看守人在瓶子里发现一封信的短篇故事。",
    "请解释埃拉托斯特尼筛法，以及它为什么比试除法更快。",
    "用五段话概括法国大革命的起因。",
    "用Rust写一个合并两个有序向量的函数，并附上单元测试。",
    "把这句话翻译成德语并解释语法：今天天气很好，所以我们要沿着河边走很久。",
    "一列火车9:40出发，13:05到达，中途停靠25分钟。全程310公里，它的平均速度是多少？请一步一步思考。",
    "写一封礼貌的邮件，婉拒一次会议邀请，并提出另外两个时间。",
]
CODE = [
    "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat, with pytest tests.",
    "Implement an LRU cache in Rust using a HashMap and a doubly linked list of indices, with unit tests.",
    "Write a SQL query that returns the top 3 customers by revenue per country from tables customers, orders and order_items.",
    "Write a TypeScript function that debounces an async function and cancels stale calls, with a usage example.",
    "Write a bash script that watches a directory and gzips any file larger than 10 MB, logging each action.",
    "Implement a thread-safe bounded blocking queue in C++17 with condition variables.",
    "Write a Go HTTP handler that streams a large JSON array from a database cursor without buffering it all.",
    "Refactor this Python into small functions and explain each change:\n\ndef f(d):\n    r = []\n    for k in d:\n        if d[k] > 0:\n            r.append((k, d[k] * 2))\n    r.sort(key=lambda x: x[1])\n    return r\n",
]
# 2 streams run consecutive pairs: (en, cjk) (code, json) (en, code) (cjk, en)
MIXED = [PROMPTS[0], CJK[1], CODE[0], 'Return only a JSON object with keys "city", "country" and "population" for the three largest cities in Japan, as an array of objects.',
         PROMPTS[3], CODE[2], CJK[6], PROMPTS[7]]

_tls = threading.local()
_itl_stats = lc.itl_stats


def _capture(marks, n_tokens):
    _tls.marks = list(marks)
    return _itl_stats(marks, n_tokens)


lc.itl_stats = _capture


def scrape(url):
    out = {}
    if not url:
        return out
    try:
        txt = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"^(intarsia_spec_\w+|intarsia_batch_decode_steps_total|intarsia_batch_decode_rows_total)\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--streams", default="1,2")
    ap.add_argument("--gen", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--metrics", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", default="default", choices=["default", "on", "off"], help="off: no think block, so the forced tokens are the answer (Chinese stays Chinese)")
    ap.add_argument("--set", default="en", choices=["en", "cjk", "code", "mixed", "long"])
    a = ap.parse_args()
    global PROMPTS
    if a.set == "cjk":
        PROMPTS = CJK
    elif a.set == "code":
        PROMPTS = CODE
    elif a.set == "mixed":
        PROMPTS = MIXED
    elif a.set == "long":
        # lane dvocab: ~3k-token prompts (a piece of this repo's own docs, read at run time) and a question about it
        repo = pathlib.Path(__file__).resolve().parents[2]
        docs = ["docs/ARCHITECTURE.md", "docs/adding-a-model.md", "docs/quant-moe.md", "docs/benchmarking.md"]
        PROMPTS = [f"Read this document and then explain its main ideas in your own words, section by section.\n\n{(repo / d).read_text()[:12000]}" for d in docs]
    probe = Target(name=a.arm, url=a.url, model=a.model)
    info = model_info(probe)
    if not info.get("reachable"):
        raise SystemExit(f"unreachable: {info}")

    def one(i, out):
        t = Target(name=a.arm, url=a.url, model=a.model, thinking=a.thinking, extra_body={"seed": 2000 + i})
        r = Client(t).chat([{"role": "user", "content": PROMPTS[i]}], max_tokens=a.gen, temperature=a.temperature, force_tokens=True)
        marks = getattr(_tls, "marks", [])
        gaps = [(marks[k][0] - marks[k - 1][0]) * 1e3 for k in range(1, len(marks))]
        out[i] = {
            "ok": r.ok, "error": r.error, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "ttft_s": r.ttft_s, "decode_tok_s": r.decode_tok_s, "itl_ms_p50": r.itl_ms_p50, "itl_ms_p99": r.itl_ms_p99,
            "chunks": len(marks), "gap_ms_p50": q(gaps, 0.5), "gap_ms_p99": q(gaps, 0.99), "gap_ms_max": max(gaps, default=None),
            "sha256": hashlib.sha256(r.raw_output.encode()).hexdigest()[:16], "dropped": r.dropped_params,
            # what the forced tokens are: share of CJK characters (and of all non-space characters) in the whole reply, think block included
            "cjk_chars": sum(1 for c in r.raw_output if "\u4e00" <= c <= "\u9fff"), "chars": len(r.raw_output), "reasoning_chars": len(r.reasoning or ""),
            "head": r.raw_output[:100],
        }

    # warm-up (kernels, graphs, the first-request costs), not measured
    w = [None] * len(PROMPTS)
    one(0, w)
    rows = []
    for s in [int(x) for x in a.streams.split(",")]:
        m0 = scrape(a.metrics)
        out = [None] * len(PROMPTS)
        groups = []
        for lo in range(0, len(PROMPTS), s):
            idx = list(range(lo, min(lo + s, len(PROMPTS))))
            ts = [threading.Thread(target=one, args=(i, out)) for i in idx]
            [t.start() for t in ts]
            [t.join() for t in ts]
            if all(out[i]["ok"] for i in idx):
                groups.append(sum(out[i]["decode_tok_s"] for i in idx))
        m1 = scrape(a.metrics)
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        ok = [o for o in out if o and o["ok"]]
        prop, acc, steps = d.get("intarsia_spec_proposed_tokens_total", 0), d.get("intarsia_spec_accepted_tokens_total", 0), d.get("intarsia_spec_steps_total", 0)
        row = {
            "streams": s, "requests": len(ok), "errors": [o["error"] for o in out if o and not o["ok"]],
            "decode_tok_s_per_stream": {"median": round(statistics.median([o["decode_tok_s"] for o in ok]), 2), "min": round(min(o["decode_tok_s"] for o in ok), 2),
                                        "max": round(max(o["decode_tok_s"] for o in ok), 2)},
            "decode_tok_s_combined": {"median": round(statistics.median(groups), 2), "min": round(min(groups), 2), "max": round(max(groups), 2)},
            "itl_ms_p99": round(statistics.median([o["itl_ms_p99"] for o in ok]), 2), "itl_ms_p99_max": round(max(o["itl_ms_p99"] for o in ok), 2),
            "gap_ms_p99": round(statistics.median([o["gap_ms_p99"] for o in ok]), 2), "gap_ms_p99_max": round(max(o["gap_ms_p99"] for o in ok), 2),
            "gap_ms_max": round(max(o["gap_ms_max"] for o in ok), 2),
            "ttft_s": round(statistics.median([o["ttft_s"] for o in ok]), 3),
            "cjk_share": round(sum(o["cjk_chars"] for o in ok) / max(1, sum(o["chars"] for o in ok)), 3),
            "mtp": {"verify_passes": steps, "drafted": prop, "accepted": acc, "accept_rate": round(acc / prop, 3) if prop else None,
                    "tokens_per_pass": round((acc + steps) / steps, 3) if steps else None, "delta": d},
            "each": out,
        }
        rows.append(row)
        print(f"[speed] {a.arm} streams {s}: per stream {row['decode_tok_s_per_stream']['median']} tok/s ({row['decode_tok_s_per_stream']['min']}-{row['decode_tok_s_per_stream']['max']}), "
              f"combined {row['decode_tok_s_combined']['median']}, ITL p99 {row['itl_ms_p99']} ms, chunk gap p99 {row['gap_ms_p99']} ms (max {row['gap_ms_max']}), "
              f"CJK share {row['cjk_share']} MTP passes {steps:.0f} accept {row['mtp']['accept_rate']} tokens/pass {row['mtp']['tokens_per_pass']}", flush=True)
    p = pathlib.Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"speed-{a.arm}-{a.set}{'-nt' if a.thinking == 'off' else ''}.json").write_text(json.dumps({"arm": a.arm, "set": a.set, "thinking": a.thinking, "url": a.url, "temperature": a.temperature, "gen": a.gen, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
