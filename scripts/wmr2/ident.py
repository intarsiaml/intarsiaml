#!/usr/bin/env python3
"""Lane wmr: served greedy identity across stream counts.

Sends 16 distinct prompts (temperature 0, thinking off, fixed max_tokens) to one running `intarsia serve --max-batch 16` and compares each prompt's
completion text when it is sent alone (1 stream, sequential) with the same prompt sent as one of 4, 8 and 16 concurrent requests. A second solo
pass checks run-to-run determinism. Exit 0 always; the verdict is the JSON and the last lines of stdout (the caller greps them).
"""
import argparse, json, threading, time, urllib.request

TOPICS = [
    "the history of the printing press", "how a transformer language model computes attention", "the water cycle", "binary search trees",
    "the causes of the French Revolution", "how vaccines train the immune system", "the rules of chess", "plate tectonics",
    "how a CPU cache works", "the life cycle of a star", "the basics of double-entry bookkeeping", "how to bake sourdough bread",
    "the TCP three-way handshake", "the theory of evolution by natural selection", "how lithium-ion batteries work", "the Roman road network",
]
PAD = ("Context notes: the following paragraph is background and may be ignored unless relevant. " +
       "Quarterly figures were reviewed on Tuesday, the committee noted several open items, and the schedule slipped by two days. ")


def prompt(i):
    # prompt i carries i*40 repetitions' worth of padding, so the 16 rows sit at different positions and sequence lengths
    pad = (PAD * (i * 3)) if i % 2 else ""
    return f"{pad}Write a clear, factual explanation of {TOPICS[i]} in about 150 words."


def ask(url, model, i, max_tokens, out, key):
    body = {"model": model, "messages": [{"role": "user", "content": prompt(i)}], "temperature": 0, "max_tokens": max_tokens,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    out[key] = {"text": j["choices"][0]["message"]["content"], "finish": j["choices"][0].get("finish_reason"), "usage": j.get("usage")}


def run_group(url, model, idx, max_tokens, res, tag):
    ths = [threading.Thread(target=ask, args=(url, model, i, max_tokens, res, (tag, i))) for i in idx]
    [t.start() for t in ths]
    [t.join() for t in ths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    a = ap.parse_args()
    n = len(TOPICS)
    res = {}
    t0 = time.time()
    for i in range(n):  # solo pass 1
        ask(a.url, a.model, i, a.max_tokens, res, ("solo", i))
    for i in range(n):  # solo pass 2 (determinism)
        ask(a.url, a.model, i, a.max_tokens, res, ("solo2", i))
    for width in (4, 8, 16):
        for g in range(0, n, width):
            run_group(a.url, a.model, range(g, g + width), a.max_tokens, res, f"c{width}")
    summary = {}
    for tag in ("solo2", "c4", "c8", "c16"):
        same, diffs = 0, []
        for i in range(n):
            s, c = res[("solo", i)]["text"], res[(tag, i)]["text"]
            if s == c:
                same += 1
            else:
                k = next((p for p in range(min(len(s), len(c))) if s[p] != c[p]), min(len(s), len(c)))
                diffs.append({"prompt": i, "first_diff_char": k, "len_solo": len(s), "len_other": len(c)})
        summary[tag] = {"identical": same, "of": n, "diffs": diffs}
    toks = sum((res[("solo", i)]["usage"] or {}).get("completion_tokens", 0) for i in range(n))
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "completion_tokens_solo": toks, "seconds": round(time.time() - t0, 1),
                   "results": {f"{k[0]}:{k[1]}": v for k, v in res.items()}}, f, indent=1)
    for tag, s in summary.items():
        print(f"IDENT {tag} vs solo: {s['identical']}/{s['of']} identical; diffs {s['diffs'][:6]}")
    print(f"IDENT done {round(time.time() - t0, 1)} s, {toks} solo completion tokens")


main()
