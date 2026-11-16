#!/usr/bin/env python3
"""Lane itl2s: what do the forced tokens after the natural end of a reply look like? (the `ignore_eos` + `min_tokens` protocol keeps generating
after <|im_end|>). Sends the mixed set's SQL prompt (thinking off) as a raw completion, 384 tokens, greedy, and prints the ids from the
first <|im_end|> (248046) on: how many are >= the draft vocab (131072) and so can never be drafted.
usage: eos_probe.py <base_url> <model> <out.json>"""
import json, sys, urllib.request
BASE, MODEL, OUT = sys.argv[1:4]
P = "Write a SQL query that returns the top 3 customers by revenue per country from tables customers, orders and order_items."
prompt = f"<|im_start|>user\n{P}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
body = {"model": MODEL, "prompt": prompt, "max_tokens": 384, "temperature": 0, "ignore_eos": True, "return_token_ids": True, "seed": 1}
req = urllib.request.Request(BASE + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
ch = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]
ids = ch["token_ids"]
res = {"n": len(ids), "ids": ids, "text_tail": ch["text"][-200:]}
json.dump(res, open(OUT, "w"))
end = ids.index(248046) if 248046 in ids else None
print("tokens", len(ids), "first <|im_end|> at", end)
if end is not None:
    tail = ids[end:]
    print("tokens from the end on:", len(tail), "ids >= 131072:", sum(1 for t in tail if t >= 131072), "distinct:", sorted(set(tail))[:20])
    print("before the end: ids >= 131072:", sum(1 for t in ids[:end] if t >= 131072), "of", end)
