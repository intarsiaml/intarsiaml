#!/usr/bin/env python3
"""style7: first-generated-token top-20 logprobs of one engine for token-id prompts, one request at a time.
usage: firsttok.py --url http://host:port/v1 --model NAME --tokens vllm-tokenize-....json --out OUT.json [--vllm] [--limit N]
Works against vLLM (--vllm: return_tokens_as_token_ids) and intarsia serve (top_logprobs_ids). Output: [{id, top: [[token_id, logprob], ...20], n_prompt}]"""
import argparse, json, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tokens", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--vllm", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--extra", default="{}", help="extra JSON body fields")
a = ap.parse_args()
P = json.load(open(a.tokens))
if a.limit: P = P[:a.limit]
extra = json.loads(a.extra)
res = []
t0 = time.time()
for i, p in enumerate(P):
    body = {"model": a.model, "prompt": p["tokens"], "max_tokens": 1, "temperature": 0, "logprobs": 20, **extra}
    if a.vllm: body["return_tokens_as_token_ids"] = True
    req = urllib.request.Request(a.url + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    lp = r["choices"][0]["logprobs"]
    if a.vllm:
        top = [[int(k.split(":")[1]), v] for k, v in lp["top_logprobs"][0].items()]
        top.sort(key=lambda x: -x[1]); top = top[:20]
        first = int(lp["tokens"][0].split(":")[1])
    else:
        top = [[x[0], x[1]] for x in lp["top_logprobs_ids"][0]]
        first = r["choices"][0].get("token_ids", [None])[0]
    res.append({"id": p["id"], "top": top, "n_prompt": r["usage"]["prompt_tokens"], "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")})
    if (i + 1) % 50 == 0: print(f"{i+1}/{len(P)} {time.time()-t0:.0f}s", file=sys.stderr)
json.dump(res, open(a.out, "w"))
print(f"done {len(res)} -> {a.out} in {time.time()-t0:.0f}s")
