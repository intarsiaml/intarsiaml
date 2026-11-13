#!/usr/bin/env python3
"""think-len: send the probe inputs to one engine's /v1/completions (token-id prompts, temperature 0, logprobs 20) and save
the generated ids, text and top-20 log-probabilities per position. Works against vLLM (return_tokens_as_token_ids) and intarsia
(top_logprobs_ids). usage: probe.py <base url> <model name> <inputs.json> <out.json> [kinds, default on,off,tf]"""
import json, sys, time, urllib.request

base, model, inp, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
kinds = set((sys.argv[5] if len(sys.argv) > 5 else "on,off,tf").split(","))
items = [x for x in json.load(open(inp)) if x["kind"] in kinds]
res = []
t0 = time.time()
for n, it in enumerate(items):
    body = {"model": model, "prompt": it["ids"], "max_tokens": it["max_tokens"], "temperature": 0, "logprobs": 20,
            "return_tokens_as_token_ids": True}
    r = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    c = d["choices"][0]
    lp = c["logprobs"]
    if "top_logprobs_ids" in lp:  # intarsia: [id, logprob, text]
        tops = [{int(i): l for i, l, _ in row} for row in lp["top_logprobs_ids"]]
        ids = c.get("token_ids")
    else:  # vLLM: keys "token_id:N"
        tops = [{int(k.split(":")[1]): v for k, v in row.items()} for row in lp["top_logprobs"]]
        ids = [int(t.split(":")[1]) for t in lp["tokens"]]
    res.append({"id": it["id"], "kind": it["kind"], "text": c["text"], "ids": ids, "chosen_lp": lp["token_logprobs"], "tops": tops,
                "prompt_tokens": d["usage"]["prompt_tokens"], "finish": c["finish_reason"]})
    if n % 50 == 0:
        print(n, len(items), round(time.time() - t0), "s", flush=True)
json.dump(res, open(out, "w"))
print("done", len(res), round(time.time() - t0), "s")
