#!/usr/bin/env python3
"""Lane hfp8: greedy outputs of N concurrent, different ~1k-token prompts (temperature 0, no nonce), written as JSONL, so two
servers (main and the lane's build) can be compared byte for byte:
  python3 greedy.py --url http://tatooine:8090/v1 --model qwen36-35b --docs <repo>/docs --streams 16 --gen 192 --out out.jsonl
The prompts are cut from the repo's larger *.md files exactly like scripts/hprof/drive.py's `div<N>` phase. All N requests are in flight
together, so the decode steps run at N rows (batched fp8 projections at M = N)."""
import argparse, glob, http.client, json, os, threading, urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--docs", required=True)
ap.add_argument("--streams", type=int, default=16)
ap.add_argument("--gen", type=int, default=192)
ap.add_argument("--chars", type=int, default=3500)
ap.add_argument("--out", required=True)
a = ap.parse_args()
u = urllib.parse.urlsplit(a.url.rstrip("/") + "/chat/completions")

files = sorted(f for f in glob.glob(os.path.join(a.docs, "**", "*.md"), recursive=True) if os.path.getsize(f) > 20000)
prompts = []
for f in files:
    t = open(f, errors="replace").read()
    s = (len(t) // 3) % max(1, len(t) - a.chars)
    prompts.append(t[s:s + a.chars] + "\n\nSummarize the text above and continue it in detail.")
assert len(prompts) >= a.streams, f"only {len(prompts)} docs with > 20 kB"
prompts = prompts[: a.streams]
res = [None] * a.streams


def one(i):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompts[i]}], "max_tokens": a.gen, "temperature": 0.0,
            "ignore_eos": True, "min_tokens": a.gen, "chat_template_kwargs": {"enable_thinking": False}}
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    c.request("POST", u.path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    res[i] = {"stream": i, "status": r.status, "text": d["choices"][0]["message"]["content"] if r.status == 200 else None,
              "usage": d.get("usage"), "err": None if r.status == 200 else str(d)[:300]}


th = [threading.Thread(target=one, args=(i,)) for i in range(a.streams)]
[t.start() for t in th]
[t.join() for t in th]
with open(a.out, "w") as fh:
    for r in res:
        fh.write(json.dumps(r) + "\n")
print(json.dumps({"streams": a.streams, "ok": sum(1 for r in res if r and r["status"] == 200),
                  "chars": sum(len(r["text"] or "") for r in res if r)}))
