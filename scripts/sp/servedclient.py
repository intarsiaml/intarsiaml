#!/usr/bin/env python3
"""Lane sp: a localeval-shaped streaming chat client with spec counters from /metrics around every request.
  servedclient.py <tag> <sizes: 128,1024,...> <reps> [url] [natural dir]
Body as localeval's speed-spec sends it (chat, temperature 0, top_p 1, seed 0, ignore_eos, min_tokens = max_tokens = 256,
enable_thinking false, a fresh `nonce <32 hex> ` in front of the frozen natural-v1 text). Prints one line per request:
tokens, TTFT, decode tok/s (first to last content chunk), spec steps / proposed / accepted deltas, tokens per pass."""
import json, re, sys, time, urllib.request, uuid

tag, sizes, reps = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3])
url = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8294"
nat = sys.argv[5] if len(sys.argv) > 5 else "/home/josef/intarsiaml-sp/sp-out"

def metrics():
    t = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    out = {}
    for k in ("steps", "proposed_tokens", "accepted_tokens", "costgate_off", "costgate_on", "costgate_plain_steps"):
        m = re.search(rf"^intarsia_spec_{k}_total\s+([0-9.e+]+)", t, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out

def one(text):
    body = {"model": "intarsia", "messages": [{"role": "user", "content": f"nonce {uuid.uuid4().hex} " + text}],
            "max_tokens": 256, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "top_p": 1, "seed": 0, "ignore_eos": True, "min_tokens": 256}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m0 = metrics(); t0 = time.time(); first = last = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content"):
                    now = time.time()
                    first = first or now; last = now; n += 1
    m1 = metrics()
    toks = (usage or {}).get("completion_tokens", n)
    dec = (toks - 1) / (last - first) if first and last and last > first else float("nan")
    d = {k: m1[k] - m0[k] for k in m0}
    tpp = (d["accepted_tokens"] + d["steps"]) / d["steps"] if d["steps"] else float("nan")
    mspp = (last - first) / d["steps"] * 1000 if d["steps"] and first and last else float("nan")
    return toks, (first - t0) if first else float("nan"), dec, d, tpp, mspp

one("warm up " * 20)
for s in sizes:
    text = open(f"{nat}/n{s}.txt").read()
    for r in range(reps):
        toks, ttft, dec, d, tpp, mspp = one(text)
        print(f"{tag} n{s} rep{r} tokens={toks} ttft={ttft:.2f}s decode={dec:.1f} tok/s steps={d['steps']:.0f} proposed={d['proposed_tokens']:.0f} accepted={d['accepted_tokens']:.0f} tok/pass={tpp:.2f} ms/pass={mspp:.1f} cg_off={d['costgate_off']:.0f} cg_on={d['costgate_on']:.0f} cg_plain={d['costgate_plain_steps']:.0f}", flush=True)
