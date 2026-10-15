#!/usr/bin/env python3
"""serve_check.py <port> <prompt file> <out json>: one streaming and one non-streaming chat completion (greedy,
thinking off) against a running `intarsia serve`, plus the spec counters from /metrics."""
import json, sys, urllib.request
port, pfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://127.0.0.1:{port}"
prompt = open(pfile).read()
def body(stream):
    return json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
                       "temperature": 0, "stream": stream, "chat_template_kwargs": {"enable_thinking": False}}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", data=body(True), headers={"Content-Type": "application/json"})
chunks, text, finish = 0, "", None
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        for c in ev.get("choices", []):
            d = c.get("delta", {}).get("content")
            if d:
                text += d; chunks += 1
            finish = c.get("finish_reason") or finish
req = urllib.request.Request(url + "/v1/chat/completions", data=body(False), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    full = json.loads(r.read())
metrics = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
spec = {l.split()[0]: float(l.split()[1]) for l in metrics.splitlines() if l.startswith("intarsia_spec_")}
json.dump({"stream_text": text, "stream_chunks": chunks, "stream_finish": finish,
           "full_text": full["choices"][0]["message"]["content"], "full_finish": full["choices"][0]["finish_reason"],
           "usage": full.get("usage"), "metrics_spec": spec}, open(out, "w"), indent=1)
print(out, "chunks", chunks, "finish", finish, "spec", spec)
