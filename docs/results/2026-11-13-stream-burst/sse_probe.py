#!/usr/bin/env python3
"""Arrival-time probe: POST a streaming chat request over a raw socket, stamp every SSE event as it is read.
usage: sse_probe.py PORT LABEL [--think] [--tools] [--max N] [--seed S]
Prints the inter-event gap histogram and writes <LABEL>.text (concatenated deltas) for a byte-identity diff."""
import json, socket, sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("port", type=int); ap.add_argument("label")
ap.add_argument("--think", action="store_true"); ap.add_argument("--tools", action="store_true")
ap.add_argument("--max", type=int, default=400); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--host", default="127.0.0.1")
a = ap.parse_args()
body = {"model": "x", "stream": True, "max_tokens": a.max, "seed": a.seed, "temperature": 0.7, "top_p": 0.95, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Write a long, detailed explanation of how a hash map works, with an example."}],
        "chat_template_kwargs": {"enable_thinking": a.think}}
if not a.tools:
    body["ignore_eos"] = True; body["min_tokens"] = a.max
if a.tools:
    body["messages"] = [{"role": "user", "content": "What is the weather in Paris and in Tokyo? Use the tool for each."}]
    body["tools"] = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
data = json.dumps(body).encode()
s = socket.create_connection((a.host, a.port)); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(data) + data)
t0 = time.perf_counter(); buf = b""; stamps = []; marks = []; usage_n = None; nevents = 0; text = []; reads = 0; done = False
while not done:
    c = s.recv(65536)
    if not c: break
    reads += 1; now = (time.perf_counter() - t0) * 1e3; buf += c
    while b"\n\n" in buf:
        ev, buf = buf.split(b"\n\n", 1)
        if b"[DONE]" in ev: done = True; break
        for ln in ev.split(b"\n"):
            if ln.startswith(b"data:"):
                try: d = json.loads(ln[5:])
                except Exception: continue
                nevents += 1
                if d.get("usage"): usage_n = d["usage"].get("completion_tokens")
                ch = (d.get("choices") or [{}])[0]; dl = ch.get("delta") or {}
                t = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if dl.get("tool_calls"): t += json.dumps(dl["tool_calls"], sort_keys=True)
                if t: stamps.append(now); text.append(t); marks.append((now / 1e3, len(t)))
json.dump({"marks": marks, "n_tokens": usage_n}, open(a.label + ".marks.json", "w"))
open(a.label + ".text", "w").write("".join(text))
g = sorted(b - a_ for a_, b in zip(stamps, stamps[1:]))
q = lambda p: g[int((len(g) - 1) * p)] if g else float("nan")
edges = [1, 5, 10, 14, 17, 20, 30, 50, 80, 1e9]; h = [0] * len(edges)
for x in g: h[next(i for i, e in enumerate(edges) if x < e)] += 1
print(f"{a.label}: content events={len(stamps)} socket reads={reads} gap ms p50={q(.5):.1f} p90={q(.9):.1f} p99={q(.99):.1f} max={g[-1] if g else 0:.1f}")
print(f"  completion_tokens={usage_n} content chunks={len(stamps)} (localeval weights chunks by chars when these differ)")
print("  hist(<1,<5,<10,<14,<17,<20,<30,<50,<80,>=80)=", h)
