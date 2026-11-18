#!/usr/bin/env python3
"""mgdnvsoak: N concurrent streamed requests (a rotation of 4 cells), one JSON line per request + one AGG line (aggregate tok/s = sum tokens / wall of the whole wave).
  conc.py <port> <cells_dir> <N> <tag> <maxtok> [--temp T --seed S]   (greedy unless --temp; seed S+i per request)"""
import hashlib, json, sys, threading, time, urllib.request, pathlib
port, cdir, n, tag, mx = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
temp = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 0
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else None
names = ["chat-short", "code-gen", "reason", "agent-edit"]
res = [None] * n


def one(i):
    cell = json.load(open(cdir / f"{names[i % 4]}.json"))
    msgs = json.loads(json.dumps(cell["messages"]))
    first = next(m for m in msgs if m["role"] == "user")
    first["content"] = f"[{tag}-{i}] " + first["content"]
    body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": min(mx, cell["max_tokens"]), "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}
    if temp > 0:
        body["top_p"] = 0.95
    if seed is not None:
        body["seed"] = seed + i
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    text = []
    usage = None
    for line in urllib.request.urlopen(req, timeout=1800):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        ch = d.get("choices") or []
        c = ch[0].get("delta", {}).get("content") if ch else None
        if c:
            text.append(c)
    s = "".join(text)
    res[i] = {"i": i, "cell": names[i % 4], "toks": (usage or {}).get("completion_tokens", 0), "sha": hashlib.sha1(s.encode()).hexdigest()[:12]}


t0 = time.time()
th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
[t.start() for t in th]
[t.join() for t in th]
wall = time.time() - t0
for r in res:
    print(json.dumps(r))
tot = sum(r["toks"] for r in res if r)
print("AGG " + json.dumps({"tag": tag, "n": n, "toks": tot, "wall": round(wall, 2), "agg_tps": round(tot / wall, 2)}), flush=True)
