#!/usr/bin/env python3
"""mcopy: one served cell, streaming, greedy, thinking off. One JSON line per rep.
  cell.py <port> <cell.json> <reps> <tag> [--out DIR]
The first user message gets a '[tag-rep] ' prefix (the same for every arm, so no request is a prefix-store hit of an earlier one and
every arm sees the same prompts). decode tok/s = (completion tokens - 1) / (end - first token), TTFT = request to first token;
sha = sha1 of the whole reply text (arms must agree: greedy, same prompt). The reply goes to DIR/<tag>-<rep>.txt when --out is given."""
import hashlib, json, pathlib, sys, time, urllib.request

port, cell, reps, tag = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3]), sys.argv[4]
out_dir = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
if out_dir:
    out_dir.mkdir(parents=True, exist_ok=True)
name = pathlib.Path(sys.argv[2]).stem

for rep in range(reps):
    msgs = json.loads(json.dumps(cell["messages"]))
    first = next(m for m in msgs if m["role"] == "user")
    first["content"] = f"[{tag}-{rep}] " + first["content"]
    body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": cell["max_tokens"], "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time(); t1 = None; chunks = 0; usage = None; text = []
    for line in urllib.request.urlopen(req, timeout=900):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        ch = d.get("choices") or []
        c = (ch[0].get("delta", {}).get("content") if ch else None)
        if c:
            chunks += 1
            text.append(c)
            if t1 is None:
                t1 = time.time()
    end = time.time()
    toks = (usage or {}).get("completion_tokens", chunks)
    s = "".join(text)
    if out_dir:
        (out_dir / f"{tag}-{rep}.txt").write_text(s)
    print(json.dumps({"cell": name, "tag": tag, "rep": rep, "toks": toks, "ptoks": (usage or {}).get("prompt_tokens"), "chunks": chunks, "ttft": round((t1 or end) - t0, 3),
                      "dec_tps": round((toks - 1) / (end - t1), 2) if t1 and end > t1 else 0.0, "wall": round(end - t0, 3),
                      "sha": hashlib.sha1(s.encode()).hexdigest()[:12]}), flush=True)
