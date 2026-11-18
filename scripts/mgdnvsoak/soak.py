#!/usr/bin/env python3
"""mgdnvsoak run 4: continuous mixed load against a served engine for DUR seconds with NC client threads.
Each client loops over a request mix: greedy and sampled (T 0.7, top_p 0.95, fixed seed), thinking on and off, fixed prompts (no tag), so a
(kind) key must always give the same reply sha (batched == solo is claimed bitwise); one JSON line per request to OUT, errors recorded.
  soak.py <port> <cells_dir> <dur_s> <nclients> <out.jsonl>"""
import hashlib, json, sys, threading, time, urllib.request, pathlib, random, re
port, cdir, dur, nc, out = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
names = ["chat-short", "code-gen", "reason", "agent-edit"]
lock = threading.Lock()
end = time.time() + dur
f = open(out, "a")


def req(kind, cell, temp, seed, think, mx):
    c = json.load(open(cdir / f"{cell}.json"))
    body = {"model": "m", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": min(mx, c["max_tokens"]), "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": think}, "messages": c["messages"]}
    if temp > 0:
        body["top_p"] = 0.95
        body["seed"] = seed
    t0 = time.time()
    rec = {"kind": kind, "t": round(t0, 1)}
    try:
        text, usage = [], None
        r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        for line in urllib.request.urlopen(r, timeout=1200):
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta", {})
                for k in ("reasoning_content", "reasoning", "content"):
                    if dl.get(k):
                        text.append(dl[k])
        s = "".join(text)
        toks = (usage or {}).get("completion_tokens", 0)
        bad = []
        if toks < 8 or len(s.strip()) == 0:
            bad.append("empty")
        if re.search(r"\bnan\b|\binf\b|�", s, re.I) and "nan" not in cell:
            bad.append("nan/inf/replacement-char")
        if len(s) > 200 and len(set(s[-200:])) < 6:
            bad.append("degenerate-tail")
        rec.update(toks=toks, wall=round(time.time() - t0, 2), sha=hashlib.sha1(s.encode()).hexdigest()[:12], bad=bad)
    except Exception as e:
        rec.update(error=repr(e)[:200], wall=round(time.time() - t0, 2))
    with lock:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def client(ci):
    rnd = random.Random(ci)
    while time.time() < end:
        cell = names[rnd.randrange(4)]
        mode = rnd.randrange(4)  # 0 greedy, 1 sampled, 2 greedy thinking, 3 sampled thinking
        temp = 0.7 if mode in (1, 3) else 0
        think = mode >= 2
        seed = 1000 + rnd.randrange(3)
        kind = f"{cell}|{'T' if temp else 'G'}{seed if temp else ''}|{'think' if think else 'nothink'}"
        req(kind, cell, temp, seed, think, 400 if think else 300)


th = [threading.Thread(target=client, args=(i,)) for i in range(nc)]
[t.start() for t in th]
[t.join() for t in th]
print("soak done")
