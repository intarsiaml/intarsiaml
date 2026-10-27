#!/usr/bin/env python3
"""Lane H27P: do the texts stay the same when a prompt prefills beside a decoding stream?
usage: mixids.py BASE_URL MODEL PROMPT_FILE...
Per prompt file: stream A (short prompt, 200 greedy forced tokens) starts, 1 s later request B (the file's prompt, 48 greedy forced tokens) arrives.
Prints md5 of A's text and of B's text. Run against serve --max-batch 16 (B prefills beside A) and --max-batch 1 (B queues behind A) and compare."""
import hashlib, json, sys, threading, time, urllib.request

base, model, files = sys.argv[1], sys.argv[2], sys.argv[3:]

def chat(prompt, max_tokens, out, started=None):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True, "min_tokens": max_tokens, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                if d.get("choices"):
                    c = d["choices"][0].get("delta", {}).get("content")
                    if c:
                        out.append(c)
                        if started is not None and len(out) == 10:
                            started.set()

for f in files:
    a, b, ev = [], [], threading.Event()
    ta = threading.Thread(target=chat, args=("Write a long story about a robot who learns to paint.", 200, a, ev))
    ta.start()
    ev.wait(60)
    time.sleep(1.0)
    chat(open(f).read(), 48, b)
    ta.join()
    h = lambda x: hashlib.md5("".join(x).encode()).hexdigest()[:10]
    print(f"{f.split('/')[-1]}: A {h(a)} ({len(a)} chunks) B {h(b)} ({len(b)} chunks)", flush=True)
