import json, sys, threading, time, urllib.request

import os
URL = os.environ.get("INTARSIA_URL", "http://127.0.0.1:8090") + "/v1/chat/completions"
WORDS = "the quick brown fox jumps over the lazy dog while observing metrics in grafana ".split()


def req(client, nwords, max_tokens):
    text = " ".join(WORDS[i % len(WORDS)] for i in range(nwords))
    body = json.dumps({"messages": [{"role": "user", "content": text + "\nSummarize in one sentence."}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    r = urllib.request.Request(URL, body, {"Content-Type": "application/json", "X-Intarsia-Client": client})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=600))
        u = d["usage"]
        print(f"{client:10s} prompt={u['prompt_tokens']:6d} gen={u['completion_tokens']:4d} {time.time()-t:6.1f}s", flush=True)
    except Exception as e:
        print(client, "ERR", e, flush=True)


rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for r in range(rounds):
    ths = [
        threading.Thread(target=req, args=("agent-a", 200, 48)),
        threading.Thread(target=req, args=("bench", 1500, 32)),
        threading.Thread(target=req, args=("curl", 40, 64)),
    ]
    if r % 2 == 0:
        ths.append(threading.Thread(target=req, args=("agent-b", 3500, 24)))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
