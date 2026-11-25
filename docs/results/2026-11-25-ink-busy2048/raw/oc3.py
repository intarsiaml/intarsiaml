"""Busy-chunk output check, fresh-server form (48b). mode solo: run the 4 prompts one at a time, write OUT/solo.json.
mode conc: on a fresh server (no cached prefix), send prompt 0, wait until it decodes, then prompts 1-3 (they prefill
at the busy chunk next to decoding), compare reasoning + answer text with SOLO.json.
usage: oc2.py PORT OUT PROMPTDIR solo|conc [SOLO.json]"""
import hashlib
import json
import sys
import threading
import time
import urllib.request

port, out, pdir, mode = sys.argv[1:5]
P = [open(f"{pdir}/p{i}.txt").read() for i in range(4)]
N = 256


def req(text):
    body = json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": text}], "max_tokens": N,
                       "temperature": 0, "stream": False}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f:
        j = json.load(f)
    m = j["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n--answer--\n" + (m.get("content") or ""), j.get("usage", {})


md = lambda s: hashlib.md5(s.encode()).hexdigest()[:8]
if mode == "solo":
    res = [req(p) for p in P]
    json.dump([{"text": t, "usage": u} for t, u in res], open(f"{out}/solo.json", "w"))
    for i, (t, u) in enumerate(res):
        print(f"solo prompt {i} ({u.get('prompt_tokens')} tokens) gen {u.get('completion_tokens')} md5 {md(t)} len {len(t)}")
    sys.exit(0)
solo = json.load(open(sys.argv[5]))
res = [None] * 4


def go(i):
    res[i] = req(P[i])


th = [threading.Thread(target=go, args=(0,))]
th[0].start()
time.sleep(2.0)
for i in (1, 2, 3):
    t = threading.Thread(target=go, args=(i,))
    t.start()
    th.append(t)
for t in th:
    t.join()
json.dump([{"text": t, "usage": u} for t, u in res], open(f"{out}/conc.json", "w"))
for i in range(4):
    a, b = solo[i]["text"], res[i][0]
    eq = a == b
    pre = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    print(f"conc prompt {i} ({res[i][1].get('prompt_tokens')} tokens): {'EQUALS' if eq else 'DIFFERS FROM'} solo"
          + ("" if eq else f" (first difference at char {pre} of {len(a)})")
          + f" md5 solo {md(a)} conc {md(b)} gen {res[i][1].get('completion_tokens')}" + (" INVALID (empty)" if len(a) <= 12 else ""))
