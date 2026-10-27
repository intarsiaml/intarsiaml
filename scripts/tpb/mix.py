#!/usr/bin/env python3
"""Lane TPB: what a long prefill does to a stream that is already decoding (serve --max-batch 16).
usage: mix.py BASE_URL MODEL PROMPT_FILE [reps]
Stream A decodes 300 forced tokens of a short prompt; 2 s in, request B (the 4k prompt, max_tokens 1) arrives. Prints, per rep: B's TTFT and
A's largest and p99 inter-token gap while B prefills (from B's arrival to B's first token), plus A's median gap before B."""
import json, sys, threading, time, urllib.request, statistics as st

base, model, pfile = sys.argv[1], sys.argv[2], sys.argv[3]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
long_prompt = open(pfile).read()

def stream(prompt, max_tokens, times, done=None):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True, "min_tokens": max_tokens, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                    times.append(time.time())
    if done:
        done.set()

for rep in range(reps):
    a, b = [], []
    ta = threading.Thread(target=stream, args=("Write a long story about a robot who learns to paint.", 300, a))
    ta.start()
    while len(a) < 20:
        time.sleep(0.05)
    time.sleep(1.0)
    t_b = time.time()
    stream(long_prompt + f"\n(request {rep})", 1, b)
    t_b1 = b[0] if b else time.time()
    ta.join()
    gaps = [(a[i] - a[i - 1], a[i]) for i in range(1, len(a))]
    before = [g for g, t in gaps if t < t_b]
    during = [g for g, t in gaps if t_b <= t <= t_b1 + 0.2]
    print(f"rep {rep}: B ttft {t_b1 - t_b:.2f} s | A gap before median {1e3 * st.median(before):.0f} ms | during B: n={len(during)} max {1e3 * max(during or [0]):.0f} ms "
          f"p99 {1e3 * sorted(during or [0])[int(0.99 * (len(during) - 1))]:.0f} ms | A tokens {len(a)}", flush=True)
