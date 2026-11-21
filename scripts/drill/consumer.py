#!/usr/bin/env python3
"""drill: what an OpenAI-compatible consumer with the SDK's own retry sees across an outage.
Every --every seconds (until killed) it sends one small streaming chat completion through the openai SDK (default max_retries=2 with the
SDK's backoff) and logs one JSON line per call: wall start/end, attempts made (httpx request hook), first-token time, outcome, exception.
A second thread keeps one LONG streaming request in flight (max_tokens 300) and logs how it ends (finish_reason, tokens seen, exception).
usage: cd /opt/code/deepeval && uv run python scripts/drill/consumer.py --base http://spark-fa00:8191/v1 --model guidance --out FILE.jsonl
"""
import argparse, json, threading, time, httpx, openai

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--long-tokens", type=int, default=300); ap.add_argument("--every", type=float, default=2.0); ap.add_argument("--retries", type=int, default=2); ap.add_argument("--timeout", type=float, default=60.0)
a = ap.parse_args()
lock = threading.Lock(); f = open(a.out, "a", buffering=1)
def log(d):
    with lock: f.write(json.dumps(d) + "\n")

def client(counter):  # one long-lived client per thread (keep-alive pool, as a real consumer has); counter = that thread's attempt list
    hc = httpx.Client(event_hooks={"request": [lambda r: counter.append(time.time())]})
    return openai.OpenAI(base_url=a.base, api_key="x", max_retries=a.retries, timeout=a.timeout, http_client=hc)

TL = threading.local()
def one(kind, prompt, max_tokens):
    if not hasattr(TL, "cl"): TL.attempts = []; TL.cl = client(TL.attempts)
    attempts, cl = TL.attempts, TL.cl; attempts.clear(); t0 = time.time(); first = None; toks = 0; fin = None; err = None; text = []
    try:
        st = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0, stream=True,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        for ch in st:
            for c in ch.choices:
                if c.delta and c.delta.content:
                    first = first or time.time(); toks += 1; text.append(c.delta.content)
                if c.finish_reason: fin = c.finish_reason
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"
    log({"kind": kind, "t0": round(t0, 3), "t1": round(time.time(), 3), "attempts": len(attempts), "attempt_t": [round(x, 3) for x in list(attempts)],
         "first_s": None if first is None else round(first - t0, 3), "chunks": toks, "finish": fin, "error": err})

def short():
    i = 0
    while True:
        s = time.time(); i += 1; one("short", f"Reply with the single word ok. ({i})", 16); time.sleep(max(0.0, a.every - (time.time() - s)))

def long():
    i = 0
    while True:
        i += 1
        p = f"Count from 1 to 200 separated by spaces, starting at {i}." if a.long_tokens <= 300 else "Write the integers from 1 to 20000 in order, separated by single spaces, with no other text."
        one("long", p, a.long_tokens); time.sleep(1.0)

for fn in (short, long): threading.Thread(target=fn, daemon=True).start()
log({"kind": "start", "t0": round(time.time(), 3)})
while True: time.sleep(3600)
