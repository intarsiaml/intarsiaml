#!/usr/bin/env python3
"""drill2: what clients see when a generation is still running at SIGTERM past the supervisor's 30 s grace.
Two long streaming requests (max_tokens large, ~minutes of decode), one through the OpenAI SDK (default retries) and one raw httpx SSE
reader (so a missing finish_reason / [DONE] / error event / clean EOF vs reset is visible). One JSON line per stream when it ends, with
wall end time (compare with the signal time), chunk count, finish_reason, whether [DONE] arrived, the exception, and the last SSE lines.
usage: cd /opt/code/deepeval && uv run python scripts/drill/longgen.py --base http://spark-4e77:8191/v1 --model guidance --out F.jsonl [--max-tokens 12000]
"""
import argparse, json, threading, time, httpx, openai

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=12000); ap.add_argument("--streams", type=int, default=2)
a = ap.parse_args()
lock = threading.Lock(); f = open(a.out, "a", buffering=1)
def log(d):
    with lock: f.write(json.dumps(d) + "\n")
PROMPT = "Write the integers from 1 to 20000 in order, separated by single spaces, with no other text."
BODY = {"model": a.model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}

def sdk(tag):
    t0 = time.time(); n = 0; fin = None; err = None; first = None; last = t0
    try:
        cl = openai.OpenAI(base_url=a.base, api_key="x", timeout=60.0)  # default max_retries=2
        st = cl.chat.completions.create(model=a.model, messages=BODY["messages"], max_tokens=a.max_tokens, temperature=0, stream=True,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        for ch in st:
            for c in ch.choices:
                if c.delta and c.delta.content: n += 1; last = time.time(); first = first or last
                if c.finish_reason: fin = c.finish_reason
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    log({"kind": "sdk", "tag": tag, "t0": round(t0, 3), "t_end": round(time.time(), 3), "t_last_chunk": round(last, 3), "first_s": first and round(first - t0, 3),
         "chunks": n, "finish_reason": fin, "error": err})

def raw(tag):
    t0 = time.time(); n = 0; fin = None; err = None; done = False; tail = []; last = t0; evs = 0; errev = None
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as hc, hc.stream("POST", a.base + "/chat/completions", json=BODY) as r:
            status = r.status_code
            for line in r.iter_lines():
                if not line: continue
                tail = (tail + [line[:200]])[-4:]
                if line.startswith("data:"):
                    d = line[5:].strip()
                    if d == "[DONE]": done = True; continue
                    evs += 1
                    try: j = json.loads(d)
                    except Exception: continue
                    if "error" in j: errev = j["error"]
                    for c in j.get("choices", []):
                        if (c.get("delta") or {}).get("content"): n += 1; last = time.time()
                        if c.get("finish_reason"): fin = c["finish_reason"]
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"; status = locals().get("status")
    log({"kind": "raw", "tag": tag, "t0": round(t0, 3), "t_end": round(time.time(), 3), "t_last_chunk": round(last, 3), "http": status, "chunks": n,
         "events": evs, "finish_reason": fin, "done_marker": done, "error_event": errev, "exception": err, "tail": tail})

log({"kind": "start", "t0": round(time.time(), 3)})
ths = [threading.Thread(target=(sdk if i % 2 == 0 else raw), args=(f"s{i}",)) for i in range(a.streams)]
for t in ths: t.start()
for t in ths: t.join()
