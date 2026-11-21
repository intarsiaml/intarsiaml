#!/usr/bin/env python3
"""drill (runs on spark2): one tiny streaming chat request, prints JSON {http, first_content_s, total_s, text}. usage: ttft.py PORT [PROMPT]"""
import http.client, json, sys, time
port = int(sys.argv[1]); prompt = sys.argv[2] if len(sys.argv) > 2 else "Reply with the single word ok."
body = {"model": "guidance", "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}
out = {"http": None, "first_content_s": None, "total_s": None, "text": "", "error": None}
t0 = time.time()
try:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse(); out["http"] = r.status
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        try: ev = json.loads(line[5:])
        except ValueError: continue
        for ch in ev.get("choices") or []:
            d = (ch.get("delta") or {}).get("content")
            if d:
                if out["first_content_s"] is None: out["first_content_s"] = round(time.time() - t0, 3)
                out["text"] += d
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"
out["total_s"] = round(time.time() - t0, 3)
print(json.dumps(out))
