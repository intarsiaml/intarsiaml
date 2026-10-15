#!/usr/bin/env python3
"""serve_check2.py <port> <prompt file> <stop> <out json>: two raw /v1/completions requests on one server. Turn 1 stops
at <stop> (with spec on, verified tokens past the stop are dropped and the KV is rolled back); turn 2 continues the
turn-1 prompt + text + the stop string + a fixed suffix, so its prefill reuses the KV prefix turn 1 left (streamed)."""
import json, sys, urllib.request
port, pfile, stop, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
url = f"http://127.0.0.1:{port}/v1/completions"
prompt = open(pfile).read()
def post(body):
    return urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
with post({"model": "x", "prompt": prompt, "max_tokens": 256, "temperature": 0, "stop": [stop]}) as r:
    t1 = json.loads(r.read())
text1 = t1["choices"][0]["text"]
prompt2 = prompt + text1 + stop + " on day 99. The"
text2, finish2, usage2 = "", None, None
with post({"model": "x", "prompt": prompt2, "max_tokens": 64, "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}}) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        for c in ev.get("choices", []):
            text2 += c.get("text", "") or ""
            finish2 = c.get("finish_reason") or finish2
        usage2 = ev.get("usage") or usage2
metrics = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30).read().decode()
spec = {l.split()[0]: float(l.split()[1]) for l in metrics.splitlines() if l.startswith("intarsia_spec_")}
res = {"turn1_text": text1, "turn1_finish": t1["choices"][0]["finish_reason"], "turn1_usage": t1.get("usage"),
       "turn2_text": text2, "turn2_finish": finish2, "turn2_usage": usage2, "metrics_spec": spec}
json.dump(res, open(out, "w"), indent=1)
print(out, res["turn1_finish"], res["turn1_usage"], res["turn2_finish"], res["turn2_usage"], spec)
