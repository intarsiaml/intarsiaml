#!/usr/bin/env python3
"""serve_check.py <port> <prompt dir> <out json>: serve-path checks on one server, in order:
1. greedy /v1/completions that stops on a stop string, then a streamed two-turn continuation (KV prefix reuse after a stop)
2. seeded sampled completion (twice, same seed) and a sampled completion with stop + min_tokens
3. client disconnect (streamed request closed after a few chunks), then a follow-up request on the same server
4. deterministic /metrics counters
Everything timing-dependent is left out of the JSON so base and new runs compare byte for byte."""
import json, sys, time, urllib.request, http.client
port, pdir, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
short = open(f"{pdir}/short.txt").read(); grounded = open(f"{pdir}/grounded.txt").read()
def post(body, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout)
def comp(body):
    with post({"model": "x", **body}) as r:
        j = json.loads(r.read())
    return {"text": j["choices"][0]["text"], "finish": j["choices"][0]["finish_reason"], "usage": j.get("usage")}
def stream(body):
    text, fin, usage = "", None, None
    with post({"model": "x", "stream": True, "stream_options": {"include_usage": True}, **body}) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ev = json.loads(line[6:])
            for c in ev.get("choices", []):
                text += c.get("text", "") or ""; fin = c.get("finish_reason") or fin
            usage = ev.get("usage") or usage
    return {"text": text, "finish": fin, "usage": usage}
res = {}
t1 = comp({"prompt": short, "max_tokens": 256, "temperature": 0, "stop": ["kettles"]})
res["turn1"] = t1
res["turn2_stream"] = stream({"prompt": short + t1["text"] + "kettles" + " on day 99. The", "max_tokens": 64, "temperature": 0})
res["turn1_again"] = comp({"prompt": short, "max_tokens": 256, "temperature": 0, "stop": ["kettles"]})
res["sampled_a"] = comp({"prompt": grounded, "max_tokens": 120, "temperature": 0.8, "top_p": 0.9, "seed": 1234})
res["sampled_b"] = comp({"prompt": grounded, "max_tokens": 120, "temperature": 0.8, "top_p": 0.9, "seed": 1234})
res["sampled_stop_min"] = comp({"prompt": short, "max_tokens": 200, "temperature": 0.7, "seed": 9, "stop": ["the"], "min_tokens": 30})
# client disconnect mid-stream
c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
c.request("POST", "/v1/completions", json.dumps({"model": "x", "prompt": grounded, "max_tokens": 400, "temperature": 0, "stream": True}), {"Content-Type": "application/json"})
r = c.getresponse(); n = 0
for line in r:
    if line.startswith(b"data: "): n += 1
    if n >= 8: break
c.close()
time.sleep(4)
res["after_disconnect"] = comp({"prompt": short, "max_tokens": 32, "temperature": 0})
res["disconnect_chunks_read"] = n
m = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
keep = ("intarsia_requests_total", "intarsia_requests_finished_total", "intarsia_requests_rejected_total", "intarsia_tokens_generated_total",
        "intarsia_tokens_prompt_total", "intarsia_tokens_cached_total", "intarsia_spec_")
res["metrics"] = {l.rsplit(" ", 1)[0]: float(l.rsplit(" ", 1)[1]) for l in m.splitlines() if l.startswith(keep)}
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
print(out, {k: (v["finish"], v["usage"]) for k, v in res.items() if isinstance(v, dict) and "finish" in v})
