import json, sys, time, urllib.request
BASE = sys.argv[1]
OUT = sys.argv[2]
res = {}

def post(body):
    t = time.time()
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        if body.get("stream"):
            chunks = [json.loads(l[6:]) for l in r.read().decode().splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
            return chunks, time.time() - t
        return json.loads(r.read()), time.time() - t

def stream_join(chunks):
    out = {"reasoning_content": "", "content": "", "tool_calls": [], "finish": None}
    for c in chunks:
        for ch in c.get("choices", []):
            d = ch["delta"]
            out["reasoning_content"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            out["tool_calls"] += d.get("tool_calls") or []
            out["finish"] = ch.get("finish_reason") or out["finish"]
    return out

W = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "capital", "schema": {"type": "object", "properties": {"country": {"type": "string"}, "capital": {"type": "string"}}, "required": ["country", "capital"], "additionalProperties": False}}}
Q = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]

def run(name, body):
    try:
        r, dt = post(body)
        res[name] = {"seconds": round(dt, 1), "response": stream_join(r) if body.get("stream") else r["choices"][0] | {"usage": r["usage"]}}
    except Exception as e:  # noqa
        res[name] = {"error": repr(e)[:500]}
    print(name, json.dumps(res[name])[:600], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

run("effort_none", {"messages": Q, "reasoning_effort": "none", "max_tokens": 400, "temperature": 0})
run("effort_none_stream", {"messages": Q, "reasoning_effort": "none", "max_tokens": 400, "temperature": 0, "stream": True})
run("system_multiturn", {"messages": [{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bonjour!"}, {"role": "user", "content": "Quelle est la capitale de l'Italie?"}], "reasoning_effort": "none", "max_tokens": 400, "temperature": 0})
run("effort_high", {"messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "high", "max_tokens": 1500, "temperature": 0})
run("tool_call", {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": W, "reasoning_effort": "none", "max_tokens": 600, "temperature": 0})
run("tool_call_stream", {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": W, "reasoning_effort": "none", "max_tokens": 600, "temperature": 0, "stream": True})
try:
    tc = res["tool_call"]["response"]["message"]["tool_calls"][0]
    follow = [{"role": "user", "content": "What's the weather in Paris right now?"}, {"role": "assistant", "content": None, "tool_calls": [tc]}, {"role": "tool", "tool_call_id": tc["id"], "content": "{\"temp_c\": 21, \"sky\": \"sunny\"}"}]
    run("tool_result_followup", {"messages": follow, "tools": W, "reasoning_effort": "none", "max_tokens": 600, "temperature": 0})
except Exception as e:  # noqa
    res["tool_result_followup"] = {"error": "no tool call to send back: " + repr(e)}
run("tool_choice_required", {"messages": [{"role": "user", "content": "Paris weather?"}], "tools": W, "tool_choice": "required", "reasoning_effort": "none", "max_tokens": 600, "temperature": 0})
run("json_schema", {"messages": [{"role": "user", "content": "What is the capital of France? Give the country and capital."}], "response_format": SCHEMA, "reasoning_effort": "none", "max_tokens": 600, "temperature": 0})
run("json_schema_stream", {"messages": [{"role": "user", "content": "What is the capital of Japan? Give the country and capital."}], "response_format": SCHEMA, "reasoning_effort": "none", "max_tokens": 600, "temperature": 0, "stream": True})
