#!/usr/bin/env python3
"""Short real-model smoke of the Inkling-2 server on the CPU backend (1.75 tok/s): each request <= 64 tokens. Prints, per request,
the marker layout of the generated ids (return_token_ids), the parsed reply and whether it parses. Usage: base_url out.json"""
import json, sys, time, urllib.request

BASE, OUT = sys.argv[1], sys.argv[2]
MARK = {200001: "<|message_model|>", 200004: "<|content_text|>", 200008: "<|content_thinking|>", 200010: "<|end_message|>",
        200049: "<|content_invoke_tool_json|>", 200057: "<|content_invoke_tool_text|>", 200022: "<|content_tool_error|>", 200006: "<EOS>"}
res = {}


def post(body):
    t = time.time()
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1500) as r:
        raw = r.read().decode()
    if body.get("stream"):
        return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"], time.time() - t
    return json.loads(raw), time.time() - t


def layout(ids):
    out, run = [], 0
    for i in ids:
        if i in MARK:
            if run:
                out.append(f"<{run} text tokens>")
                run = 0
            out.append(MARK[i])
        else:
            run += 1
    if run:
        out.append(f"<{run} text tokens>")
    return " ".join(out)


def run(name, body):
    body = {"max_tokens": 64, "temperature": 0, **body}
    assert body["max_tokens"] <= 64
    try:
        if body.get("stream"):
            chunks, dt = post(body)
            d = {"reasoning_content": "", "content": "", "tool_calls": [], "finish": None}
            for c in chunks:
                for ch in c.get("choices", []):
                    x = ch["delta"]
                    d["reasoning_content"] += x.get("reasoning_content") or ""
                    d["content"] += x.get("content") or ""
                    d["tool_calls"] += x.get("tool_calls") or []
                    d["finish"] = ch.get("finish_reason") or d["finish"]
            res[name] = {"seconds": round(dt, 1), "stream": d}
        else:
            body["return_token_ids"] = True
            r, dt = post(body)
            ch = r["choices"][0]
            res[name] = {"seconds": round(dt, 1), "layout": layout(ch.get("token_ids", [])), "message": ch["message"], "finish_reason": ch["finish_reason"], "usage": r["usage"]}
    except Exception as e:  # noqa
        res[name] = {"error": repr(e)[:600]}
    print("==", name, json.dumps(res[name], ensure_ascii=False)[:1500], flush=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


W = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "c", "schema": {"type": "object", "properties": {"capital": {"type": "string"}}, "required": ["capital"], "additionalProperties": False}}}
Q = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
run("chat_effort_none", {"messages": Q, "reasoning_effort": "none"})
run("chat_effort_default_stream", {"messages": Q, "stream": True})
run("tool_call_effort_none", {"messages": [{"role": "user", "content": "Weather in Paris? Use the tool."}], "tools": W, "reasoning_effort": "none"})
run("json_schema_effort_none", {"messages": Q, "response_format": SCHEMA, "reasoning_effort": "none"})
