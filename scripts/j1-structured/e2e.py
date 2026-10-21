# /// script
# requires-python = ">=3.10"
# dependencies = ["jsonschema==4.23.0"]
# ///
"""J1 structured-output correctness harness (docs/structured-output.md).

50 varied requests against a running intarsia server: OpenAI chat `response_format` (json_schema over ten schemas,
json_object), `/v1/completions` json_schema, OpenAI `tools` with tool_choice required / named / auto, Anthropic
`tools` with tool_choice any / tool; streaming and non-streaming, greedy and sampled. Every JSON answer and every tool
call's arguments must parse and validate against its schema (jsonschema, Draft 2020-12).

`--constrained-speed` additionally measures decode tok/s of one long constrained generation against the same prompt
unconstrained (forced to the same length), interleaved A/B x3. localeval cannot drive this: its speed runs force the
length with ignore_eos, which a grammar that ends in EOS cannot honour.

    uv run --script scripts/j1-structured/e2e.py --url http://127.0.0.1:8107 --out OUT.json [--constrained-speed]
"""

import argparse
import json
import sys
import time
import urllib.request

import jsonschema

SCHEMAS = {
    "number": {"properties": {"answer": {"type": "integer"}}, "required": ["answer"], "type": "object"},
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string", "pattern": "^[a-z0-9._]+@[a-z0-9.]+\\.[a-z]{2,}$"},
        },
        "required": ["name", "age", "email"],
        "additionalProperties": False,
    },
    "sentiment": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": False,
    },
    "shopping": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
                    "required": ["name", "qty"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "address": {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
                },
                "required": ["street", "city", "zip"],
                "additionalProperties": False,
            },
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["address", "tags"],
        "additionalProperties": False,
    },
    "prime": {
        "type": "object",
        "properties": {
            "is_prime": {"type": "boolean"},
            "factors": {"type": "array", "items": {"type": "integer"}},
            "note": {"type": ["string", "null"]},
        },
        "required": ["is_prime", "factors", "note"],
        "additionalProperties": False,
    },
    "anyof": {
        "type": "object",
        "properties": {"result": {"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 20}]}},
        "required": ["result"],
        "additionalProperties": False,
    },
    "points": {
        "$defs": {
            "pt": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/pt"}, "b": {"$ref": "#/$defs/pt"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    "title": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 5, "maxLength": 40},
            "words": {"type": "integer", "minimum": 1},
        },
        "required": ["title", "words"],
        "additionalProperties": False,
    },
    "version": {
        "type": "object",
        "properties": {"version": {"const": "1.0"}, "ok": {"type": "boolean"}},
        "required": ["version", "ok"],
        "additionalProperties": False,
    },
}

PROMPTS = {
    "number": [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the "
        "rest at $2 each. How many dollars does she make every day? Answer with the number only.",
        "A train travels 60 km per hour for 3 hours, then 40 km per hour for 2 hours. How far does it go in total?",
    ],
    "person": ["Make up a fictional software engineer.", "Describe Ada Lovelace as a contact card entry."],
    "sentiment": ["Classify: 'The movie was a total waste of two hours.'", "Classify: 'Best pizza in town, hands down!'"],
    "shopping": ["I need stuff for pancakes.", "Plan a small grocery list for a picnic."],
    "address": ["Give the address of a made-up bakery in Berlin.", "Invent an office address for a startup."],
    "prime": ["Is 91 prime? Give its factors.", "Is 97 prime?"],
    "anyof": ["What is 17 times 3?", "What colour is the sky on a clear day?"],
    "points": ["Two corners of a unit square.", "The endpoints of a line from the origin to (3, 4)."],
    "title": ["Suggest a blog post title about Rust async.", "Title a short story about a lighthouse keeper."],
    "version": ["Report the API version and whether the system is healthy.", "Status check: version and ok flag."],
}

WEATHER = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 7},
    },
    "required": ["city", "unit"],
    "additionalProperties": False,
}
SEARCH = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
    "required": ["query"],
    "additionalProperties": False,
}
EMAIL = {
    "type": "object",
    "properties": {
        "to": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "subject": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["to", "subject", "body"],
    "additionalProperties": False,
}
TOOLS = {"get_weather": WEATHER, "web_search": SEARCH, "send_email": EMAIL}
TOOL_PROMPTS = [
    "What's the weather in Tokyo for the next 3 days, in celsius?",
    "Search the web for the latest Rust release notes.",
    "Email bob@example.com that the meeting moved to 3pm.",
    "Is it going to rain in Paris tomorrow? Use fahrenheit.",
    "Find five articles about llguidance.",
]


def openai_tools(names):
    return [
        {"type": "function", "function": {"name": n, "description": f"{n.replace('_', ' ')}", "parameters": TOOLS[n]}}
        for n in names
    ]


def post(url, body, stream):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), [], time.time() - t0
    if not stream:
        return resp.status, resp.read().decode(), [], time.time() - t0
    events, stamps, name = [], [], None
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]":
                continue
            events.append((name, json.loads(d)))
            stamps.append(time.time())
            name = None
    return resp.status, events, stamps, time.time() - t0


def check(schema, obj):
    jsonschema.Draft202012Validator(schema).validate(obj)


def run_chat_json(base, schema_name, prompt, stream, temp, seed, rf_type="json_schema"):
    schema = SCHEMAS[schema_name] if rf_type == "json_schema" else {"type": "object"}
    rf = (
        {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
        if rf_type == "json_schema"
        else {"type": "json_object"}
    )
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "response_format": rf,
        "max_tokens": 512,
        "temperature": temp,
        "seed": seed,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    st, out, _, dt = post(base + "/v1/chat/completions", body, stream)
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {str(out)[:300]}")
    if stream:
        text = "".join(
            (e["choices"][0]["delta"].get("content") or "") for _, e in out if e.get("choices")
        )
        finish = [e["choices"][0]["finish_reason"] for _, e in out if e.get("choices") and e["choices"][0]["finish_reason"]]
        finish = finish[-1] if finish else None
    else:
        v = json.loads(out)
        text, finish = v["choices"][0]["message"]["content"], v["choices"][0]["finish_reason"]
    obj = json.loads(text)
    check(schema, obj)
    if finish != "stop":
        raise RuntimeError(f"finish_reason {finish}")
    return text, dt


def run_completion_json(base, schema_name, prompt, stream, temp, seed):
    schema = SCHEMAS[schema_name]
    body = {
        "prompt": f"Question: {prompt}\nAnswer as JSON:\n",
        "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
        "max_tokens": 512,
        "temperature": temp,
        "seed": seed,
        "stream": stream,
    }
    st, out, _, dt = post(base + "/v1/completions", body, stream)
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {str(out)[:300]}")
    text = "".join(e["choices"][0]["text"] for _, e in out if e.get("choices")) if stream else json.loads(out)["choices"][0]["text"]
    check(schema, json.loads(text))
    return text, dt


def run_openai_tools(base, names, choice, prompt, stream, temp, seed):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "tools": openai_tools(names),
        "tool_choice": choice,
        "max_tokens": 512,
        "temperature": temp,
        "seed": seed,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    st, out, _, dt = post(base + "/v1/chat/completions", body, stream)
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {str(out)[:300]}")
    if stream:
        calls = [tc for _, e in out if e.get("choices") for tc in (e["choices"][0]["delta"].get("tool_calls") or [])]
        finish = [e["choices"][0]["finish_reason"] for _, e in out if e.get("choices") and e["choices"][0]["finish_reason"]][-1]
    else:
        v = json.loads(out)
        calls = v["choices"][0]["message"].get("tool_calls") or []
        finish = v["choices"][0]["finish_reason"]
    if choice == "auto" and not calls:
        # free text is a valid answer under auto; reported as a call-rate miss, not a validity failure
        return "NO_CALL", dt
    if not calls:
        raise RuntimeError("no tool call")
    want = choice["function"]["name"] if isinstance(choice, dict) else None
    for c in calls:
        n = c["function"]["name"]
        if n not in names or (want and n != want):
            raise RuntimeError(f"unexpected tool {n}")
        check(TOOLS[n], json.loads(c["function"]["arguments"]))
    if finish != "tool_calls":
        raise RuntimeError(f"finish_reason {finish}")
    return json.dumps([c["function"] for c in calls]), dt


def run_anthropic_tools(base, names, choice, prompt, stream, temp):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"name": n, "description": n.replace("_", " "), "input_schema": TOOLS[n]} for n in names],
        "tool_choice": choice,
        "max_tokens": 512,
        "temperature": temp,
        "stream": stream,
        "thinking": {"type": "disabled"},
    }
    st, out, _, dt = post(base + "/v1/messages", body, stream)
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {str(out)[:300]}")
    if stream:
        uses, cur = [], {}
        for name, e in out:
            if name == "content_block_start" and e["content_block"]["type"] == "tool_use":
                cur[e["index"]] = [e["content_block"]["name"], ""]
            elif name == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
                cur[e["index"]][1] += e["delta"]["partial_json"]
            elif name == "content_block_stop" and e["index"] in cur:
                n, js = cur.pop(e["index"])
                uses.append((n, json.loads(js)))
        reason = [e["delta"]["stop_reason"] for name, e in out if name == "message_delta"][-1]
    else:
        v = json.loads(out)
        uses = [(b["name"], b["input"]) for b in v["content"] if b["type"] == "tool_use"]
        reason = v["stop_reason"]
    if not uses:
        raise RuntimeError("no tool_use block")
    want = choice.get("name")
    for n, inp in uses:
        if n not in names or (want and n != want):
            raise RuntimeError(f"unexpected tool {n}")
        check(TOOLS[n], inp)
    if reason != "tool_use":
        raise RuntimeError(f"stop_reason {reason}")
    return json.dumps(uses), dt


def cases():
    """The 50 requests: (label, callable(base))."""
    out = []
    names = list(SCHEMAS)
    k = 0
    for i, s in enumerate(names):  # 20 chat json_schema
        for j, p in enumerate(PROMPTS[s]):
            stream, temp = (k % 2 == 1), (0.0 if k % 3 else 0.7)
            out.append((f"chat-json_schema-{s}-{j}{'-stream' if stream else ''}-t{temp}",
                        lambda b, s=s, p=p, st=stream, t=temp, sd=k: run_chat_json(b, s, p, st, t, sd)))
            k += 1
    for j, s in enumerate(["person", "shopping", "address", "prime", "title"]):  # 5 chat json_object
        p = PROMPTS[s][0] + " Reply in JSON."
        out.append((f"chat-json_object-{s}{'-stream' if j % 2 else ''}",
                    lambda b, s=s, p=p, st=bool(j % 2): run_chat_json(b, s, p, st, 0.0, 1, rf_type="json_object")))
    for j, s in enumerate(["number", "sentiment", "points", "anyof", "version"]):  # 5 completions
        out.append((f"completions-json_schema-{s}{'-stream' if j % 2 else ''}",
                    lambda b, s=s, st=bool(j % 2): run_completion_json(b, s, PROMPTS[s][1], st, 0.0, 2)))
    tool_sets = [["get_weather", "web_search", "send_email"], ["get_weather", "web_search"], ["send_email", "web_search"]]
    want_for = ["get_weather", "web_search", "send_email", "get_weather", "web_search"]
    for j, p in enumerate(TOOL_PROMPTS):  # 5 required + 5 named (OpenAI)
        ts = tool_sets[0]
        out.append((f"openai-tools-required-{j}{'-stream' if j % 2 else ''}",
                    lambda b, p=p, st=bool(j % 2), ts=ts, t=(0.7 if j == 4 else 0.0): run_openai_tools(b, ts, "required", p, st, t, j)))
        out.append((f"openai-tools-named-{want_for[j]}-{j}{'-stream' if j % 2 == 0 else ''}",
                    lambda b, p=p, st=(j % 2 == 0), w=want_for[j]: run_openai_tools(
                        b, ["get_weather", "web_search", "send_email"], {"type": "function", "function": {"name": w}}, p, st, 0.0, j)))
    for j, p in enumerate(TOOL_PROMPTS[:5]):  # 5 auto (OpenAI)
        out.append((f"openai-tools-auto-{j}{'-stream' if j % 2 else ''}",
                    lambda b, p=p, st=bool(j % 2): run_openai_tools(b, ["get_weather", "web_search", "send_email"], "auto", p, st, 0.0, j)))
    for j, p in enumerate(TOOL_PROMPTS):  # 5 Anthropic
        choice = {"type": "any"} if j % 2 == 0 else {"type": "tool", "name": want_for[j]}
        out.append((f"anthropic-tools-{choice['type']}-{j}{'-stream' if j % 2 else ''}",
                    lambda b, p=p, c=choice, st=bool(j % 2): run_anthropic_tools(b, ["get_weather", "web_search", "send_email"], c, p, st, 0.0)))
    return out


def constrained_speed(base, reps=3):
    """Decode tok/s, constrained vs unconstrained, same prompt and length, interleaved A/B x reps (streaming)."""
    schema = {"type": "object", "properties": {"fruits": {"type": "array", "items": {"type": "string"},
              "minItems": 40, "maxItems": 40}}, "required": ["fruits"], "additionalProperties": False}
    prompt = "List forty different fruits as JSON.\n"

    def one(body):
        st, ev, stamps, _ = post(base + "/v1/completions", body, True)
        assert st == 200, ev
        toks = [s for (_, e), s in zip(ev, stamps) if e.get("choices") and e["choices"][0].get("text")]
        usage = [e["usage"] for _, e in ev if e.get("usage")][-1]
        n = usage["completion_tokens"]
        return n, (len(toks) - 1) / (toks[-1] - toks[0]) if len(toks) > 1 else 0.0, toks

    rows = []
    for r in range(reps):
        nb, cons, _ = one({"prompt": prompt, "max_tokens": 600, "temperature": 0, "stream": True,
                           "stream_options": {"include_usage": True},
                           "response_format": {"type": "json_schema", "json_schema": {"name": "f", "schema": schema}}})
        _, plain, _ = one({"prompt": prompt, "max_tokens": nb, "temperature": 0, "stream": True, "ignore_eos": True,
                           "stream_options": {"include_usage": True}})
        rows.append({"rep": r, "tokens": nb, "constrained_tok_s": cons, "unconstrained_tok_s": plain})
        print(f"rep {r}: {nb} tokens, constrained {cons:.2f} tok/s, unconstrained {plain:.2f} tok/s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8107")
    ap.add_argument("--out", required=True)
    ap.add_argument("--constrained-speed", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight (exercises batched decoding)")
    a = ap.parse_args()

    def one(case):
        label, fn = case
        try:
            text, dt = fn(a.url)
            print(f"OK   {label} ({dt:.1f}s) {text[:120]}", flush=True)
            return {"case": label, "ok": True, "seconds": round(dt, 2), "output": text}
        except Exception as e:  # noqa: BLE001 - every failure is a result
            print(f"FAIL {label}: {type(e).__name__}: {e}", flush=True)
            return {"case": label, "ok": False, "error": f"{type(e).__name__}: {e}"}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        results = list(ex.map(one, cases()))
    n_ok = sum(r["ok"] for r in results)
    no_call = sum(r.get("output") == "NO_CALL" for r in results)
    report = {"url": a.url, "concurrency": a.concurrency, "n": len(results), "valid": n_ok, "auto_without_call": no_call, "results": results}
    if a.constrained_speed:
        report["constrained_speed"] = constrained_speed(a.url)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"valid {n_ok}/{len(results)} (tool_choice auto answered without a call: {no_call})")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
