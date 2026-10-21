# /// script
# requires-python = ">=3.10"
# dependencies = ["jsonschema==4.23.0"]
# ///
"""R2 reasoning parser + structured output (docs/structured-output.md#thinking-models), against a running server.

50 OpenAI chat requests with thinking ON and a `json_schema` response_format (10 schemas x 5; half streamed, greedy
and sampled): each must give a non-empty `reasoning_content`, a `content` that is JSON valid for the schema, no think
tags in `content`, finish_reason `stop`. Streams must deliver every reasoning delta before the first content delta.
Then 5 Anthropic requests (`thinking` enabled, `tool_choice` any): a `thinking` block, then one valid `tool_use`.

uv run --script scripts/r2-reasoning/e2e.py --url http://127.0.0.1:8107 --out OUT.json [--concurrency 4]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("j1e2e", os.path.join(HERE, "..", "j1-structured", "e2e.py"))
j1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j1)


def chat_case(base, name, prompt, stream, temp, seed):
    schema = j1.SCHEMAS[name]
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
        "max_tokens": 6000,
        "temperature": temp,
        "seed": seed,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    st, out, _, dt = j1.post(base + "/v1/chat/completions", body, stream)
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {str(out)[:300]}")
    if stream:
        reasoning, content, order, finish = "", "", [], None
        for _, e in out:
            if not e.get("choices"):
                continue
            c = e["choices"][0]
            if c["delta"].get("reasoning_content"):
                reasoning += c["delta"]["reasoning_content"]
                order.append("r")
            if c["delta"].get("content"):
                content += c["delta"]["content"]
                order.append("c")
            finish = c["finish_reason"] or finish
        if "r" in order and "c" in order and order.index("c") < len(order) - 1 - order[::-1].index("r"):
            raise RuntimeError("a reasoning delta came after a content delta")
    else:
        v = json.loads(out)
        m = v["choices"][0]["message"]
        reasoning, content, finish = m.get("reasoning_content") or "", m["content"], v["choices"][0]["finish_reason"]
    if not reasoning.strip():
        raise RuntimeError("empty reasoning_content")
    if "think>" in content:
        raise RuntimeError(f"think tag in content: {content[:80]!r}")
    if finish != "stop":
        raise RuntimeError(f"finish_reason {finish}")
    obj = json.loads(content)
    j1.check(schema, obj)
    return {"reasoning_chars": len(reasoning), "content": content, "secs": round(dt, 1)}


def anthropic_case(base, i):
    tools = [{"name": n, "description": n.replace("_", " "), "input_schema": j1.TOOLS[n]} for n in list(j1.TOOLS)[:2]]
    body = {
        "model": "m",
        "max_tokens": 6000,
        "stream": i % 2 == 1,
        "temperature": 0,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
        "tools": tools,
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What is the weather in Tokyo and what time is it there?"}],
    }
    st, out, _, dt = j1.post(base + "/v1/messages", body, body["stream"])
    if st != 200:
        raise RuntimeError(f"HTTP {st}: {str(out)[:300]}")
    if body["stream"]:
        kinds = [e["content_block"]["type"] for n, e in out if n == "content_block_start"]
        think = "".join(e["delta"]["thinking"] for _, e in out if e.get("delta", {}).get("type") == "thinking_delta")
        args = "".join(e["delta"]["partial_json"] for _, e in out if e.get("delta", {}).get("type") == "input_json_delta")
        names = [e["content_block"]["name"] for n, e in out if n == "content_block_start" and e["content_block"]["type"] == "tool_use"]
        inputs = [json.loads(args)]
    else:
        v = json.loads(out)
        kinds = [b["type"] for b in v["content"]]
        think = "".join(b.get("thinking", "") for b in v["content"])
        names = [b["name"] for b in v["content"] if b["type"] == "tool_use"]
        inputs = [b["input"] for b in v["content"] if b["type"] == "tool_use"]
    if kinds[:1] != ["thinking"] or not think.strip():
        raise RuntimeError(f"no thinking block: {kinds}")
    if "tool_use" not in kinds or len(names) < 1:
        raise RuntimeError(f"no tool_use: {kinds}")
    j1.check(j1.TOOLS[names[0]], inputs[0])
    return {"kinds": kinds, "think_chars": len(think), "tool": names[0], "secs": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    jobs = []
    for name in j1.SCHEMAS:
        ps = j1.PROMPTS[name]
        ps = [ps] if isinstance(ps, str) else ps
        for k in range(5):
            jobs.append((f"chat-{name}-{k}-{'stream' if k % 2 else 'plain'}-t{0.0 if k < 3 else 0.7}", chat_case,
                         (a.url, name, ps[k % len(ps)], k % 2 == 1, 0.0 if k < 3 else 0.7, k)))
    for i in range(5):
        jobs.append((f"anthropic-tool-{i}", anthropic_case, (a.url, i)))

    def run(j):
        label, fn, args = j
        try:
            r = fn(*args)
            print("OK  ", label, r.get("secs"), flush=True)
            return label, True, r
        except Exception as e:  # noqa: BLE001
            print("FAIL", label, repr(e)[:300], flush=True)
            return label, False, repr(e)

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(run, jobs))
    chat = [r for r in res if r[0].startswith("chat-")]
    ant = [r for r in res if r[0].startswith("anthropic-")]
    summary = {
        "chat_ok": sum(r[1] for r in chat), "chat_total": len(chat),
        "anthropic_ok": sum(r[1] for r in ant), "anthropic_total": len(ant),
        "wall_s": round(time.time() - t0, 1),
    }
    json.dump({"summary": summary, "results": res}, open(a.out, "w"), indent=1)
    print(json.dumps(summary))
    sys.exit(0 if all(r[1] for r in res) else 1)


main()
