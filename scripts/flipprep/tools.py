#!/usr/bin/env python3
"""Lane flipprep: real `tools`-parameter traffic. 24 chat requests (OpenAI API, `tools` array, the server's parser as shipped) to an intarsia server:
18 single-turn requests that should end in a tool call (English, Chinese, code, parallel calls) and 6 follow-ups carrying a tool result (the
model answers in text). Sequential (1 stream) or in pairs (2 streams). Per request: finish_reason, tool_calls (name, arguments parsed as JSON?),
completion tokens, and (sequential only) the /metrics MTP deltas around that request. Same requests, temperature and seed on every arm, so
the `signature` of each reply (tool names + arguments, or the text) can be compared across arms.
tools.py --url URL --model guidance --arm NAME --out DIR [--metrics URL] [--temperature 0] [--streams 1]"""
import argparse, hashlib, json, pathlib, re, statistics, threading, time, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web and return the top results.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query on the analytics database.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "database": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601"}, "duration_minutes": {"type": "integer"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email.", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "Execute Python code in a sandbox and return stdout.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]
SINGLE = [
    "What's the weather in Paris right now, in celsius?",
    "Search the web for the release date of Rust 1.90 and give me the top 3 results.",
    "What is (1234 * 5678) + 91011? Use the calculator.",
    "Open src/main.rs lines 1 to 60 so I can see the entry point.",
    "How many orders per country did we have last month? The table is orders(country, created_at, total). Database: analytics.",
    "Schedule a 45 minute design review titled 'Design review' tomorrow at 14:00 (2026-11-16) with ana@example.com and bo@example.com.",
    "Email carol@example.com with subject 'Status' and a three line summary that the build is green and the deploy is scheduled for Friday.",
    "Write and run Python that prints the first 15 Fibonacci numbers.",
    "Compare the weather in Tokyo and Berlin today, I want both.",
    "查一下北京今天的天气，用摄氏度。",
    "帮我搜索“Rust 异步运行时 对比”，返回前5条结果。",
    "用计算器算一下 (88 * 17) - 4096 / 8。",
    "读取 config/settings.toml 的全部内容。",
    "给 li@example.com 发一封邮件，主题是“周会纪要”，正文简单写三句话。",
    "Run Python to check whether 1000003 is prime and print the result.",
    "Look up the population of Lagos with a web search, then compute what 3 percent of it is.",
    "Show me the top 5 customers by revenue from the sales database (tables customers, orders).",
    "Read README.md and tell me what the project is called.",
]
FOLLOW = [  # (user, assistant tool call, tool name, tool result)
    ("What's the weather in Oslo?", "get_weather", {"city": "Oslo", "unit": "celsius"}, '{"temp": -2, "condition": "light snow", "wind_kph": 14}'),
    ("What is 17 * 23 + 5?", "calculator", {"expression": "17 * 23 + 5"}, '{"result": 396}'),
    ("Search for the capital of Australia.", "web_search", {"query": "capital of Australia", "max_results": 3}, '[{"title": "Canberra - Wikipedia", "snippet": "Canberra is the capital city of Australia."}]'),
    ("读一下 notes.txt。", "read_file", {"path": "notes.txt"}, '{"content": "买牛奶\\n给妈妈打电话\\n周五提交报告"}'),
    ("How many rows are in the users table?", "run_sql", {"query": "SELECT COUNT(*) AS n FROM users", "database": "analytics"}, '{"rows": [{"n": 18342}]}'),
    ("Run Python that prints 2**20.", "run_python", {"code": "print(2**20)"}, '{"stdout": "1048576\\n"}'),
]


def messages(i):
    if i < len(SINGLE):
        return [{"role": "user", "content": SINGLE[i]}]
    u, name, args, res = FOLLOW[i - len(SINGLE)]
    return [{"role": "user", "content": u},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": res}]


N = len(SINGLE) + len(FOLLOW)


def scrape(url):
    out = {}
    if not url:
        return out
    try:
        txt = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"^(intarsia_spec_\w+)\s+([-0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def one(a, i, out, metrics):
    body = {"model": a.model, "messages": messages(i), "tools": TOOLS, "max_tokens": a.max_tokens, "temperature": a.temperature, "seed": 3000 + i}
    if a.temperature > 0:
        body.update({"top_k": 20, "top_p": 0.95})
    if a.tool_choice:
        body["tool_choice"] = a.tool_choice
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m0 = scrape(metrics) if metrics else {}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
    except Exception as e:
        out[i] = {"i": i, "error": repr(e)}
        return
    dt = time.time() - t0
    m1 = scrape(metrics) if metrics else {}
    ch = j["choices"][0]
    msg = ch["message"]
    calls, valid = [], True
    for c in msg.get("tool_calls") or []:
        f = c["function"]
        try:
            args = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
        except Exception:
            args, valid = None, False
        calls.append({"name": f["name"], "args": args})
    names = {t["function"]["name"] for t in TOOLS}
    sig = json.dumps(calls, sort_keys=True, ensure_ascii=False) if calls else (msg.get("content") or "")
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    out[i] = {"i": i, "kind": "single" if i < len(SINGLE) else "follow-up", "finish": ch["finish_reason"], "n_calls": len(calls), "calls_valid_json": valid,
              "names_known": all(c["name"] in names for c in calls), "completion_tokens": j["usage"]["completion_tokens"], "wall_s": round(dt, 2),
              "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""), "content_head": (msg.get("content") or "")[:80],
              "signature": hashlib.sha256(sig.encode()).hexdigest()[:16], "calls": calls, "mtp": d}


PLAIN = ["Explain how a hash map handles collisions, with an example in Python.", "Write a short story about a lighthouse keeper who finds a message in a bottle.",
         "用中文解释埃拉托斯特尼筛法，以及它为什么比试除法更快。", "Summarise the causes of the French Revolution in five paragraphs."]


def plain(a, i, out):
    """--mix: a plain chat request (no tools, so it drafts) beside tool request i, greedy, same on every arm."""
    body = {"model": a.model, "messages": [{"role": "user", "content": PLAIN[i % len(PLAIN)]}], "max_tokens": 160, "temperature": 0, "seed": 4000 + i}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
    except Exception as e:
        out[i] = {"i": i, "error": repr(e)}
        return
    m = j["choices"][0]["message"]
    out[i] = {"i": i, "signature": hashlib.sha256(((m.get("content") or "") + "|" + (m.get("reasoning_content") or m.get("reasoning") or "")).encode()).hexdigest()[:16], "completion_tokens": j["usage"]["completion_tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--metrics", default=""); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--streams", type=int, default=1); ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--tool-choice", default="", help="e.g. none: the tools array is in the prompt but no grammar constrains the reply, so the request drafts")
    ap.add_argument("--mix", action="store_true", help="with --streams 2: each tool request runs beside a plain chat request (which drafts); both signatures are recorded")
    a = ap.parse_args()
    out = [None] * N
    pl = [None] * N
    one(a, 0, [None] * N, "")  # warm-up
    m0 = scrape(a.metrics)
    for lo in range(0, N, a.streams):
        idx = range(lo, min(lo + a.streams, N))
        ts = [threading.Thread(target=one, args=(a, i, out, a.metrics if a.streams == 1 else "")) for i in idx]
        if a.mix:
            ts += [threading.Thread(target=plain, args=(a, i, pl)) for i in idx]
        [t.start() for t in ts]; [t.join() for t in ts]
    m1 = scrape(a.metrics)
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    prop, acc, steps = d.get("intarsia_spec_proposed_tokens_total", 0), d.get("intarsia_spec_accepted_tokens_total", 0), d.get("intarsia_spec_steps_total", 0)
    ok = [o for o in out if o and "error" not in o]
    single = [o for o in ok if o["kind"] == "single"]
    summ = {"mix": a.mix, "tool_choice": a.tool_choice, "arm": a.arm, "temperature": a.temperature, "streams": a.streams, "requests": N, "errors": [o for o in out if o and "error" in o],
            "single_with_tool_call": sum(1 for o in single if o["n_calls"] > 0), "single": len(single), "tool_calls_total": sum(o["n_calls"] for o in ok),
            "all_arguments_valid_json": all(o["calls_valid_json"] for o in ok), "all_names_known": all(o["names_known"] for o in ok),
            "finish_reasons": {k: sum(1 for o in ok if o["finish"] == k) for k in {o["finish"] for o in ok}},
            "completion_tokens": sum(o["completion_tokens"] for o in ok), "verify_passes": steps, "drafted": prop, "accepted": acc,
            "accept_rate": round(acc / prop, 4) if prop else None, "tokens_per_pass": round((acc + steps) / steps, 3) if steps else None}
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(a.out) / f"tools-{a.arm}-t{a.temperature}-s{a.streams}{'-mix' if a.mix else ''}{'-tc' + a.tool_choice if a.tool_choice else ''}.json").write_text(json.dumps({"summary": summ, "each": out, "plain": pl}, ensure_ascii=False, indent=1))
    print("[tools]", json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
