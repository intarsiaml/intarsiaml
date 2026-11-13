#!/usr/bin/env python3
"""ab35b run 3b: tool-call smoke against one OpenAI-compatible endpoint, each engine with its own tool parser.

Plain stdlib (no localeval): the checks are on the response shape, not timing.
  per scenario, N reps at temperature 0, thinking off, tool_choice auto:
    - tool call: `tool_calls` present, finish_reason `tool_calls`, function name right, `arguments` strict JSON
      (json.loads, no trailing text) and every required key present with the right type/value, no raw `<tool_call>`
      markup left in `content`;
    - tool receipt: the tool result goes back as role `tool`, the final reply must contain the result value;
    - stream: the same request with stream=true; the tool_call deltas are assembled and checked the same way;
    - JSON mode: response_format json_schema, the reply must be strict JSON with the required keys.
  python3 scripts/ab35b/toolsmoke.py --url http://spark-fa00:8000/v1 --model guidance --arm A --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                    "required": ["city", "unit"]}}},
    {"type": "function", "function": {"name": "add_numbers", "description": "Add two integers.",
     "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}},
         "when": {"type": "object", "properties": {"date": {"type": "string"}, "hour": {"type": "integer"}}, "required": ["date", "hour"]}},
         "required": ["title", "attendees", "when"]}}},
]
# (prompt, tool name, arg checker, tool result text, value the final reply must contain)
SCEN = [
    ("What is the weather in Paris right now? Use celsius.", "get_weather",
     lambda a: isinstance(a.get("city"), str) and "paris" in a["city"].lower() and a.get("unit") == "celsius", '{"temp": 17, "sky": "rain"}', "17"),
    ("Please add 417 and 285 with the tool.", "add_numbers",
     lambda a: a.get("a") in (417, 285) and a.get("b") in (417, 285) and a["a"] != a["b"] and type(a["a"]) is int, '{"sum": 702}', "702"),
    ("Schedule a meeting titled 'Design review' with Ana and Bo on 2026-12-03 at 14:00.", "create_event",
     lambda a: isinstance(a.get("title"), str) and "design review" in a["title"].lower() and isinstance(a.get("attendees"), list)
     and len(a["attendees"]) == 2 and isinstance(a.get("when"), dict) and a["when"].get("date") == "2026-12-03" and a["when"].get("hour") == 14,
     '{"event_id": "evt-5521"}', "5521"),
]
JSON_SCHEMA = {"name": "person", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
               "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age", "tags"], "additionalProperties": False}}


def post(url, body, timeout=300):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def base(model, msgs, **kw):
    b = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(kw)
    return b


def call(url, model, msgs, stream=False, **kw):
    """-> (message dict {content, tool_calls}, finish_reason, error)"""
    try:
        body = base(model, msgs, tools=TOOLS, tool_choice="auto", **kw)
        if not stream:
            d = json.loads(post(url, body).read())
            ch = d["choices"][0]
            return ch["message"], ch.get("finish_reason"), None
        body["stream"] = True
        content, tcs, fin = "", {}, None
        for raw in post(url, body):
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if not d.get("choices"):
                continue
            ch = d["choices"][0]
            dl = ch.get("delta") or {}
            content += dl.get("content") or ""
            for tc in dl.get("tool_calls") or []:
                s = tcs.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                s["id"] = tc.get("id") or s["id"]
                f = tc.get("function") or {}
                s["function"]["name"] += f.get("name") or ""
                s["function"]["arguments"] += f.get("arguments") or ""
            fin = ch.get("finish_reason") or fin
        return {"content": content, "tool_calls": [tcs[k] for k in sorted(tcs)]}, fin, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


def check_call(msg, fin, name, ok_args):
    notes = []
    tcs = (msg or {}).get("tool_calls") or []
    if len(tcs) != 1:
        return False, [f"{len(tcs)} tool_calls"], None
    tc = tcs[0]
    if tc["function"]["name"] != name:
        notes.append(f"name {tc['function']['name']}")
    try:
        args = json.loads(tc["function"]["arguments"])
        if not isinstance(args, dict):
            raise ValueError("not an object")
    except Exception as e:  # noqa: BLE001
        return False, notes + [f"arguments not strict JSON ({e}): {tc['function']['arguments'][:120]!r}"], tc
    if not ok_args(args):
        notes.append(f"args wrong: {json.dumps(args)[:160]}")
    if fin != "tool_calls":
        notes.append(f"finish_reason {fin}")
    if "<tool_call" in (msg.get("content") or "") or "<function=" in (msg.get("content") or ""):
        notes.append("raw tool markup in content")
    return not notes, notes, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stream-reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"arm": a.arm, "url": a.url, "scenarios": [], "json_mode": {}}
    for prompt, name, ok_args, tool_result, expect in SCEN:
        rec = {"tool": name, "call_ok": 0, "receipt_ok": 0, "stream_call_ok": 0, "stream_receipt_ok": 0, "fails": []}
        for stream, reps in ((False, a.reps), (True, a.stream_reps)):
            for rep in range(reps):
                msgs = [{"role": "user", "content": prompt}]
                msg, fin, err = call(a.url, a.model, msgs, stream=stream)
                tag = ("stream_" if stream else "")
                if err:
                    rec["fails"].append(f"{tag}call error {err}")
                    continue
                ok, notes, tc = check_call(msg, fin, name, ok_args)
                rec[tag + "call_ok"] += ok
                if not ok:
                    rec["fails"].append(f"{tag}call: {notes}")
                if tc is None:
                    continue
                msgs2 = msgs + [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": [{"id": tc.get("id") or "call_1", "type": "function", "function": tc["function"]}]},
                                {"role": "tool", "tool_call_id": tc.get("id") or "call_1", "content": tool_result}]
                m2, f2, e2 = call(a.url, a.model, msgs2, stream=stream)
                if e2:
                    rec["fails"].append(f"{tag}receipt error {e2}")
                    continue
                txt = (m2.get("content") or "")
                good = expect in txt and not (m2.get("tool_calls")) and "<tool_call" not in txt
                rec[tag + "receipt_ok"] += good
                if not good:
                    rec["fails"].append(f"{tag}receipt: expected {expect!r} in {txt[:160]!r} tool_calls={bool(m2.get('tool_calls'))}")
        rec["reps"], rec["stream_reps"] = a.reps, a.stream_reps
        res["scenarios"].append(rec)
        print(f"[tools] {name}: call {rec['call_ok']}/{a.reps}, receipt {rec['receipt_ok']}/{a.reps}, stream call {rec['stream_call_ok']}/{a.stream_reps}, "
              f"stream receipt {rec['stream_receipt_ok']}/{a.stream_reps}; fails={rec['fails'][:3]}", flush=True)
    jm = {"ok": 0, "reps": a.reps, "fails": []}
    for rep in range(a.reps):
        try:
            d = json.loads(post(a.url, base(a.model, [{"role": "user", "content": f"Return a JSON person: name Ada, age {36 + rep}, tags two words."}],
                                            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA})).read())
            txt = d["choices"][0]["message"]["content"]
            o = json.loads(txt)
            good = set(o) == {"name", "age", "tags"} and isinstance(o["age"], int) and isinstance(o["tags"], list)
            jm["ok"] += good
            if not good:
                jm["fails"].append(f"shape {txt[:120]!r}")
        except Exception as e:  # noqa: BLE001
            jm["fails"].append(f"{type(e).__name__}: {str(e)[:160]}")
    res["json_mode"] = jm
    print(f"[tools] json_schema response_format: {jm['ok']}/{a.reps}; fails={jm['fails'][:2]}", flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}-tools.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
