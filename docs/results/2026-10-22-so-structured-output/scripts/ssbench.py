# /// script
# requires-python = ">=3.10"
# dependencies = ["jsonschema"]
# ///
"""Structured-output bench client (lane so, 2026-10-22).

  ssbench.py speed    --url U --engine intarsia|llama --reps N --out F.jsonl   constrained / greedy / sampled tok/s
  ssbench.py identity --url U --out F.json                                     greedy outputs, every e2e schema + 3 new

Reuses scripts/j1-structured/e2e.py (schemas, HTTP helper). The speed arms use one request each, streaming:
  cons     fruits-40 schema, greedy, response_format json_schema (host logits + mask)
  greedy   same prompt, same token count, ignore_eos, greedy (device argmax)
  sampled  same prompt, same token count, ignore_eos, temperature 0.7 (host logits, no mask)
tok/s = (events - 1) / (last event - first event), as in e2e.py --constrained-speed.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../../scripts/j1-structured"))
import e2e  # noqa: E402

NEW_SCHEMAS = {
    "ss_nested": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {"zip": {"type": "integer", "minimum": 0, "maximum": 99999}, "city": {"type": "string"}},
                        "required": ["zip", "city"],
                        "additionalProperties": False,
                    },
                },
                "required": ["name", "address"],
                "additionalProperties": False,
            }
        },
        "required": ["user"],
        "additionalProperties": False,
    },
    "ss_enum": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            "level": {"enum": ["low", "mid", "high", 7, None]},
        },
        "required": ["answer", "level"],
        "additionalProperties": False,
    },
    "ss_array": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "minimum": 1, "maximum": 99}, "tag": {"type": "string", "enum": ["x", "yy"]}},
                    "required": ["id", "tag"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}
NEW_PROMPTS = {
    "ss_nested": ["Describe a made-up user with a home address.", "Invent a customer in Lisbon."],
    "ss_enum": ["Which option is best, A, B, C or D? Also rate the level.", "Pick an answer and a level at random."],
    "ss_array": ["List three tagged items.", "Give a small inventory with ids."],
}
FRUITS = {
    "type": "object",
    "properties": {"fruits": {"type": "array", "items": {"type": "string"}, "minItems": 40, "maxItems": 40}},
    "required": ["fruits"],
    "additionalProperties": False,
}
PROMPT = "List forty different fruits as JSON.\n"


def one(base, body):
    st, ev, stamps, dt = e2e.post(base + "/v1/completions", body, True)
    assert st == 200, ev
    toks = [s for (_, e), s in zip(ev, stamps) if e.get("choices") and e["choices"][0].get("text")]
    usage = [e["usage"] for _, e in ev if e.get("usage")]
    n = usage[-1]["completion_tokens"] if usage else len(toks)
    tps = (len(toks) - 1) / (toks[-1] - toks[0]) if len(toks) > 1 else 0.0
    return n, tps, dt, len(toks)


def llama_native(a):
    """llama-server /completion (native): `json_schema` is converted to a grammar by the server. Streams `content`."""
    def one_native(body):
        st, ev, stamps, dt = e2e.post(a.url + "/completion", body, True)
        assert st == 200, ev
        pairs = [(e["content"], s) for (_, e), s in zip(ev, stamps) if e.get("content")]
        n = [e for _, e in ev if e.get("stop")][-1].get("tokens_predicted", len(pairs))
        text = "".join(c for c, _ in pairs)
        ts = [s for _, s in pairs]
        return n, (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0, dt, text

    rows = []
    for r in range(a.reps):
        base = {"prompt": PROMPT, "temperature": 0, "stream": True, "cache_prompt": False}
        n, cons, dt, text = one_native({**base, "n_predict": 600, "json_schema": FRUITS})
        try:
            ok = len(json.loads(text)["fruits"]) == 40
        except Exception:  # noqa: BLE001
            ok = False
        _, gr, dtg, _ = one_native({**base, "n_predict": n, "ignore_eos": True})
        row = {"rep": r, "tokens": n, "cons_tok_s": cons, "greedy_tok_s": gr, "cons_s": dt, "greedy_s": dtg, "valid_40": ok,
               "text_head": text[:80]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        for row in rows:
            f.write(json.dumps({"label": a.label, **row}) + "\n")


def speed(a):
    rows = []
    rf = {"type": "json_schema", "json_schema": {"name": "f", "schema": FRUITS}}
    base_body = {"prompt": PROMPT, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if a.engine == "llama":
        base_body["cache_prompt"] = False
    for r in range(a.reps):
        n, cons, dt, ne = one(a.url, {**base_body, "max_tokens": 600, "response_format": rf})
        _, gr, dtg, _ = one(a.url, {**base_body, "max_tokens": n, "ignore_eos": True})
        _, sm, dts, _ = one(a.url, {**base_body, "max_tokens": n, "ignore_eos": True, "temperature": 0.7})
        row = {"rep": r, "tokens": n, "events": ne, "cons_tok_s": cons, "greedy_tok_s": gr, "sampled_tok_s": sm,
               "cons_s": dt, "greedy_s": dtg, "sampled_s": dts}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        for row in rows:
            f.write(json.dumps({"label": a.label, **row}) + "\n")


def identity(a):
    e2e.SCHEMAS.update(NEW_SCHEMAS)
    e2e.PROMPTS.update(NEW_PROMPTS)
    out = {}
    for name, prompts in e2e.PROMPTS.items():
        for j, p in enumerate(prompts):
            key = f"chat-{name}-{j}"
            try:
                out[key], _ = e2e.run_chat_json(a.url, name, p, False, 0.0, 0)
            except Exception as ex:  # noqa: BLE001
                out[key] = f"ERROR {type(ex).__name__}: {ex}"
            print(key, out[key][:100], flush=True)
    for name in ("number", "person", "ss_nested", "ss_enum", "ss_array"):
        key = f"completion-{name}"
        try:
            out[key], _ = e2e.run_completion_json(a.url, name, e2e.PROMPTS[name][0], False, 0.0, 0)
        except Exception as ex:  # noqa: BLE001
            out[key] = f"ERROR {type(ex).__name__}: {ex}"
        print(key, out[key][:100], flush=True)
    for j, p in enumerate(e2e.TOOL_PROMPTS[:3]):
        key = f"tools-required-{j}"
        try:
            r = e2e.run_openai_tools(a.url, ["get_weather", "web_search", "send_email"], "required", p, False, 0.0, 0)
            out[key] = json.dumps(r[0] if isinstance(r, tuple) else r, sort_keys=True)
        except Exception as ex:  # noqa: BLE001
            out[key] = f"ERROR {type(ex).__name__}: {ex}"
        print(key, out[key][:100], flush=True)
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["speed", "identity", "llama-native"])
ap.add_argument("--url", default="http://127.0.0.1:8093")
ap.add_argument("--engine", default="intarsia")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--label", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
{"speed": speed, "identity": identity, "llama-native": llama_native}[a.mode](a)
