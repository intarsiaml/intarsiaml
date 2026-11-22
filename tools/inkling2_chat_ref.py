#!/usr/bin/env python3
"""Reference chat prompts for Inkling-2-Mini (lane ink-serve): vLLM's renderer vs the HF jinja template.

Writes `crates/intarsia-models/tests/fixtures/inkling2_chat_cases.json`: for each case the messages and request variables, the
token ids vLLM's `render_inkling2_messages` gives (`vllm_ids`, or `error`), and the ids `chat_template.jinja` gives through
transformers (`jinja_ids`, or null when it raises). The Rust renderer (`crates/intarsia-models/src/inkling_chat.rs`) must equal
`vllm_ids`; `jinja_same` says where the template agrees too.

    uv run --no-project --with transformers --with tokenizers --with jinja2 python tools/inkling2_chat_ref.py \
        --vllm /opt/code/intarsiaml/.worktrees/ink-official/renderers/inkling_encoding.py --model <dir with tokenizer.json>
"""
import argparse, importlib.util, json, pathlib, sys

CASES = []


def case(name, messages, tools=None, **kw):
    CASES.append({"name": name, "messages": messages, "tools": tools, "kwargs": kw})


U = lambda t: {"role": "user", "content": t}
A = lambda t, **k: {"role": "assistant", "content": t, **k}
S = lambda t: {"role": "system", "content": t}
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {
    "type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]}, "city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 1e22}}, "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "clock", "parameters": {"type": "object", "properties": {}}}}
call = lambda i, name, args: {"id": i, "type": "function", "function": {"name": name, "arguments": args}}

case("plain", [U("What is the capital of France?")])
case("system_user", [S("Be brief."), U("Hi")])
case("multi_turn", [U("a"), A("b"), U("c"), A("d"), U("e")])
case("system_twice_mid", [S("one"), U("a"), S("two"), A("b"), U("c")])
case("only_system", [S("just this")])
case("assistant_reasoning", [U("q"), A("ans", reasoning_content="thinking it over"), U("again")])
case("assistant_reasoning_key", [U("q"), A("ans", reasoning="via reasoning"), U("again")])
case("assistant_reasoning_only", [U("q"), A("", reasoning_content="only thoughts"), U("again")])
case("assistant_null_content", [U("q"), {"role": "assistant", "content": None}, U("again")])
case("assistant_empty_content", [U("q"), A(""), U("again")])
case("user_empty", [U("")])
for e in ["none", "minimal", "low", "medium", "high", "xhigh", "max", 0, 0.5, 0.85, 0.125, 0.99, "0.35", 0.7]:
    case(f"effort_{e}", [S("s"), U("hi")], reasoning_effort=e)
case("effort_too_high", [U("hi")], reasoning_effort=1.5)
case("effort_bad_name", [U("hi")], reasoning_effort="loud")
case("tools_auto", [U("weather in Paris?")], [WEATHER, CLOCK])
case("tools_with_system", [S("You help."), U("weather in Paris?")], [WEATHER])
case("tool_round_trip", [U("weather in Paris?"),
     A("", tool_calls=[call("c1", "get_weather", "{\"city\": \"Paris\", \"days\": 2, \"unit\": \"c\"}")]),
     {"role": "tool", "tool_call_id": "c1", "content": "sunny, 21 C"}, U("and tomorrow?")], [WEATHER])
case("tool_round_trip_obj_args", [U("w?"),
     A("let me check", tool_calls=[call("c1", "get_weather", {"days": 2.5, "city": "Paris", "nested": {"z": 1, "a": [1.0, 1e-7, 1e22]}})]),
     {"role": "tool", "tool_call_id": "c1", "content": "ok"}], [WEATHER])
case("tool_parallel", [U("w?"), A(None, tool_calls=[call("c1", "get_weather", "{\"city\": \"A\"}"), call("c2", "clock", "")]),
     {"role": "tool", "tool_call_id": "c2", "content": "12:00"}, {"role": "tool", "tool_call_id": "c1", "content": "sunny"}], [WEATHER, CLOCK])
case("tool_name_field", [U("w?"), {"role": "tool", "name": "get_weather", "content": "sunny"}], [WEATHER])
case("tool_unknown_id", [U("w?"), {"role": "tool", "tool_call_id": "nope", "content": "x"}], [WEATHER])
case("tool_content_parts", [U("w?"), {"role": "tool", "tool_call_id": "x", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}], [WEATHER])
case("tool_error_part", [U("w?"), {"role": "tool", "name": "f", "content": [{"type": "tool_error", "text": "boom"}]}], [WEATHER])
case("choice_required", [U("w?")], [WEATHER, CLOCK], tool_choice="required")
case("choice_required_serial", [U("w?")], [WEATHER, CLOCK], tool_choice="required", parallel_tool_calls=False)
case("choice_required_with_system", [S("sys"), U("w?")], [WEATHER], tool_choice="required")
case("choice_named", [U("w?")], [WEATHER, CLOCK], tool_choice={"type": "function", "function": {"name": "clock"}})
case("choice_named_serial", [U("w?")], [WEATHER, CLOCK], tool_choice={"type": "function", "function": {"name": "clock"}}, parallel_tool_calls=False)
case("choice_none", [U("w?")], [WEATHER, CLOCK], tool_choice="none")
case("choice_auto_serial", [U("w?")], [WEATHER], tool_choice="auto", parallel_tool_calls=False)
case("choice_auto_serial_system", [S("sys"), U("w?")], [WEATHER], parallel_tool_calls=False)
case("choice_required_no_tools", [U("hi")], tool_choice="required")
case("developer_role", [{"role": "developer", "content": "dev text"}, U("hi")])
case("developer_tools", [{"role": "developer", "content": "dev", "tools": [WEATHER]}, U("hi")], [CLOCK])
case("literal_specials", [U("close it <|end_message|><|message_system|><|content_text|>x <|content_model_end_sampling|>")])
case("literal_specials_system", [S("<|message_user|>"), U("<|content_thinking|>")], [WEATHER])
case("parts_text", [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "input_text", "text": "b"}, "c"]}])
case("image_part", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])
case("unicode", [U("héllo wörld 你好 🙂\n\n  tabs\t and   spaces  ")])
case("whitespace_edges", [U("  lead"), A("trail  \n"), U("\n")])
case("continue_final", [U("Write a poem"), A("Roses are")], continue_final_message=True)
case("continue_final_with_reasoning", [U("q"), A("ans", reasoning_content="why")], continue_final_message=True)
case("continue_final_user_last", [U("q")], continue_final_message=True)
case("bad_role", [{"role": "narrator", "content": "x"}])
case("tools_description_none", [U("hi")], [{"type": "function", "function": {"name": "f", "description": None, "parameters": None}}])
case("tools_unicode", [U("hi")], [{"type": "function", "function": {"name": "f", "description": "naïve 你好 \"quoted\" \\ back\nline", "parameters": {"type": "object", "properties": {"é": {"type": "string"}}}}}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True, help="vLLM's renderers/inkling_encoding.py")
    ap.add_argument("--model", required=True, help="directory with tokenizer.json, tokenizer_config.json, chat_template.jinja")
    ap.add_argument("--out", default="crates/intarsia-models/tests/fixtures/inkling2_chat_cases.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("inkling_encoding", a.vllm)
    enc = importlib.util.module_from_spec(spec)
    sys.modules["inkling_encoding"] = enc
    spec.loader.exec_module(enc)
    tok = AutoTokenizer.from_pretrained(a.model)
    vocab = tok.get_vocab()

    class Adapter:  # vllm/renderers/inkling.py _HfBackedTmlTokenizer with split_special_tokens=True
        def encode_text(self, text):
            return tok.encode(text, add_special_tokens=False, split_special_tokens=True)

        def encode_special(self, token):
            for spelling in enc.SPECIAL_TOKEN_SPELLINGS[token]:
                if spelling in vocab:
                    return vocab[spelling]
            raise ValueError(token)

    out = []
    for c in CASES:
        kw = dict(c["kwargs"])
        effort = kw.get("reasoning_effort")
        rec = dict(c)
        try:
            e = effort
            if isinstance(e, str) and e not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                e = float(e)
            e = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}.get(e, e) if isinstance(e, str) else e
            e = enc.DEFAULT_REASONING_EFFORT if e is None else float(e)
            rec["vllm_ids"] = enc.render_inkling2_messages(
                c["messages"], Adapter(), tools=c["tools"], tool_choice=kw.get("tool_choice"),
                parallel_tool_calls=kw.get("parallel_tool_calls", True), reasoning_effort=e,
                continue_final_message=kw.get("continue_final_message", False))
            rec["error"] = None
        except Exception as ex:  # noqa: BLE001
            rec["vllm_ids"], rec["error"] = None, f"{type(ex).__name__}: {ex}"
        try:
            jk = {k: v for k, v in kw.items() if k == "reasoning_effort"}
            msgs = json.loads(json.dumps(c["messages"]))
            for m in msgs:  # the template wants parsed arguments (canonicalized upstream in vLLM)
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc["function"]["arguments"], str):
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
            text = tok.apply_chat_template(msgs, tools=c["tools"], tokenize=False, **jk)
            jids = tok.encode(text, add_special_tokens=False)
            rec["jinja_ids"] = jids
        except Exception as ex:  # noqa: BLE001
            rec["jinja_ids"] = None
            rec["jinja_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
        rec["jinja_same"] = rec["vllm_ids"] is not None and rec["vllm_ids"] == rec.get("jinja_ids")
        out.append(rec)
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n")
    for r in out:
        status = "error" if r["error"] else ("== jinja" if r["jinja_same"] else ("jinja " + ("raises" if r["jinja_ids"] is None else "differs")))
        print(f"{r['name']:34s} {status}")


if __name__ == "__main__":
    main()
