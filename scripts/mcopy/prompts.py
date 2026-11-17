#!/usr/bin/env python3
"""mcopy: writes the speed / ids cells to OUT (default scripts/mcopy/cells): <name>.json = {messages, max_tokens} for the served
client (cell.py), and <name>.txt = the single user message, for `intarsia generate --chat -p` (the ids gate). Deterministic.
  copy3k      copy-heavy: re-emit a ~3K-token source file with one small edit
  agent-batch agent turn with a call per site: several edit_file JSON objects, each quoting lines of the tool result
  agent-edit  agent turn: tool-call JSON (edit_file with the whole function as old_string and the renamed one as new_string)
  json-xform  JSON in, the same JSON out with a field changed in some records
  chat-short  control: plain explanation, nothing to copy
  reason      control: step-by-step arithmetic reasoning
  code-gen    control: new code with repeated boilerplate (tests), nothing to copy from the prompt
"""
import json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent / "cells"
OUT.mkdir(parents=True, exist_ok=True)


def put(name, messages, max_tokens):
    (OUT / f"{name}.json").write_text(json.dumps({"messages": messages, "max_tokens": max_tokens}))
    if len(messages) == 1:
        (OUT / f"{name}.txt").write_text(messages[0]["content"])


# 1. copy-heavy: a source file of about 3K tokens, one small edit
src = (REPO / "scripts/ab35b/toolsmoke.py").read_text()
put("copy3k", [{"role": "user", "content":
    "Here is a Python file.\n\n```python\n" + src + "```\n\nOutput the complete file again, exactly as it is, except for one change: "
    "in the function `base`, the default `max_tokens` 512 becomes 640. Output only the code block, nothing else."}], 3800)

# 2. agent turn: the tool result holds a file; the next call quotes a function of it as old_string and renames it
fn_src = (REPO / "scripts/ab35b/sampled.py").read_text()
SYSTEM = ("You are a coding agent. You act only by emitting one JSON tool call per turn, as a single JSON object and nothing else. Tools:\n"
          '{"tool": "read_file", "path": "<path>"}  returns the file text.\n'
          '{"tool": "edit_file", "path": "<path>", "old_string": "<exact text to replace, copied verbatim from the file>", '
          '"new_string": "<replacement>"}  replaces old_string once.')
put("agent-edit", [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "In scripts/ab35b/sampled.py, rename the function `one` to `one_stream`, the whole function including its body and every use of the name inside it. First read the file."},
    {"role": "assistant", "content": '{"tool": "read_file", "path": "scripts/ab35b/sampled.py"}'},
    {"role": "user", "content": "Tool result for read_file:\n```\n" + fn_src + "```\nNow make the edit: emit the edit_file call whose old_string is the complete definition of `one` copied exactly from the file, and whose new_string is that definition renamed."},
], 1200)

# 2b. agent turn with several calls: the file of a tool result has call sites of three helpers; one edit_file call per call site
helpers = "\n\n".join(
    f"def {n}(x, cfg):\n    # {n} step of the pipeline\n    y = x * cfg['scale'] + cfg['offset']\n    if y < 0:\n        raise ValueError('{n}: negative value')\n    return y"
    for n in ("fetch_rows", "parse_row", "emit_row"))
body = "\n".join(
    f"    rows_{i} = fetch_rows({i}, cfg)\n    item_{i} = parse_row(rows_{i}, cfg)\n    out_{i} = emit_row(item_{i}, cfg)\n    log.append(out_{i})"
    for i in range(1, 8))
pipe = "import logging\n\n" + helpers + "\n\n\ndef run_pipeline(cfg, log):\n" + body + "\n    return log\n"
put("agent-batch", [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "In pipeline.py rename fetch_rows to load_rows, parse_row to decode_row and emit_row to write_row, at every definition and call site. First read the file."},
    {"role": "assistant", "content": '{"tool": "read_file", "path": "pipeline.py"}'},
    {"role": "user", "content": "Tool result for read_file:\n```\n" + pipe + "```\nEmit the edit_file calls now, one JSON object per line and nothing else: one call per definition and one per group of the three call lines of run_pipeline (old_string = those three lines copied exactly)."},
], 1500)

# 3. JSON transform: records in, the same records out with `status` set to "archived" for region eu
recs = []
for i in range(34):
    recs.append({"id": 1000 + 7 * i, "name": f"widget-{i:03d}", "region": ["eu", "us", "apac"][i % 3], "status": ["active", "pending"][i % 2],
                 "tags": [f"t{i % 5}", f"t{(i * 3) % 7}"], "price": round(9.5 + 1.25 * i, 2), "stock": 40 + (i * 13) % 90})
put("json-xform", [{"role": "user", "content":
    "Here is a JSON array of records.\n\n```json\n" + json.dumps(recs, indent=2) + "\n```\n\nOutput the same array, formatted the same way, "
    'with "status" changed to "archived" in every record whose "region" is "eu". Output only the JSON, in a code block.'}], 2600)

# 4. controls
put("chat-short", [{"role": "user", "content": "Explain, in a few paragraphs, how a hash table handles collisions and why the load factor matters for lookup time."}], 400)
put("reason", [{"role": "user", "content":
    "A train leaves a station at 9:10 and travels 84 km at 60 km/h, then stops for 12 minutes, then travels another 45 km at 90 km/h. "
    "At what time does it arrive? Think step by step and show the arithmetic."}], 500)
put("code-gen", [{"role": "user", "content":
    "Write a Python module with an LRU cache class (get, put, delete, __len__) built on a doubly linked list and a dict, followed by a pytest "
    "file with six small tests that each construct the cache, call put and get a few times and assert the result."}], 800)
print("cells in", OUT)
