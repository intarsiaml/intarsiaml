#!/usr/bin/env python3
"""Lane flipprep: table of tools.py results. tools_table.py <tools dir>   (reference arm for the signature check: A)"""
import glob, json, sys
d = sys.argv[1]
runs = {}
for f in sorted(glob.glob(f"{d}/tools-*.json")):
    j = json.load(open(f)); s = j["summary"]
    if s.get("tool_choice") or s.get("mix"):
        continue  # the plain table: tool_choice auto, one request per stream
    runs[(s["arm"], s["temperature"], s["streams"])] = j
print("| arm | T | streams | tool call / single | calls | args valid JSON | names known | finish | tokens | verify passes | accept rate | tokens/pass |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (arm, t, st), j in sorted(runs.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
    s = j["summary"]
    print(f"| {arm} | {t} | {st} | {s['single_with_tool_call']}/{s['single']} | {s['tool_calls_total']} | {s['all_arguments_valid_json']} | {s['all_names_known']} | {s['finish_reasons']} | {s['completion_tokens']} | {s['verify_passes']:.0f} | {s['accept_rate']} | {s['tokens_per_pass']} |")
print()
for (arm, t, st), j in sorted(runs.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
    ref = runs.get(("A", t, st))
    if ref is None or arm == "A":
        continue
    same = sum(1 for x, y in zip(j["each"], ref["each"]) if x and y and x.get("signature") == y.get("signature"))
    print(f"{arm} T={t} streams={st}: {same}/{len(j['each'])} replies' tool calls / text identical to A's" + ("  (T=0: must be all)" if t == 0 else ""))
