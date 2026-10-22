#!/usr/bin/env python3
"""fx6 diagnostic (finding 3, 8x MMLU latency): send the request shape localeval uses for deepeval MMLU (chat, temperature 0,
enable_thinking=false, response_format json_schema MultipleChoiceSchema) to one server and print content, usage and finish_reason.
usage: probe-mmlu.py <base url incl /v1> <model> <out file>"""
import json, sys, time, urllib.request

url, model, out = sys.argv[1:4]
SCHEMA = {"properties": {"answer": {"enum": ["A", "B", "C", "D"], "title": "Answer", "type": "string"}},
          "required": ["answer"], "title": "MultipleChoiceSchema", "type": "object"}
QS = [
    "The following are multiple choice questions (with answers) about virology.\n\nWhich of the following is a DNA virus?\nA. Poliovirus\nB. Influenza virus\nC. Herpes simplex virus\nD. HIV\n\nAnswer:",
    "The following are multiple choice questions (with answers) about high school physics.\n\nA 2 kg object accelerates at 3 m/s^2. What net force acts on it?\nA. 1.5 N\nB. 5 N\nC. 6 N\nD. 9 N\n\nAnswer:",
    "The following are multiple choice questions (with answers) about business ethics.\n\nWhich of these best describes a conflict of interest?\nA. Two employees disagree\nB. A personal interest that could bias professional judgement\nC. A company loses money\nD. A supplier raises prices\n\nAnswer:",
]
res = []
for mode in ("json_schema", "none"):
    for q in QS:
        body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 1024, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}}
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "MultipleChoiceSchema", "schema": SCHEMA}}
        req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            c = r["choices"][0]
            res.append({"mode": mode, "s": round(time.time() - t, 3), "content": c["message"].get("content"), "finish": c.get("finish_reason"),
                        "usage": r.get("usage"), "reasoning": c["message"].get("reasoning_content")})
        except Exception as e:
            res.append({"mode": mode, "error": str(e)[:300], "s": round(time.time() - t, 3)})
open(out, "w").write("\n".join(json.dumps(x) for x in res) + "\n")
print("\n".join(json.dumps(x)[:300] for x in res))
