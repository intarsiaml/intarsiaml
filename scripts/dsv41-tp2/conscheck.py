#!/usr/bin/env python3
"""Lane dsv41/sampled-spec: constrained requests (json_schema) at a temperature: every reply must parse and match the schema.

    conscheck.py URL OUT.json TEMPERATURE [--n 8]

Chat requests with a json_schema response_format (thinking off, max_tokens 300, seed 500 + i). Prints per request the tokens, tok/s and
whether the reply is valid JSON with the schema's keys and types; writes the same to OUT.json. Exit 1 when a reply is invalid.
"""
import argparse
import http.client
import json
import sys
import time
import urllib.parse

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "age", "tags", "address"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}},
                         "address": {"type": "object", "additionalProperties": False, "required": ["city", "zip"],
                                     "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}}}}
SUBJECTS = ["a retired lighthouse keeper", "a robotics student", "a jazz pianist", "a long-haul pilot", "a pastry chef", "a glaciologist",
            "a bike courier", "a high-school chemistry teacher", "a harbour pilot", "a violin maker"]


def valid(o):
    return (isinstance(o, dict) and set(o) == {"name", "age", "tags", "address"} and isinstance(o["name"], str) and isinstance(o["age"], int)
            and isinstance(o["tags"], list) and all(isinstance(t, str) for t in o["tags"]) and isinstance(o["address"], dict)
            and set(o["address"]) == {"city", "zip"} and all(isinstance(v, str) for v in o["address"].values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out"); ap.add_argument("temperature", type=float); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    rows, bad = [], 0
    for i in range(a.n):
        body = {"model": "dsv41-intarsia", "messages": [{"role": "user", "content": f"Invent a person: {SUBJECTS[i % len(SUBJECTS)]}. Give a long list of tags."}],
                "temperature": a.temperature, "seed": 500 + i, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": SCHEMA}}}
        t0 = time.time()
        conn.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = json.loads(conn.getresponse().read())
        dt = time.time() - t0
        ch = r["choices"][0]
        text = ch["message"].get("content") or ""
        n = (r.get("usage") or {}).get("completion_tokens") or 0
        try:
            ok = valid(json.loads(text))
        except ValueError:
            ok = False
        bad += not ok
        rows.append({"i": i, "tokens": n, "wall_s": round(dt, 2), "tps_wall": round(n / dt, 1), "valid": ok, "finish": ch.get("finish_reason"), "text": text})
        print(f"constrained T {a.temperature} #{i}: {n} tokens in {dt:.1f} s ({n / dt:.1f} tok/s incl. prefill), valid {ok}, finish {ch.get('finish_reason')}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"constrained T {a.temperature}: {a.n - bad}/{a.n} valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
