#!/usr/bin/env python3
"""MMTP gate request: req2.py PORT PROMPT_FILE SUFFIX OUT (greedy, 256 tokens, ignore_eos); writes the text, prints usage."""
import json, sys, urllib.request
port, pf, suf, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = {"model": "intarsia", "prompt": open(pf).read() + suf, "max_tokens": 256, "temperature": 0, "ignore_eos": True}
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
j = json.load(r)
open(out, "w").write(j["choices"][0]["text"])
print(out.rsplit("/", 1)[-1], j.get("usage"))
