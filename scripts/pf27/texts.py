#!/usr/bin/env python3
"""Lane TPB: greedy texts of the 18 gate prompts from one running `serve`, for spec-on == spec-off and batched == solo comparisons.
usage: texts.py BASE_URL MODEL PROMPT_DIR [name...]
Per prompt (short-0..5, 4k-0..5, 16k-0..5): one chat request, temperature 0, 256 forced tokens (ignore_eos, min_tokens), thinking off. Prints
"<name> <md5 of the text> <n chars>"; the texts go to PROMPT stdout only as md5s (compare two runs with diff)."""
import hashlib, json, sys, time, urllib.request

base, model, d = sys.argv[1], sys.argv[2], sys.argv[3]
names = sys.argv[4:] or [f"{k}-{i}" for k in ("short", "4k", "16k") for i in range(6)]
for n in names:
    prompt = open(f"{d}/{n}.txt").read()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "min_tokens": 256, "ignore_eos": True,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    text = j["choices"][0]["message"]["content"] or ""
    u = j.get("usage", {})
    print(f"{n} {hashlib.md5(text.encode()).hexdigest()[:12]} {len(text)} chars, {u.get('completion_tokens')} tokens, {time.time() - t0:.1f}s", flush=True)
