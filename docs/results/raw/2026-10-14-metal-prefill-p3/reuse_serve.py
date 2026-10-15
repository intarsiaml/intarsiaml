#!/usr/bin/env python3
"""Warm-vs-cold prefix reuse through `intarsia serve`: turn 2 of a two-turn chat on a ~4k prompt, answered (a) by a server
that has just served turn 1 (prefix cache warm) and (b) by a fresh server (cold). Greedy, 48 tokens; the outputs must match.
Run for the default chunk (recipe, 2048) and --prefill-chunk 256.
usage: reuse_serve.py <intarsia binary> <model dir> <prompt file> <out prefix>"""
import json, os, subprocess, sys, time, urllib.request

binary, model, prompt_file, out = sys.argv[1:5]
prompt = open(prompt_file).read()
PORT = 8140


def post(msgs, max_tokens):
    body = json.dumps({"model": "m", "messages": msgs, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"], r.get("usage")


def server(chunk, tag):
    cmd = [binary, "serve", "--model", model, "--backend", "metal", "--device", "0", "--host", "127.0.0.1", "--port", str(PORT),
           "--served-model-name", "m", "--max-seq", "8192", "--profile-every", "0", "--no-supervisor", "--no-ui"]
    if chunk:
        cmd += ["--prefill-chunk", str(chunk)]
    err = open(f"{out}-{tag}.err", "w")
    p = subprocess.Popen(cmd, stdout=open(f"{out}-{tag}.out", "w"), stderr=err)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"server {tag} died")
            time.sleep(1)
    raise SystemExit("server did not come up")


def stop(p):
    p.terminate()
    try:
        p.wait(30)
    except Exception:
        p.kill()
        p.wait()
    time.sleep(3)


results = {}
for label, chunk in [("default", None), ("c256", 256)]:
    p = server(chunk, f"{label}-warm")
    chunk_line = [l for l in open(f"{out}-{label}-warm.err") if "prefill chunk" in l][:1]
    t1 = [{"role": "user", "content": prompt}]
    reply1, u1 = post(t1, 16)
    t2 = t1 + [{"role": "assistant", "content": reply1}, {"role": "user", "content": "Now answer in one short sentence: what was the first topic?"}]
    warm, uw = post(t2, 48)
    stop(p)
    p = server(chunk, f"{label}-cold")
    cold, uc = post(t2, 48)
    stop(p)
    same = warm == cold
    results[label] = {"chunk_line": chunk_line[0].strip() if chunk_line else "", "turn1_usage": u1, "warm_usage": uw, "cold_usage": uc,
                      "warm_text": warm, "cold_text": cold, "identical": same}
    print(label, chunk_line[0].strip() if chunk_line else "", "| turn1", u1, "| warm", uw, "| cold", uc, "| identical:", same)
json.dump(results, open(f"{out}.json", "w"), indent=1)
