#!/usr/bin/env python3
"""hrr: cached prefix + TTFT, old (INTARSIA_PREFIX_REUSE=any) vs new, on a running serve. Usage: probe.py PORT OUT.json"""
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{port}/v1/completions"
TXT = open("docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt").read()
CPT = len(TXT) / 16384.0  # chars per token (approx)
FLUSH = "Zebra crossing facts and other unrelated matters. " * 20

def req(prompt, n=48):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": n, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    r = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; cached = None; ptoks = None
    with urllib.request.urlopen(r, timeout=600) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            ch = j.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None: ttft = time.time() - t0
                text += ch[0]["text"]
            u = j.get("usage")
            if u:
                ptoks = u.get("prompt_tokens"); cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(ttft_ms=round(1000 * ttft, 1) if ttft else None, cached=cached, prompt_tokens=ptoks, text=text)

def flush():
    r = req(FLUSH, 2); return r

res = {}
# 1. the reported case: B = A + suffix, A = 4096-token prompt
# calibrate chars per token on this tokenizer so that A is exactly 4096 tokens
probe = req(TXT[:20000], 1); CPT = 20000 / probe["prompt_tokens"]
def exact(n):
    c = int(n * CPT); seen = {}
    def cnt(k):
        if k not in seen: seen[k] = req(TXT[:k], 1)["prompt_tokens"]
        return seen[k]
    for _ in range(12):  # walk to the neighbourhood
        d = n - cnt(c)
        if abs(d) <= 3: break
        c += int(d * CPT * 0.8)
    for k in range(c - 60, c + 60):  # then scan chars: the count is not monotone at char level
        if cnt(k) == n: return TXT[:k]
    raise SystemExit("could not hit exact length")
A = exact(4096); B = A + " Summarize the above in one paragraph."
flush(); res["fresh_B"] = req(B, 256)           # first request on a fresh server (after calibration requests; each cold)
flush(); res["A"] = req(A, 256)
res["B_after_A"] = req(B, 256)
res["A_after_B"] = req(A, 256)
flush(); res["cold_B"] = req(B, 256)
res["B_after_A_equals_fresh"] = res["B_after_A"]["text"] == res["fresh_B"]["text"]
res["cold_equals_fresh"] = res["cold_B"]["text"] == res["fresh_B"]["text"]
res["A_after_B_equals_A"] = res["A_after_B"]["text"] == res["A"]["text"]
# 2. chat continuation: turn1 = L tokens, reply, turn2 = turn1 + reply + follow-up
Q = "\n\nUser: Now list the three most important points above as bullets.\nAssistant:"
for L in (() if len(sys.argv) > 3 else (1500, 3000, 5000, 8500, 12000)):
    X = TXT[: int(L * CPT)] + "\n\nUser: Summarize the passage.\nAssistant:"
    flush(); t1 = req(X, 32)
    T2 = X + t1["text"] + Q
    warm = req(T2); flush(); cold = req(T2)
    res[f"chat_{L}"] = dict(turn1=t1, warm=warm, cold=cold, text_equal=warm["text"] == cold["text"],
                            lost=(t1["prompt_tokens"] + 32) - (warm["cached"] or 0))
json.dump(res, open(out, "w"), indent=1)
def row(k, r): return f'{k:<14} ptoks {r["prompt_tokens"]:>6} cached {str(r["cached"]):>6} ttft {r["ttft_ms"]:>8} ms'
for k in ("fresh_B", "A", "B_after_A", "A_after_B", "cold_B"): print(row(k, res[k]))
print("B_after_A == fresh:", res["B_after_A_equals_fresh"], " cold == fresh:", res["cold_equals_fresh"])
for L in (() if len(sys.argv) > 3 else (1500, 3000, 5000, 8500, 12000)):
    c = res[f"chat_{L}"]; print(f"chat_{L}: turn1 ptoks {c['turn1']['prompt_tokens']} | warm cached {c['warm']['cached']} ttft {c['warm']['ttft_ms']} | cold ttft {c['cold']['ttft_ms']} | text_equal {c['text_equal']}")
