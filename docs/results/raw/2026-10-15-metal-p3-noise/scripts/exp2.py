#!/usr/bin/env python3
"""exp2.py PORT MODEL OUT.tsv ROUNDS T1,T2,... : back-to-back max_tokens=1 requests with prompt_tokens == T (calibrated), shuffled per round."""
import sys, json, time, random, urllib.request
port, model, out, rounds, targets = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(x) for x in sys.argv[5].split(",")]
BASE = f"http://127.0.0.1:{port}"
rnd = random.Random(7)
W = "the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()
words = [rnd.choice(W) + (" " if i % 17 else ".\n") for i in range(20000)]
def body(k, nonce):
    return "".join(words[:k]) , nonce
def req(k):
    nonce = "".join(rnd.choice("0123456789") for _ in range(8))
    text = f"{nonce} " + "".join(words[:k])
    b = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t0e = time.time(); ttft = None; pt = 0
    with urllib.request.urlopen(r, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("usage"): pt = j["usage"]["prompt_tokens"]
            if ttft is None and j.get("choices") and (j["choices"][0].get("delta") or {}).get("content"): ttft = time.perf_counter() - t0
    return pt, (ttft or time.perf_counter() - t0), t0e
ks = {}
for T in sorted(set(targets)):
    k = T - 20
    for _ in range(4):
        pt, _, _ = req(k)
        if pt == T: break
        k += T - pt
    assert pt == T, (T, pt)
    ks[T] = k
print("calibrated", ks, flush=True)
req(ks[targets[0]]); req(ks[targets[0]])
with open(out, "w") as f:
    f.write("round\ttarget\tpt\tttft_ms\tt0_epoch\n")
    for r in range(rounds):
        order = targets[:]; rnd.shuffle(order)
        for T in order:
            pt, ttft, t0e = req(ks[T])
            f.write(f"{r}\t{T}\t{pt}\t{ttft*1e3:.0f}\t{t0e:.3f}\n"); f.flush()
