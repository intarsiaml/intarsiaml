#!/usr/bin/env python3
# Serve-path prefill experiment: client TTFT vs server-side prefill (from /metrics) under
# different preceding-load conditions. usage: exp.py PORT MODEL PROMPT OUT.tsv ROUNDS [idle]
import sys, json, time, urllib.request, secrets, re
port, model, pf, out, rounds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
TXT = open(pf).read()
BASE = f"http://127.0.0.1:{port}"
def metrics():
    m = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    g = lambda k: float(re.search(r"^%s\s+([0-9.eE+-]+)" % k, m, re.M).group(1))
    return g("intarsia_last_prefill_tokens_per_second"), g("intarsia_last_decode_tokens_per_second")
def req(prompt, gen):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if gen > 1:
        body["ignore_eos"] = True; body["min_tokens"] = gen
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); T0E = time.time(); ttft = None; ntok = 0; pt = 0; t_last = t0
    with urllib.request.urlopen(r, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("usage"): pt = j["usage"]["prompt_tokens"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None: ttft = time.perf_counter() - t0
                ntok += 1; t_last = time.perf_counter()
    tot = time.perf_counter() - t0
    dec = (ntok - 1) / (t_last - t0 - ttft) if ntok > 1 and t_last - t0 - ttft > 0 else 0
    pfs, ds = metrics()
    return dict(t0e=T0E, ttft=ttft or tot, tot=tot, pt=pt, srv_pf_toks=pfs, srv_pf_ms=(pt / pfs * 1e3 if pfs else 0), dec=dec)
P4 = lambda: secrets.token_hex(6) + " " + TXT
SH = lambda: secrets.token_hex(6) + " Write a long story about a lighthouse keeper."
f = open(out, "w"); f.write("round\tcond\tstep\tttft_ms\tsrv_pf_ms\tpt\tdec_toks\twall\tt0_epoch\tttft_s\n")
def rec(rnd, cond, step, r):
    f.write(f"{rnd}\t{cond}\t{step}\t{r['ttft']*1e3:.0f}\t{r['srv_pf_ms']:.0f}\t{r['pt']}\t{r['dec']:.1f}\t{time.strftime('%H:%M:%S')}\t{r['t0e']:.3f}\t{r['ttft']:.4f}\n"); f.flush()
    print(rnd, cond, step, f"ttft={r['ttft']*1e3:.0f} srv_pf={r['srv_pf_ms']:.0f} pt={r['pt']} dec={r['dec']:.1f}", flush=True)
def cond_A(rnd):  # 4k right after a 256-token decode burst
    req(SH(), 256); rec(rnd, "A_after_decode", "p4k", req(P4(), 1))
def cond_B(rnd):  # 4k after decode + 5s idle
    req(SH(), 256); time.sleep(5); rec(rnd, "B_decode_idle5", "p4k", req(P4(), 1))
def cond_C(rnd):  # back-to-back 4k, max_tokens 1
    req(P4(), 1); rec(rnd, "C_b2b_prefill", "p4k", req(P4(), 1))
def cond_E(rnd):  # localeval pattern: 4k+256gen then 4k+256gen
    req(P4(), 256); rec(rnd, "E_leval_pattern", "p4k+256", req(P4(), 256))
def cond_G1(rnd):
    req(SH(), 256); time.sleep(1); rec(rnd, "G1_decode_idle1", "p4k", req(P4(), 1))
def cond_G2(rnd):
    req(SH(), 256); time.sleep(2); rec(rnd, "G2_decode_idle2", "p4k", req(P4(), 1))
def cond_H(rnd):  # short decode (32 tokens), no gap
    req(SH(), 32); rec(rnd, "H_decode32", "p4k", req(P4(), 1))
def cond_F(rnd):  # 4k after 30s idle
    time.sleep(30); rec(rnd, "F_idle30", "p4k", req(P4(), 1))
conds = ([cond_A, cond_B, cond_C, cond_E, cond_G1, cond_G2, cond_H] if len(sys.argv) > 6 and sys.argv[6] == "more" else [cond_A, cond_B, cond_C, cond_E]) + ([cond_F] if len(sys.argv) > 6 and "idle" in sys.argv[6:] else [])
for _ in range(2): req(P4(), 1)
for rnd in range(rounds):
    k = rnd % len(conds)
    for c in conds[k:] + conds[:k]: c(rnd)
