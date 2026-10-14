#!/usr/bin/env python3
"""Per-call decode attention time from INTARSIA_METAL_PROFILE=kernel:<file> traces (run-checks.sh).

usage: attn-trace.py <trace-dir> <ctx-tag>...    (files kernel-pre-<tag>.jsonl, kernel-new-<tag>.jsonl)

A record is one launch in its own command buffer, so its duration is kernel time plus Metal's fixed per-command-buffer cost
(a few us, the same before and after); `attn_bw` gives the in-stream numbers. Takes the last 128 decode steps (a step starts
at the embedding record with t == 1), prints per-call attention us, calls/token, ms/token and GB/s over the K+V bytes a
call has to read (2 * Nkv * visible * D * 2, Qwen3-8B: 8 kv heads, head_dim 128, visible = prompt tokens + step).
"""
import json, sys

ATTN = ("attn_decode_bf16", "attn_decode_bf16_splitkv")
NKV, D = 8, 128


def load(p):
    return [json.loads(l) for l in open(p)]


def steps(recs, n=128):
    idx = [i for i, r in enumerate(recs) if r["op"] == "embedding_bf16" and r["t"] == 1]
    segs = [recs[a:b] for a, b in zip(idx, idx[1:] + [len(recs)])]
    return segs[-n:]


def prompt_tokens(path):
    # `generate` prints the prompt token count on stderr ("prompt tokens: N" or similar); fall back to the tag
    try:
        for l in open(path):
            if "prompt" in l and "token" in l:
                for w in l.replace(",", " ").split():
                    if w.isdigit():
                        return int(w)
    except OSError:
        pass
    return None


d = sys.argv[1]
print("| context | build | attention launches/token | us/call (median) | p10-p90 | ms/token (median x calls) | K+V MB/call | GB/s (median) |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for tag in sys.argv[2:]:
    nominal = {"1k": 1000, "4k": 4090, "16k": 16300}.get(tag)
    for b in ("pre", "new"):
        segs = steps(load(f"{d}/kernel-{b}-{tag}.jsonl"))
        durs = [(r["e"] - r["s"]) * 1e6 for s in segs for r in s if r["op"] in ATTN]
        calls = len(durs) / len(segs)
        sd = sorted(durs)
        mean = sd[len(sd) // 2]
        p10, p90 = sd[len(sd) // 10], sd[len(sd) * 9 // 10]
        n = prompt_tokens(f"{d}/{b}-{tag}.err") or nominal
        mb = 2 * NKV * (n + len(segs) // 2) * D * 2 / 1e6
        print(f"| {tag} ({n} tok) | {b} | {calls:.0f} | {mean:.1f} | {p10:.1f}-{p90:.1f} | {mean * calls / 1e3:.2f} | {mb:.1f} | {mb * 1e6 / (mean * 1e-6) / 1e9:.0f} |")
