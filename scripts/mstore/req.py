#!/usr/bin/env python3
"""Lane mstore: one streamed /v1/chat/completions request; prints one JSON line.

usage: req.py URL MESSAGES_JSON TAG [max_tokens]
Measures TTFT (first streamed content or reasoning chunk), the gaps between SSE chunks (a chunk can carry several tokens
under MTP: the plain per-chunk gap, not a weighted ITL), decode tok/s = (completion_tokens - 1) / (last chunk - first
chunk), usage (prompt, cached, completion tokens), and the server's spec and prefix-store counter deltas (/metrics).
Greedy, thinking off, ignore_eos, so every arm generates the same number of tokens."""
import hashlib, json, sys, time, urllib.request

url, mf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 256
WANT = ("intarsia_spec_steps_total", "intarsia_spec_proposed_tokens_total", "intarsia_spec_accepted_tokens_total",
        "intarsia_prefix_store_hits_total", "intarsia_prefix_store_tokens_saved_total", "intarsia_prefix_store_restore_seconds_total",
        "intarsia_prefix_store_bytes_restored_total", "intarsia_prefix_store_inserts_total", "intarsia_prefix_store_insert_seconds_total",
        "intarsia_prefix_store_settle_seconds_total")


def metrics():
    out = {}
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    out[parts[0]] = float(parts[1])
    except Exception as e:  # metrics are a diagnostic: the request still counts
        out["error"] = str(e)
    return out


def pick(m):
    r = {}
    for k, v in m.items():
        base = k.split("{")[0]
        if base in WANT or base == "intarsia_prefix_store_tier_hits_total":
            r[k] = v
    return r


msgs = json.load(open(mf))
body = json.dumps({"model": "intarsia", "messages": msgs, "max_tokens": mt, "temperature": 0, "stream": True, "ignore_eos": True,
                   "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}).encode()
m0 = pick(metrics())
t0 = time.perf_counter()
ts, usage, text = [], None, []
with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900) as r:
    for line in r:
        if not line.startswith(b"data: ") or b"[DONE]" in line:
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        ch = d.get("choices") or []
        if ch:
            delta = ch[0].get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                ts.append(time.perf_counter() - t0)
                text.append(piece)
time.sleep(0.3)  # let the settle after Done land in the counters
m1 = pick(metrics())
gaps = sorted(b - a for a, b in zip(ts, ts[1:]))


def pct(p):
    return round(gaps[min(len(gaps) - 1, int(p * len(gaps)))] * 1e3, 1) if gaps else None


comp = (usage or {}).get("completion_tokens", 0)
ptd = (usage or {}).get("prompt_tokens_details") or {}
delta = {k: round(m1.get(k, 0) - m0.get(k, 0), 4) for k in sorted(set(m0) | set(m1)) if m1.get(k, 0) != m0.get(k, 0)}
prop = sum(v for k, v in delta.items() if k.startswith("intarsia_spec_proposed"))
acc = sum(v for k, v in delta.items() if k.startswith("intarsia_spec_accepted"))
print(json.dumps({
    "tag": tag, "ttft_ms": round(ts[0] * 1e3, 1) if ts else None, "total_ms": round(ts[-1] * 1e3, 1) if ts else None,
    "chunks": len(ts), "gap_p50_ms": pct(0.5), "gap_p99_ms": pct(0.99), "gap_max_ms": round(gaps[-1] * 1e3, 1) if gaps else None,
    "decode_tok_s": round((comp - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and comp > 1 else None,
    "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": ptd.get("cached_tokens", 0), "completion_tokens": comp,
    "acceptance": round(acc / prop, 4) if prop else None, "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16],
    "metrics_delta": delta}))
