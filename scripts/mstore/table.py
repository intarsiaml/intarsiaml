#!/usr/bin/env python3
"""Lane mstore: tables from window.sh's requests.jsonl. usage: table.py RAW_DIR [win-...]
Per window: mean over the reps of TTFT, decode tok/s, acceptance, chunk-gap p99 per arm; the catch-up cost =
(decode time of the 256-token reply restored - cold, store on, same tokens) / (TTFT cold store on - TTFT restored)."""
import json, os, sys, statistics as st
raw = sys.argv[1]
wins = sys.argv[2:] or sorted(d for d in os.listdir(raw) if d.startswith("win-"))
ARMS = [("A", "-Q", "cold, store off (today's default)"), ("B0", "-Q", "cold, store on (empty store)"),
        ("B", "-Q-disk", "restored from disk after a restart (page cache): a new turn after the restore point"),
        ("B", "-Q-repeat", "the same prompt again, restored from the host tier (all but the last tokens)")]
print("| window | arm | n | TTFT ms | decode tok/s | acceptance | gap p99 ms | cached / prompt | same text as cold store-on |")
print("|---|---|---|---|---|---|---|---|---|")
costs = []
for w in wins:
    rows = [json.loads(l) for l in open(os.path.join(raw, w, "requests.jsonl"))]
    def arm(prefix, suffix):
        return [r for r in rows if r["tag"].startswith(prefix) and r["tag"][len(prefix):].lstrip("0123456789") == suffix]
    got = {}
    ref_sha = {r["text_sha"] for r in arm("B0", "-Q")}
    for p, s, label in ARMS:
        rs = arm(p, s)
        if not rs:
            continue
        m = lambda k: st.mean(r[k] for r in rs if r[k] is not None)
        got[(p, s)] = (m("ttft_ms"), m("decode_tok_s"), rs[0]["completion_tokens"])
        same = all(r["text_sha"] in ref_sha for r in rs)
        print(f"| {w[4:]} | {label} | {len(rs)} | {m('ttft_ms'):.0f} | {m('decode_tok_s'):.1f} | {100*m('acceptance'):.1f} % | {m('gap_p99_ms'):.1f} | {rs[0]['cached_tokens']} / {rs[0]['prompt_tokens']} | {same} |")
    c = got.get(("B0", "-Q"))
    for key, label in ((("B", "-Q-disk"), "new turn, from disk"), (("B", "-Q-repeat"), "repeat, from host")):
        r = got.get(key)
        if c and r:
            n = c[2] - 1
            extra = n / r[1] - n / c[1]
            saved = (c[0] - r[0]) / 1e3
            costs.append((w[4:], label, extra, saved, extra / saved))
print()
print("| window | case | extra decode s (256 tokens) | TTFT saved s | cost / saved | break-even reply length |")
print("|---|---|---|---|---|---|")
for w, label, extra, saved, ratio in costs:
    per_tok = extra / 255
    be = f"{saved / per_tok:.0f} tokens" if per_tok > 0 else "never"
    print(f"| {w} | {label} | {extra:.2f} | {saved:.2f} | {100*ratio:.0f} % | {be} |")
