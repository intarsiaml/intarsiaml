#!/usr/bin/env python3
"""Lane mhkv: tables from window.sh's requests.jsonl. usage: table.py RAW_DIR [win-...]
Per window and bin (main, fix): mean over the visits of TTFT, decode tok/s, acceptance, server step time per arm; the
catch-up cost = (decode time of the 256-token reply restored - cold, store on, same tokens) / (TTFT cold - TTFT restored),
against lane mstore's 20 % rule. Thermal check: every request's server step time against its arm's median (flag > 10 %)."""
import json, os, sys, statistics as st

raw = sys.argv[1]
wins = sys.argv[2:] or sorted(d for d in os.listdir(raw) if d.startswith("win-"))
ARMS = [("B0", "-Q", "cold, store on (empty store)"),
        ("B", "-Q-disk", "restored from disk after a restart: a new turn after the restore point"),
        ("B", "-Q-repeat", "the same prompt again, restored from the host tier")]
flags = []
costs = []
print("| window | bin | arm | n | TTFT ms | decode tok/s | acceptance | server step ms | cached / prompt | text == main cold |")
print("|---|---|---|---|---|---|---|---|---|---|")
for w in wins:
    rows = [json.loads(l) for l in open(os.path.join(raw, w, "requests.jsonl"))]

    def arm(kind, b, suffix):
        out = []
        for r in rows:
            t = r["tag"].split("-")
            # B0-main-3-Q, B-fix-2-Q-disk
            if t[0] == kind and t[1] == b and "-" + "-".join(t[3:]) == suffix:
                out.append(r)
        return out

    ref = {r["text_sha"] for r in arm("B0", "main", "-Q")}
    got = {}
    for b in ("main", "fix"):
        for kind, suffix, label in ARMS:
            rs = arm(kind, b, suffix)
            if not rs:
                continue
            m = lambda k: st.mean(r[k] for r in rs if r.get(k) is not None)
            steps = [r["step_ms"] for r in rs if r.get("step_ms")]
            med = st.median(steps) if steps else None
            for r in rs:
                if med and r.get("step_ms") and abs(r["step_ms"] / med - 1) > 0.10:
                    flags.append(f"{w[4:]} {r['tag']}: step {r['step_ms']:.2f} ms vs arm median {med:.2f} ms")
            got[(b, kind, suffix)] = (m("ttft_ms"), m("decode_tok_s"), rs[0]["completion_tokens"])
            same = all(r["text_sha"] in ref for r in rs)
            print(f"| {w[4:]} | {b} | {label} | {len(rs)} | {m('ttft_ms'):.0f} | {m('decode_tok_s'):.1f} | {100*m('acceptance'):.1f} % | "
                  f"{m('step_ms'):.2f} | {rs[0]['cached_tokens']} / {rs[0]['prompt_tokens']} | {same} |")
        c = got.get((b, "B0", "-Q"))
        for key, label in (((b, "B", "-Q-disk"), "new turn, from disk"), ((b, "B", "-Q-repeat"), "repeat, from host")):
            r = got.get(key)
            if c and r:
                n = c[2] - 1
                extra = n / r[1] - n / c[1]
                saved = (c[0] - r[0]) / 1e3
                costs.append((w[4:], b, label, extra, saved, extra / saved))
print()
print("| window | bin | case | extra decode s (256 tokens) | TTFT saved s | cost / saved (rule: <= 20 %) | break-even reply length |")
print("|---|---|---|---|---|---|---|")
for w, b, label, extra, saved, ratio in costs:
    per_tok = extra / 255
    be = f"{saved / per_tok:.0f} tokens" if per_tok > 0 else "never (restored decodes at least as fast)"
    print(f"| {w} | {b} | {label} | {extra:+.2f} | {saved:.2f} | {100*ratio:+.0f} % | {be} |")
print()
print("thermal flags (server step time > 10 % off its arm's median):", "none" if not flags else "")
for f in flags:
    print("-", f)
