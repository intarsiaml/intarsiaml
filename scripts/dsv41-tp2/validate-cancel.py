#!/usr/bin/env python3
"""Lane dsv41/validate, check 1 follow-up: a client close that lands the cancel after chunk 1 (state 2048 rows), the review's case
(docs/results/2026-11-05-dsv41-prefill3.md, "Review": A cancelled after chunk 1, B = A[:2040] + 50).

  validate-cancel.py URL OUT.json ARM TOKENIZER DELAY...

Sweeps the close delays (seconds) and, for each, separator request, streaming A (6000 tokens, as validate-client.py), close after the
delay, wait 8 s, probe A[:2048] + 200 whose cached_tokens is the state length (2048 / 4096 / 0 = never started or completed). For the
first delay that leaves 2048 rows: cold B3 (separator, B3), then the cancelled A at that delay followed by B3 = A[:2040] + 50, and
the same with B3'' = A[:2040] + 50 on a second repetition (determinism of the landing). Prediction (replay.rs reuse_ok): REPLAY=0
resumes at 2040; default and skip0 prefill cold when replay_from (the cancelled state's: chunk 1 is dead with the skip, so 2048;
without it 1920) is above the floor of B3 (1962): default cold (0), skip0 resumes (2040).
"""
import importlib.util
import json
import os
import sys
import time

spec = importlib.util.spec_from_file_location("vc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate-client.py"))
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)


def main(url, out, arm, tok_path, delays):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    bos = tok.token_to_id("<｜begin▁of▁sentence｜>")
    ids = [bos] + tok.encode(open(vc.DOC, encoding="utf-8").read(), add_special_tokens=False).ids
    a, bs = vc.variants(ids)
    b3 = bs["B3_A[:2040]+50"]
    n = [1000]

    def evict():
        n[0] += 1
        e = tok.encode(f"yq{n[0]} separator number {n[0]}, sharing nothing with any other prompt. " * 4, add_special_tokens=False).ids
        vc.completion(url, e[:64], 2)

    res = {"arm": arm, "sweep": [], "landing": None}
    probe = a[:2048] + ids[vc.NA + 900:vc.NA + 1100]
    for d in delays:
        evict()
        m0 = vc.metrics(url)
        vc.stream_cancel(url, a, d)
        time.sleep(8)
        m1 = vc.metrics(url)
        p = vc.completion(url, probe, 1)
        res["sweep"].append({"delay_s": d, "probe_cached": p["cached"], "cancelled_delta": [m1.get(k) for k in m1 if "cancelled" in k and m1.get(k) != m0.get(k)]})
        print(arm, "delay", d, "state", p["cached"], flush=True)
        if p["cached"] == 2048:
            res["landing"] = d
            break
    if res["landing"] is not None:
        evict()
        res["cold_B3"] = vc.completion(url, b3, 64, ignore_eos=True)
        for rep in (1, 2):
            evict()
            vc.stream_cancel(url, a, res["landing"])
            time.sleep(8)
            r = vc.completion(url, b3, 64, ignore_eos=True)
            r["equal_to_cold"] = r["ids"] == res["cold_B3"]["ids"]
            r["equal_prefix"] = vc.lcp(r["ids"], res["cold_B3"]["ids"])
            res[f"B3_after_cancel_{rep}"] = r
            print(arm, "B3 after cancel", rep, "cached", r["cached"], "equal to cold", r["equal_to_cold"], "prefix", r["equal_prefix"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [float(x) for x in sys.argv[5:]])
