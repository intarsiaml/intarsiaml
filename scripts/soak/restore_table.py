#!/usr/bin/env python3
"""restore-acc: markdown table of the gsm8k_prefix arms. usage: restore_table.py <raw-dir> ARM [ARM...]  (uses <arm>.summary.json, <arm>.storediff.json)"""
import json
import os
import sys

raw = sys.argv[1]
print("| arm | n | errors | correct | acc | truncated (512) | mean output tokens | prompt tokens median (min-max) | requests with cached tokens | cached frac | median cached tokens (hits) | store hits / misses / local_wins | tokens_saved | rejected inserts | paused steps | median latency s | wall s | concurrency |")
print("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for a in sys.argv[2:]:
    p = f"{raw}/{a}.summary.json"
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    d = json.load(open(f"{raw}/{a}.storediff.json")) if os.path.exists(f"{raw}/{a}.storediff.json") else {}
    g = lambda k: int(d.get(k, 0)) if d else "n/a"  # noqa: E731
    print(f"| {a} | {s['n']} | {s['errors']} | {s['correct']} | {s['acc']:.3f} | {s['truncated']} | {s['mean_completion_tokens']} | {s['prompt_tokens_median']:.0f} ({s['prompt_tokens_min']}-{s['prompt_tokens_max']}) | "
          f"{s['requests_with_cached_tokens']} | {s['cached_frac']} | {s['cached_tokens_median_of_hits']:.0f} | {g('prefix_store_hits_total')} / {g('prefix_store_misses_total')} / {g('prefix_store_local_wins_total')} | "
          f"{g('prefix_store_tokens_saved_total')} | {g('prefix_store_rejected_total')} | {g('batch_paused_steps_total')} | {s['latency_s_median']} | {s['wall_s']:.0f} | {s['concurrency']} |")
