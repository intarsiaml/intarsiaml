#!/usr/bin/env python3
"""soak2 (copy of scripts/soak/metric_diff.py plus the spec counters): store counter deltas between two /metrics scrapes. usage: metric_diff.py BEFORE AFTER  (prints JSON)"""
import json
import sys

KEYS = ["intarsia_prefix_store_lookups_total", "intarsia_prefix_store_hits_total", "intarsia_prefix_store_misses_total",
        "intarsia_prefix_store_local_wins_total", "intarsia_prefix_store_tokens_saved_total", "intarsia_prefix_store_bytes_restored_total",
        "intarsia_prefix_store_inserts_total", "intarsia_prefix_store_rejected_total", "intarsia_requests_total",
        "intarsia_tokens_prompt_total", "intarsia_tokens_cached_total", "intarsia_batch_paused_steps_total", "intarsia_requests_rejected_total",
        "intarsia_engine_failed", "intarsia_spec_steps_total", "intarsia_spec_proposed_tokens_total", "intarsia_spec_accepted_tokens_total",
        "intarsia_tokens_generated_total", 'intarsia_prefix_store_tier_evictions_total{tier="host"}']


def load(p):
    d = {}
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        k, _, v = l.rpartition(" ")
        try:
            d[k] = float(v)
        except ValueError:
            pass
    return d


b, a = load(sys.argv[1]), load(sys.argv[2])
print(json.dumps({k.replace("intarsia_", ""): a.get(k, 0) - b.get(k, 0) for k in KEYS}))
