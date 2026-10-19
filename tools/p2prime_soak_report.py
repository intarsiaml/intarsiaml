#!/usr/bin/env python3
"""Markdown tables from the lane directories of one soak window (tools/p2prime_soak.py):
    python3 tools/p2prime_soak_report.py RAWDIR/w1/gpu0-blk RAWDIR/w1/gpu1-off
"""
import json
import pathlib
import sys


def load(d):
    return json.loads((pathlib.Path(d) / "lane_summary.json").read_text())


def main():
    lanes = [(pathlib.Path(d).name, load(d)) for d in sys.argv[1:]]
    print("| lane | verdict | load wall (s) | replays | requests (replays) | cbsoak sent / ok / cancelled / error | solo-vs-replay compare | solo-vs-solo | hang / stall / 503 | decode tok/s (sentinel) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in lanes:
        reps = s.get("replays", [])
        n = sum(r["n"] for r in reps)
        cs = (s.get("cbsoak") or {}).get("counts") or {}
        v = [x for x in s["poll"]["violations"]]
        sent = f"{cs.get('sent')} / {cs.get('ok')} / {cs.get('cancelled')} / {cs.get('error')}"
        tps = ", ".join(f"{x:.1f}" for x in s.get("decode_tps_sentinel", []))
        print(f"| {name} | {'PASS' if s['pass'] else 'FAIL'} | {s.get('load_wall_s')} | {len(reps)} | {n} | {sent} | rc {s.get('compare_rc')} | rc {s.get('compare_ref2_rc')} | {len(v)} | {tps} |")
    print()
    for name, s in lanes:
        print(f"### {name}: replays\n")
        print("| replay | B | stream | arrival x | cases | wall s | peak client inflight | cancelled | errors | slow >60 s | latency p50 / p99 / max (s) | cached_tokens |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in s.get("replays", []):
            lat = r.get("latency_s") or {}
            print(f"| {r['file'][:-6]} | {r['b']} | {r['stream']} | {r['arrival_scale']} | {r['n']} | {r['wall_s']} | {r['peak_client_inflight']} | {r['cancelled']} | {len(r['errors'])} | {len(r['slow'])} | {lat.get('p50')} / {lat.get('p99')} / {lat.get('max')} | {r['cached_tokens_sum']} |")
        cb = s.get("cbsoak") or {}
        if cb:
            print(f"\ncbsoak: counts {cb.get('counts')}, errors {cb.get('error_kinds')}, http {cb.get('http_status')}")
            print(f"cbsoak TTFT ms {_pct(cb.get('ttft_ms'))}; ITL ms {_pct(cb.get('itl_ms'))}")
        print(f"\nidle invariants: {json.dumps(s.get('idle_invariants'))}")
        print(f"poll: {s['poll']['samples']} samples, max running {s['poll']['max_running']}, max waiting {s['poll']['max_waiting']}, violations {s['poll']['violations']}")
        print(f"fails: {s.get('fails')}\n")


def _pct(h):
    if not h:
        return None
    return {k: round(h[k], 2) for k in ("n", "p50", "p95", "p99", "max") if k in h}


main()
