#!/usr/bin/env python3
"""Analyse a server log written with INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1 (gate B of
docs/results/2026-10-16-metal-cb-gates-reuse-spec.md): is speculative drafting off while >= 4 sequences decode, and does
it come back once fewer than 4 decode?

Log lines used, in server order (stderr is one ordered stream):
  cb-trace t=<epoch ms> DecodePhase { iter, b, spec_allowed }   one per scheduler iteration with decoding sequences,
                                                                printed BEFORE the sequences are prepared
  spec: token N pos P drafted D accepted A emits E             one per verify pass, printed while preparing
Every `spec:` line after a DecodePhase marker and before the next cb-trace line belongs to that iteration.

  python3 tools/cbspectrace.py SERVER.log [--window NAME START_MS END_MS]... [--json OUT]
"""
import argparse
import collections
import json
import re
import sys

PH = re.compile(r"^cb-trace t=(\d+) DecodePhase \{ iter: (\d+), b: (\d+), spec_allowed: (true|false) \}")
CB = re.compile(r"^cb-trace t=(\d+) ")
SP = re.compile(r"^spec: token (\d+) pos (\d+) drafted (\d+) accepted (\d+)")


def parse(path):
    """-> list of phases {t, iter, b, allowed, passes, drafted, accepted}"""
    phases, cur = [], None
    with open(path, errors="replace") as f:
        for ln in f:
            m = PH.match(ln)
            if m:
                cur = {"t": int(m[1]), "iter": int(m[2]), "b": int(m[3]), "allowed": m[4] == "true", "passes": 0, "drafted": 0, "accepted": 0}
                phases.append(cur)
                continue
            if CB.match(ln):
                cur = None
                continue
            m = SP.match(ln)
            if m and cur is not None:
                d = int(m[3])
                if d > 0:
                    cur["passes"] += 1
                    cur["drafted"] += d
                    cur["accepted"] += int(m[4])
            elif m and cur is None:
                phases.append({"t": None, "iter": None, "b": None, "allowed": None, "passes": 1, "drafted": int(m[3]), "accepted": int(m[4]), "orphan": True})
    return phases


def analyse(phases):
    real = [p for p in phases if not p.get("orphan")]
    orphans = [p for p in phases if p.get("orphan")]
    by_b = collections.OrderedDict()
    for p in sorted(real, key=lambda p: p["b"]):
        d = by_b.setdefault(p["b"], {"phases": 0, "allowed_phases": 0, "verify_passes": 0, "drafted": 0, "accepted": 0})
        d["phases"] += 1
        d["allowed_phases"] += p["allowed"]
        d["verify_passes"] += p["passes"]
        d["drafted"] += p["drafted"]
        d["accepted"] += p["accepted"]
    off = [p for p in real if not p["allowed"]]
    violations = [p for p in off if p["passes"]]
    gate_wrong = [p for p in real if p["allowed"] != (p["b"] < 4)]
    # transitions: runs of consecutive phases with the same spec_allowed
    runs = []
    for p in real:
        if runs and runs[-1]["allowed"] == p["allowed"]:
            r = runs[-1]
        else:
            r = {"allowed": p["allowed"], "phases": 0, "passes": 0, "drafted": 0, "accepted": 0, "first_iter": p["iter"]}
            runs.append(r)
        r["phases"] += 1
        r["passes"] += p["passes"]
        r["drafted"] += p["drafted"]
        r["accepted"] += p["accepted"]
    off_to_on = [(i, r) for i, r in enumerate(runs) if i > 0 and r["allowed"] and not runs[i - 1]["allowed"]]
    on_to_off = [i for i, r in enumerate(runs) if i > 0 and not r["allowed"] and runs[i - 1]["allowed"]]
    resumed = [r for _, r in off_to_on if r["passes"] > 0]
    return {
        "decode_phases": len(real), "orphan_spec_lines": len(orphans), "by_b": by_b,
        "phases_spec_off": len(off), "spec_off_phases_with_verify_pass": len(violations),
        "phases_where_gate_value_is_not_(b<4)": len(gate_wrong),
        "runs": len(runs), "on_to_off_transitions": len(on_to_off), "off_to_on_transitions": len(off_to_on),
        "off_to_on_followed_by_drafting": len(resumed),
        "drafted_total": sum(p["drafted"] for p in real), "accepted_total": sum(p["accepted"] for p in real),
        "verify_passes_total": sum(p["passes"] for p in real),
        "drafted_while_allowed": sum(p["drafted"] for p in real if p["allowed"]),
        "drafted_while_off": sum(p["drafted"] for p in off),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", nargs=3, action="append", metavar=("NAME", "START_MS", "END_MS"))
    ap.add_argument("--json")
    a = ap.parse_args()
    phases = parse(a.log)
    out = {"all": analyse(phases)}
    for name, s, e in a.window or []:
        s, e = int(s), int(e)
        out[name] = analyse([p for p in phases if p.get("t") is not None and s <= p["t"] <= e])
    for k, v in out.items():
        print(f"== {k}: {v['decode_phases']} decode phases, spec-off phases {v['phases_spec_off']} (with a verify pass: "
              f"{v['spec_off_phases_with_verify_pass']}), gate value != (b<4): {v['phases_where_gate_value_is_not_(b<4)']}, "
              f"on->off {v['on_to_off_transitions']}, off->on {v['off_to_on_transitions']} (drafting resumed after {v['off_to_on_followed_by_drafting']}), "
              f"drafted {v['drafted_total']} accepted {v['accepted_total']} in {v['verify_passes_total']} passes")
        print("   b: " + ", ".join(f"{b}: {d['phases']}ph/{d['verify_passes']}vp/{d['drafted']}dr/{d['accepted']}ac" for b, d in v["by_b"].items()))
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
