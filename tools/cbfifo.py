#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""FIFO fairness check of a real batched server from its INTARSIA_CB_TRACE=1 stderr log (M2 gate "no admitted request
waits behind a later one", docs/results/2026-10-16-metal-cb-m2-closeout.md).

The scheduler assigns request ids in submit order (`BatchEngine::submit`, under the failure lock, in channel order) and
traces every state change as `cb-trace t=<ms> <Event> { ... }`. Replaying the trace checks, for every request id:

  1. admission: a request is admitted only when every older request has left the queue (admitted, cancelled while
     queued or rejected at admission): nothing admitted waits behind a later one, with prefix reuse and cancellations;
  2. prefill: a chunk goes to the oldest prefilling request (no chunk while an older request is still prefilling);
  3. first tokens (final chunks) come in submit order;
  4. every request ends exactly once, and nothing is left (unless the log is cut: `--allow-open`).

Same properties as `check_fifo` in crates/intarsia-core/tests/scheduler.rs, on real traffic. Accepts several logs
(one per server run; ids restart at 0 per server) and `.gz`.

    tools/cbfifo.py check LOG [LOG ...] [--json OUT]
    tools/cbfifo.py selftest
"""
import argparse
import gzip
import json
import re
import sys

EV = re.compile(r"cb-trace t=(\d+) (Admitted|PrefillStart|Chunk|Finished|Decode)\b\s*\{(.*)\}\s*$")
KV = re.compile(r"(\w+): (\[[^\]]*\]|[^,\s]+)")


def parse(lines):
    for ln in lines:
        m = EV.search(ln)
        if not m:
            continue
        t, kind, body = int(m.group(1)), m.group(2), m.group(3)
        f = {k: v for k, v in KV.findall(body)}
        yield t, kind, f


QUEUED, PREFILL, DECODE, GONE = "queued", "prefill", "decode", "gone"


def check(events, allow_open=False):
    """events: iterable of (t, kind, fields). Returns (summary dict, violations list)."""
    st = {}      # id -> state; ids appear at Admitted or Finished
    viol = []
    cov = dict(admitted=0, reused=0, contested_chunks=0, cancel_queued=0, cancel_prefill=0,
               cancel_decode=0, done=0, failed=0, rejected=0, chunks=0, first_tokens=0)
    seen_ids = set()
    max_id = -1
    last_first = None
    ev = list(events)
    for _, kind, f in ev:
        if "id" in f:
            max_id = max(max_id, int(f["id"]))
    n = max_id + 1
    state = [QUEUED] * n
    for t, kind, f in ev:
        if kind == "Admitted":
            i = int(f["id"])
            if state[i] != QUEUED:
                viol.append(f"t={t}: request {i} admitted twice")
                continue
            older = next((j for j in range(i) if state[j] == QUEUED), None)
            if older is not None:
                viol.append(f"t={t}: request {i} admitted while older request {older} still waits")
            state[i] = PREFILL
            cov["admitted"] += 1
            cov["reused"] += int(int(f.get("cached", "0")) > 0)
        elif kind == "Chunk":
            i = int(f["id"])
            cov["chunks"] += 1
            if state[i] != PREFILL:
                viol.append(f"t={t}: chunk for request {i} that is {state[i]}")
                continue
            older = next((j for j in range(i) if state[j] == PREFILL), None)
            if older is not None:
                viol.append(f"t={t}: chunk for request {i} while older request {older} is still prefilling")
            if any(state[j] == PREFILL for j in range(i + 1, n)):
                cov["contested_chunks"] += 1
            if f.get("last") == "true":
                if last_first is not None and last_first >= i:
                    viol.append(f"t={t}: first tokens out of order: {last_first} then {i}")
                last_first = i
                state[i] = DECODE
                cov["first_tokens"] += 1
        elif kind == "Finished":
            i = int(f["id"])
            how = f["how"]
            if state[i] == GONE:
                viol.append(f"t={t}: request {i} finished twice")
                continue
            if how == "Cancelled":
                cov["cancel_" + {QUEUED: "queued", PREFILL: "prefill", DECODE: "decode"}[state[i]]] += 1
            elif how == "Done":
                cov["done"] += 1
            elif how == "Rejected":
                cov["rejected"] += 1
            else:
                cov["failed"] += 1
            state[i] = GONE
    open_ids = [i for i, s in enumerate(state) if s != GONE]
    if open_ids and not allow_open:
        viol.append(f"requests never finished: {open_ids[:10]}{' ...' if len(open_ids) > 10 else ''}")
    # ids that never appear at all (submitted, no event) are open (QUEUED) above
    cov["requests"] = n
    return cov, viol


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as fh:
        return list(parse(fh))


def cmd_check(a):
    out, bad = [], 0
    for p in a.logs:
        cov, viol = check(read(p), allow_open=a.allow_open)
        print(f"{p}: {cov['requests']} requests, admitted {cov['admitted']} (reused {cov['reused']}), done {cov['done']}, "
              f"cancelled q/p/d {cov['cancel_queued']}/{cov['cancel_prefill']}/{cov['cancel_decode']}, failed {cov['failed']}, "
              f"rejected {cov['rejected']}, chunks {cov['chunks']} (contested {cov['contested_chunks']}): "
              f"{'OK' if not viol else f'{len(viol)} VIOLATIONS'}")
        for v in viol[:20]:
            print("  ", v)
        bad += len(viol)
        out.append({"log": p, "coverage": cov, "violations": viol})
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 1 if bad else 0


def _tr(*lines):
    return list(parse(f"cb-trace t={i} {l}" for i, l in enumerate(lines)))


def cmd_selftest(_):
    A = lambda i, s=0, c=0: f"Admitted {{ iter: 1, id: {i}, slot: {s}, cached: {c} }}"
    C = lambda i, last: f"Chunk {{ iter: 1, id: {i}, start: 0, len: 10, last: {str(last).lower()} }}"
    F = lambda i, how: f"Finished {{ iter: 2, id: {i}, how: {how} }}"
    ok = _tr(F(0, "Cancelled"), A(1), A(2, 1, 5), C(1, True), C(2, False), C(2, True), F(1, "Done"), F(2, "Done"))
    cov, v = check(ok)
    assert not v and cov["cancel_queued"] == 1 and cov["reused"] == 1, (cov, v)
    cases = {
        "admission overtake": _tr(A(1), A(0), F(0, "Done"), F(1, "Done")),
        "prefill overtake": _tr(A(0), A(1, 1), C(1, False), C(0, True), C(1, True), F(0, "Done"), F(1, "Done")),
        "chunk after final": _tr(A(0), C(0, True), C(0, False), F(0, "Done")),
        "never finished": _tr(A(0), C(0, True)),
        "finished twice": _tr(A(0), C(0, True), F(0, "Done"), F(0, "Done")),
    }
    for name, tr in cases.items():
        _, v = check(tr)
        assert v, f"{name} not caught"
    cov, v = check(_tr(A(0), C(0, True)), allow_open=True)
    assert not v
    print("selftest: ok (1 pass case, %d violation cases caught)" % len(cases))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check")
    c.add_argument("logs", nargs="+")
    c.add_argument("--json")
    c.add_argument("--allow-open", action="store_true", help="the log is cut: requests still running are fine")
    c.set_defaults(fn=cmd_check)
    s = sp.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
