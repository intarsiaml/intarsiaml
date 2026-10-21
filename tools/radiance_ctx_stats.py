#!/usr/bin/env python3
"""Context-length statistics for the radiance service on tatooine, from its journal.

The native radiance engine logs one line per finished request:

    ... radiance-native.sh[PID]: I req chatcmpl-X done from ::ffff:IP:PORT finish=stop prompt=17 (cache 0) gen=16 ...

`prompt` is the full prompt length (cached prefix included), so prompt+gen is the KV length at the end
of decode. Every engine start is followed by a fixed 19-request smoke test (radiance-health); those are
split out because they are not traffic.

Usage (read-only; no GPU load, nothing sent to the service):

    ssh tatooine 'journalctl --user -u radiance --no-pager -o short-iso' > journal.txt
    tools/radiance_ctx_stats.py journal.txt
    tools/radiance_ctx_stats.py journal.txt --markdown     # tables for docs/results

Decode share at threshold T = fraction of generated tokens produced while the KV length was >= T
(token-weighted; with speculative decoding one step emits several tokens, so this is a token share,
not an exact step share).
"""
import argparse
import collections
import re
import sys

DONE = re.compile(
    r"^(\S+) \S+ radiance-native\.sh\[(\d+)\]: I req (\S+) done from (\S+) "
    r"finish=(\S+) prompt=(\d+) \(cache (\d+)\) gen=(\d+)"
)
SMOKE_HEAD = (17, 42, 42, 42, 53, 53)
SMOKE_TAIL = (18, 21, 21, 21)  # requests 16..19 of the smoke test
THRESHOLDS = (2048, 4096, 8192, 12288, 16384)


def ip_of(addr):
    return re.sub(r":\d+$", "", addr).replace("::ffff:", "")


def load(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = DONE.match(line)
            if m:
                rows.append(
                    dict(ts=m[1], pid=m[2], id=m[3], ip=ip_of(m[4]), fin=m[5],
                         prompt=int(m[6]), cache=int(m[7]), gen=int(m[8]))
                )
    by_pid = collections.defaultdict(list)
    for r in rows:
        r["smoke"] = False
        by_pid[r["pid"]].append(r)
    starts = 0
    for v in by_pid.values():
        if (len(v) >= 19 and tuple(x["prompt"] for x in v[:6]) == SMOKE_HEAD
                and tuple(x["prompt"] for x in v[15:19]) == SMOKE_TAIL):
            starts += 1
            for x in v[:19]:
                x["smoke"] = True
    return rows, starts, len(by_pid)


def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))]


def decode_share(rs, t):
    """(generated tokens at KV length >= t, all generated tokens)"""
    tot = sum(r["gen"] for r in rs)
    ge = sum(max(0, r["prompt"] + r["gen"] - max(t, r["prompt"])) for r in rs)
    return ge, tot


def summarize(name, rs, md):
    if not rs:
        print(f"{name}: no requests")
        return
    P = [r["prompt"] for r in rs]
    G = [r["gen"] for r in rs]
    T = [r["prompt"] + r["gen"] for r in rs]
    first, last = min(r["ts"] for r in rs)[:16], max(r["ts"] for r in rs)[:16]
    dec = [100 * decode_share(rs, t)[0] / max(1, decode_share(rs, t)[1]) for t in THRESHOLDS]
    req = [100 * sum(1 for x in T if x >= t) / len(T) for t in THRESHOLDS]
    if md:
        print(f"| {name} | {len(rs)} | {q(P,.5)} / {q(P,.9)} / {max(P)} | {q(G,.5)} / {q(G,.9)} / {max(G)} "
              f"| {q(T,.5)} / {q(T,.9)} / {max(T)} | {sum(G)} | "
              + " / ".join(f"{d:.1f}" for d in dec[1:]) + " | "
              + " / ".join(f"{d:.1f}" for d in req[1:]) + f" | {first} .. {last} |")
    else:
        print(f"{name}: n={len(rs)} {first}..{last}")
        print(f"  prompt p50/p90/max {q(P,.5)}/{q(P,.9)}/{max(P)}  gen {q(G,.5)}/{q(G,.9)}/{max(G)}  "
              f"total {q(T,.5)}/{q(T,.9)}/{max(T)}  gen tokens {sum(G)}")
        print("  decode-token share >= " + " / ".join(f"{t//1024}k" for t in THRESHOLDS) + ": "
              + " / ".join(f"{d:.1f}%" for d in dec))
        print("  request final ctx >= " + " / ".join(f"{t//1024}k" for t in THRESHOLDS) + ": "
              + " / ".join(f"{d:.1f}%" for d in req))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--clients", default="", help="comma list of ip=label to name clients")
    a = ap.parse_args()
    names = dict(kv.split("=", 1) for kv in a.clients.split(",") if "=" in kv)
    rows, smoke_starts, pids = load(a.journal)
    if not rows:
        sys.exit("no 'req ... done' lines found (native engine era only; vLLM-era logs carry no token counts)")
    print(f"done-lines={len(rows)} engine-pids={pids} smoke-tested-starts={smoke_starts} "
          f"smoke-requests={sum(r['smoke'] for r in rows)}")
    if a.markdown:
        print("| slice | requests | prompt p50/p90/max | gen p50/p90/max | total p50/p90/max | gen tokens "
              "| decode share >=4k/8k/12k/16k (%) | requests >=4k/8k/12k/16k (%) | span |")
        print("|---|---|---|---|---|---|---|---|---|")
    summarize("all (incl. smoke)", rows, a.markdown)
    real = [r for r in rows if not r["smoke"]]
    summarize("excluding smoke", real, a.markdown)
    by = collections.defaultdict(list)
    for r in real:
        by[r["ip"]].append(r)
    for ip, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        summarize(f"{ip} {names.get(ip, '')}".strip(), rs, a.markdown)


if __name__ == "__main__":
    main()
