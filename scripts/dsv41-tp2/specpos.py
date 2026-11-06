#!/usr/bin/env python3
"""Lane dsv41/sampled-spec: tokens a round and acceptance per draft position from the engine's per-request speculation lines.

    specpos.py LINES.txt LABEL

LINES.txt holds `dsv41: spec N tokens in R rounds (..), A/D drafts accepted (..), by position a/b c/d ..` lines (one per request).
`a/b` at position i: drafts accepted of drafts checked there (a position is checked only when the ones before it were accepted, so
a/b is the conditional acceptance). Prints one line: requests, tokens, rounds, tokens a round, conditional acceptance per position.
"""
import re
import sys

pat = re.compile(r"spec (\d+) tokens in (\d+) rounds .*?, (\d+)/(\d+) drafts accepted .*?by position ((?:\d+/\d+ ?)*)")


def read(path):
    reqs = toks = rounds = 0
    ok = [0] * 8
    seen = [0] * 8
    for l in open(path):
        m = pat.search(l)
        if not m:
            continue
        reqs += 1
        toks += int(m[1])
        rounds += int(m[2])
        for i, c in enumerate(m[5].split()):
            a, b = c.split("/")
            ok[i] += int(a)
            seen[i] += int(b)
    return reqs, toks, rounds, ok, seen


def show(label, path):
    reqs, toks, rounds, ok, seen = read(path)
    pos = " ".join(f"{100 * ok[i] / seen[i]:.1f}% ({ok[i]}/{seen[i]})" for i in range(8) if seen[i])
    print(f"specpos {label}: {reqs} requests, {toks} tokens in {rounds} rounds = {toks / max(rounds, 1):.3f} a round; accepted by position: {pos}")


if sys.argv[1] == "ratio":
    # ratio SAMPLED.txt GREEDY.txt: retention = the sampled conditional acceptance over the greedy one, per position, and weighted by the
    # sampled drafts checked there; and from tokens a round: (tpr_s - 1) / (tpr_g - 1)
    rs, ts, ns, oks, seens = read(sys.argv[2])
    rg, tg, ng, okg, seeng = read(sys.argv[3])
    w = [(seens[i], (oks[i] / seens[i]) / (okg[i] / seeng[i])) for i in range(8) if seens[i] and seeng[i] and okg[i]]
    per = " ".join(f"{r:.2f}" for _, r in w)
    r_pos = sum(n * r for n, r in w) / max(sum(n for n, _ in w), 1)
    r_tok = (ts / max(ns, 1) - 1) / max(tg / max(ng, 1) - 1, 1e-9)
    print(f"retention by position {per}; weighted {r_pos:.3f}; from tokens a round {r_tok:.3f}")
else:
    show(sys.argv[2], sys.argv[1])
