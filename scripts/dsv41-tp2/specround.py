#!/usr/bin/env python3
"""Lane dsv41/graphs: what another draft count or confidence floor would have done, from the engine's round trace.

Usage: specround.py SERVE.log [--verify "1:31.3,2:38.4,3:45.5,4:52.9"] [--draft 6.6]

Reads `dsv41: spec round k=K conf=c1,c2,.. ok=N` lines (INTARSIA_DSV41_SPEC_TRACE=1). For each rule it replays the rounds:
verify only the leading drafts the rule keeps (at least one), so a round yields min(ok, kept) accepted drafts + 1 token and
costs a verify pass of kept + 1 rows + the draft block. Estimated ms per token = total time / total tokens. An estimate only:
it assumes a rejected or unverified draft's round would have gone on from the same text.
"""
import re
import statistics as st
import sys

PAT = re.compile(r"spec round k=(\d+) conf=([-\d.,einf]+) ok=(\d+)")


def main():
    a = sys.argv[1:]
    verify = {1: 31.3, 2: 38.4, 3: 45.5, 4: 52.9}
    if "--verify" in a:
        verify = {int(p.split(":")[0]): float(p.split(":")[1]) for p in a[a.index("--verify") + 1].split(",")}
    draft = float(a[a.index("--draft") + 1]) if "--draft" in a else 6.6
    rounds = []
    for l in open(a[0], errors="replace"):
        m = PAT.search(l)
        if m:
            rounds.append(([float(x) for x in m[2].split(",")], int(m[3])))
    if not rounds:
        sys.exit("no round lines: run with INTARSIA_DSV41_SPEC_TRACE=1")
    K = max(len(c) for c, _ in rounds)
    print(f"{len(rounds)} rounds, k up to {K}")
    reached, ok = [0] * K, [0] * K
    buckets = [[] for _ in range(K)]
    for c, n in rounds:
        for i in range(min(len(c), n + 1)):
            reached[i] += 1
            ok[i] += i < n
            buckets[i].append((c[i], i < n))
    for i in range(K):
        print(f"position {i + 1}: {ok[i]}/{reached[i]} accepted ({100 * ok[i] / max(reached[i], 1):.0f} %, given the earlier ones were)")
    vals = sorted(v for c, _ in rounds for v in c)
    qs = [vals[int(len(vals) * q)] for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    print("confidence deciles:", " ".join(f"{q:.2f}" for q in qs))
    print(f"{'rule':24s} {'tok/round':>9s} {'rows':>5s} {'ms/round':>9s} {'ms/token':>9s} {'tok/s':>6s}")

    def show(name, keep):
        tok = t = rows = 0
        for c, n in rounds:
            kept = keep(c)
            tok += min(n, kept) + 1
            rows += kept + 1
            t += verify[kept + 1] + draft
        print(f"{name:24s} {tok / len(rounds):9.2f} {rows / len(rounds):5.2f} {t / len(rounds):9.1f} {t / tok:9.2f} {1000 * tok / t:6.1f}")

    for k in range(1, K + 1):
        show(f"k={k}", lambda c, k=k: min(k, len(c)))
    for f in sorted({round(q, 1) for q in qs}):
        for k in (2, K) if K > 2 else (K,):
            def keep(c, f=f, k=k):
                n = 0
                for v in c[:k]:
                    if v < f:
                        break
                    n += 1
                return max(n, 1)
            show(f"k={k} floor {f:+.1f}", keep)


if __name__ == "__main__":
    main()
