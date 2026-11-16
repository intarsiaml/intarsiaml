#!/usr/bin/env python3
"""Lane dsubset: the ABBA table from speed-<visit>-<set>-nt.json (runs.sh speed). speed_table.py <speed dir>
Arms P (prefix 131072) and S (subset 65536), visits P.1 S.1 S.2 P.2. Per set and stream count: per-stream decode tok/s
(median over requests, both visits), combined tok/s, ITL p99 (localeval's, per visit), tokens per verify pass, S / P, and
whether the replies are identical across all four visits (seeded T = 1.0 on the batched engine: they must be)."""
import glob, json, os, statistics, sys

d = sys.argv[1]
res = {}
for f in glob.glob(f"{d}/speed-*-nt.json"):
    _, visit, set_, _ = os.path.basename(f)[:-5].split("-")
    res[(visit, set_)] = json.load(open(f))
sets = [s for s in ("mixed", "en", "code", "zh") if all((v, s) in res for v in ("P.1", "S.1", "S.2", "P.2"))]
m = statistics.mean
print("| set | streams | P per stream (P.1 / P.2) | S per stream (S.1 / S.2) | S / P per stream | P combined | S combined | S / P combined | ITL p99 ms P (P.1 / P.2) | ITL p99 ms S (S.1 / S.2) | tokens/pass P / S | replies identical |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = {}
for s in sets:
    for i, row in enumerate(res[("P.1", s)]["rows"]):
        g = lambda v, k: res[(v, s)]["rows"][i][k]
        ps = [g(v, "decode_tok_s_per_stream")["median"] for v in ("P.1", "P.2")]
        ss = [g(v, "decode_tok_s_per_stream")["median"] for v in ("S.1", "S.2")]
        pc = [g(v, "decode_tok_s_combined")["median"] for v in ("P.1", "P.2")]
        sc = [g(v, "decode_tok_s_combined")["median"] for v in ("S.1", "S.2")]
        pi = [g(v, "itl_ms_p99") for v in ("P.1", "P.2")]
        si = [g(v, "itl_ms_p99") for v in ("S.1", "S.2")]
        tp = (g("P.1", "mtp")["tokens_per_pass"], g("S.1", "mtp")["tokens_per_pass"])
        sha = lambda v: [e and e.get("sha256") for e in g(v, "each")]
        same = sum(1 for x in zip(*(sha(v) for v in ("P.1", "S.1", "S.2", "P.2"))) if len(set(x)) == 1)
        n = len(sha("P.1"))
        tot.setdefault(row["streams"], []).append(m(ss) / m(ps))
        print(f"| {s} | {row['streams']} | {ps[0]} / {ps[1]} | {ss[0]} / {ss[1]} | {m(ss) / m(ps):.3f} | {pc[0]} / {pc[1]} | {sc[0]} / {sc[1]} | {m(sc) / m(pc):.3f} | {pi[0]} / {pi[1]} | {si[0]} / {si[1]} | {tp[0]} / {tp[1]} | {same} of {n} |")
for k, v in sorted(tot.items()):
    print(f"\narithmetic mean of S / P per stream over sets at {k} stream(s): {m(v):.3f}")
