#!/usr/bin/env python3
"""Lane mtp2s: A/B tables from speed-<visit>.json (ab.sh). ab_table2.py <raw dir> [baseline arm, default A]
Visit names are the arm plus a trailing digit (A1, A2, VD1, ...) or, for the k = 1 arm, VD1a / VD1b."""
import glob, json, os, re, statistics, sys

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "A"
res = {os.path.basename(f)[6:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{d}/speed-*.json"))}


def arm_of(v):
    return "VD1(k1)" if re.fullmatch(r"VD1[a-z]", v) else re.sub(r"\d+$", "", v)


print("| visit | streams | decode tok/s per stream, median (min-max) | combined tok/s | ITL p99 ms | chunk gap p99 ms (max) | MTP passes | accepted / drafted | tokens per pass |")
print("|---|---|---|---|---|---|---|---|---|")
for name, r in res.items():
    for row in r["rows"]:
        m = row["mtp"]; ps = row["decode_tok_s_per_stream"]
        print(f"| {name} | {row['streams']} | {ps['median']} ({ps['min']}-{ps['max']}) | {row['decode_tok_s_combined']['median']} | {row['itl_ms_p99']} | "
              f"{row['gap_ms_p99']} ({row['gap_ms_max']}) | {m['verify_passes']:.0f} | {m['accepted']:.0f} / {m['drafted']:.0f}"
              + (f" ({m['accept_rate']})" if m['accept_rate'] is not None else "") + f" | {m['tokens_per_pass']} |")

arms = {}
for name, r in res.items():
    arms.setdefault(arm_of(name), []).append(r)
streams = sorted({row["streams"] for r in res.values() for row in r["rows"]})
print()
print(f"| arm | visits | streams | combined tok/s (each visit) | mean | vs {base} | per stream median (mean) | ITL p99 ms (mean) | acceptance | replies identical to {base} |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in streams:
    def rows(a):
        return [row for r in arms.get(a, []) for row in r["rows"] if row["streams"] == s]
    b = rows(base)
    bm = statistics.mean(x["decode_tok_s_combined"]["median"] for x in b) if b else None
    bsha = [e and e["sha256"] for e in b[0]["each"]] if b else None
    for a in arms:
        rr = rows(a)
        if not rr:
            continue
        c = [x["decode_tok_s_combined"]["median"] for x in rr]
        m = statistics.mean(c)
        same = "-"
        if bsha:
            same = f"{sum(1 for x in rr for i, e in enumerate(x['each']) if e and e['sha256'] == bsha[i])} / {sum(len(x['each']) for x in rr)}"
        acc = rr[0]["mtp"]["accept_rate"]
        print(f"| {a} | {len(rr)} | {s} | {' / '.join(f'{v:.1f}' for v in c)} | {m:.1f} | {(m / bm if bm else float('nan')):.3f} | "
              f"{statistics.mean(x['decode_tok_s_per_stream']['median'] for x in rr):.1f} | {statistics.mean(x['itl_ms_p99'] for x in rr):.1f} | {acc} | {same} |")
