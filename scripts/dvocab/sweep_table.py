#!/usr/bin/env python3
"""Lane dvocab: tables from speed-<visit>-<set>.json (sweep.sh). sweep_table.py <raw dir> [baseline arm, default A]
Visit names are <ARM>.<tag> (A.a, P65536.b, ...). Prints (1) the full table per set and stream count, (2) a pivot of tok/s vs the baseline."""
import glob, json, os, statistics, sys
from collections import defaultdict

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "A"
cells = defaultdict(list)  # (set, streams, arm) -> [row...]
for f in sorted(glob.glob(f"{d}/speed-*-*.json")):
    r = json.load(open(f))
    arm = r["arm"].split(".")[0]
    for row in r["rows"]:
        row["_visit"] = r["arm"]
        cells[(r["set"] + ("-nt" if r.get("thinking") == "off" else ""), row["streams"], arm)].append(row)
sets = [s for s in ("en", "code", "cjk", "mixed", "long", "en-nt", "code-nt", "cjk-nt", "mixed-nt") if any(k[0] == s for k in cells)]
streams = sorted({k[1] for k in cells})
arms = sorted({k[2] for k in cells}, key=lambda a: (a != base, int(a[1:]) if a[1:].isdigit() else 0))
order = lambda a: (a != base, int(a[1:]) if a[1:].isdigit() else 0)
arms = sorted(arms, key=order)


def agg(rows):
    c = [x["decode_tok_s_combined"]["median"] for x in rows]
    prop = sum(x["mtp"]["drafted"] for x in rows); acc = sum(x["mtp"]["accepted"] for x in rows)
    return dict(c=c, mean=statistics.mean(c), itl=statistics.mean(x["itl_ms_p99"] for x in rows), gap=statistics.mean(x["gap_ms_p99"] for x in rows),
                acc=(acc / prop if prop else None), cjk=statistics.mean(x["cjk_share"] for x in rows) if "cjk_share" in rows[0] else None,
                tpp=statistics.mean(x["mtp"]["tokens_per_pass"] for x in rows if x["mtp"]["tokens_per_pass"]) if any(x["mtp"]["tokens_per_pass"] for x in rows) else None)


for s in sets:
    for n in streams:
        if not any((s, n, a) in cells for a in arms):
            continue
        b = agg(cells[(s, n, base)]) if (s, n, base) in cells else None
        bsha = [e["sha256"] for e in cells[(s, n, base)][0]["each"]] if b else None
        print(f"\n### set {s}, {n} stream(s)\n")
        print(f"| arm | visits | combined tok/s (each visit) | mean | vs {base} | ITL p99 ms | chunk gap p99 ms | acceptance | CJK share | replies identical to {base} |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            if (s, n, a) not in cells:
                continue
            rr = cells[(s, n, a)]; g = agg(rr)
            same = "-"
            if bsha:
                same = f"{sum(1 for x in rr for i, e in enumerate(x['each']) if e and e['sha256'] == bsha[i])} / {sum(len(x['each']) for x in rr)}"
            vis = " / ".join("%.1f" % v for v in g["c"])
            acc = "-" if g["acc"] is None else "%.3f" % g["acc"]
            cjk = "-" if g["cjk"] is None else "%.2f" % g["cjk"]
            rel = g["mean"] / b["mean"] if b else float("nan")
            print(f"| {a} | {len(rr)} | {vis} | {g['mean']:.1f} | {rel:.3f} | {g['itl']:.0f} | {g['gap']:.0f} | {acc} | {cjk} | {same} |")

print(f"\n### pivot: combined tok/s mean (vs {base})\n")
print("| set | streams | " + " | ".join(arms) + " |")
print("|---|---|" + "---|" * len(arms))
for s in sets:
    for n in streams:
        if not any((s, n, a) in cells for a in arms):
            continue
        b = agg(cells[(s, n, base)])["mean"] if (s, n, base) in cells else None
        out = []
        for a in arms:
            if (s, n, a) in cells:
                m = agg(cells[(s, n, a)])["mean"]
                out.append(f"{m:.1f} ({m / b:.3f})" if b else f"{m:.1f}")
            else:
                out.append("-")
        print(f"| {s} | {n} | " + " | ".join(out) + " |")
