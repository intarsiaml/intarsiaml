#!/usr/bin/env python3
"""Lane mtp-sampled: the A/B table from speed-*.json (ab.sh). ab_table.py <raw dir>"""
import glob, json, os, statistics, sys

d = sys.argv[1]
res = {os.path.basename(f)[6:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{d}/speed-*.json"))}
print("| visit | streams | decode tok/s per stream, median (min-max) | combined tok/s | ITL p99 ms (localeval) | chunk gap p99 ms (max) | MTP passes | accepted / drafted | tokens per pass |")
print("|---|---|---|---|---|---|---|---|---|")
for name, r in res.items():
    for row in r["rows"]:
        m = row["mtp"]
        ps = row["decode_tok_s_per_stream"]
        print(f"| {name} | {row['streams']} | {ps['median']} ({ps['min']}-{ps['max']}) | {row['decode_tok_s_combined']['median']} | {row['itl_ms_p99']} | {row['gap_ms_p99']} ({row['gap_ms_max']}) | "
              f"{m['verify_passes']:.0f} | {m['accepted']:.0f} / {m['drafted']:.0f}" + (f" ({m['accept_rate']})" if m['accept_rate'] is not None else "") + f" | {m['tokens_per_pass']} |")
print()
print("| --max-batch | streams | A per stream (two visits) | B per stream (two visits) | B / A per stream | A combined | B combined | B / A combined | replies identical A vs B |")
print("|---|---|---|---|---|---|---|---|---|")
for mb in ("mb2", "mb4"):
    vis = {k: res.get(f"{mb}-{k}") for k in ("A1", "B1", "B2", "A2")}
    if not all(vis.values()):
        continue
    for i, row in enumerate(vis["A1"]["rows"]):
        g = lambda k, f: vis[k]["rows"][i][f]["median"]
        a = [g("A1", "decode_tok_s_per_stream"), g("A2", "decode_tok_s_per_stream")]
        b = [g("B1", "decode_tok_s_per_stream"), g("B2", "decode_tok_s_per_stream")]
        ac = [g("A1", "decode_tok_s_combined"), g("A2", "decode_tok_s_combined")]
        bc = [g("B1", "decode_tok_s_combined"), g("B2", "decode_tok_s_combined")]
        sha = lambda k: [e and e["sha256"] for e in vis[k]["rows"][i]["each"]]
        same = sum(1 for x in zip(sha("A1"), sha("B1"), sha("B2"), sha("A2")) if len(set(x)) == 1)
        m = statistics.mean
        print(f"| {mb[2:]} | {row['streams']} | {a[0]} / {a[1]} | {b[0]} / {b[1]} | {m(b) / m(a):.3f} | {ac[0]} / {ac[1]} | {bc[0]} / {bc[1]} | {m(bc) / m(ac):.3f} | {same} of {len(sha('A1'))} |")
