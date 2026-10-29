#!/usr/bin/env python3
"""Lane msd: aggregate the localeval runs of a window dir (raw/<name>/msd-<arm>-r<round>.out) into the concurrency table.

    agg.py raw/gateA [raw/other ...] [--base base] [--json out.json]
Per arm and stream count: aggregate output tok/s of every round, their median, TTFT median (median of the rounds' medians) and p95
(nearest rank over a burst of n <= 16 streams is the maximum; the median of the rounds' maxima), per-stream decode tok/s median.
With --base, the change of every other arm against it (median of the per-round ratios and the ratio of medians).
"""
import glob, json, os, re, statistics as st, sys

args = sys.argv[1:]
base = None; out = None; dirs = []
while args:
    a = args.pop(0)
    if a == "--base": base = args.pop(0)
    elif a == "--json": out = args.pop(0)
    else: dirs.append(a)
runs = {}  # arm -> round -> {streams: row}
for d in dirs:
    for f in sorted(glob.glob(f"{d}/msd-*-r*.out")):
        m = re.match(r"msd-(.+)-r(\d+)\.out$", os.path.basename(f))
        arm, rnd = m.group(1), int(m.group(2))
        txt = open(f, errors="replace").read()
        p = re.search(r"result file: `([^`]+)`", txt)
        if not p: continue
        res = json.load(open("/opt/code/deepeval/" + p.group(1)))
        rows = {c["streams"]: c for c in res["speed"]["concurrency"]} if "speed" in res and "concurrency" in res["speed"] else {c["streams"]: c for c in res.get("concurrency", [])}
        sp = res.get("speed", {}).get("rows") or res.get("rows") or []
        runs.setdefault(arm, {})[rnd] = {"conc": rows, "speed": sp}
table = {}
for arm, rs in runs.items():
    streams = sorted({s for r in rs.values() for s in r["conc"]})
    for s in streams:
        c = [rs[r]["conc"][s] for r in sorted(rs) if s in rs[r]["conc"]]
        table.setdefault(arm, {})[s] = {
            "agg_rounds": [x["aggregate_decode_tok_s"] for x in c],
            "agg_median": st.median(x["aggregate_decode_tok_s"] for x in c),
            "ttft_p50": st.median(x["ttft_s"]["median"] for x in c),
            "ttft_p95": st.median(x["ttft_s"]["max"] for x in c),
            "ttft_max_rounds": [x["ttft_s"]["max"] for x in c],
            "per_stream_median": st.median(x["per_stream_decode_tok_s"]["median"] for x in c),
        }
    sp = {}
    for r in rs.values():
        for row in r["speed"]:
            sp.setdefault(row["target_prompt_tokens"], []).append(row)
    for size, rows in sorted(sp.items()):
        table[arm].setdefault("single", {})[size] = {
            "decode_median": st.median(x["decode_tok_s"]["median"] for x in rows),
            "decode_rounds": [x["decode_tok_s"]["median"] for x in rows],
            "prefill_median": st.median(x["prefill_tok_s"]["median"] for x in rows),
            "prefill_rounds": [x["prefill_tok_s"]["median"] for x in rows],
            "ttft_median": st.median(x["ttft_s"]["median"] for x in rows),
        }
for arm, t in table.items():
    print(f"== {arm}")
    for s in sorted(k for k in t if k != "single"):
        c = t[s]
        print(f"  {s:>2} streams: agg {c['agg_median']:7.1f} tok/s {c['agg_rounds']}  TTFT p50 {c['ttft_p50']:.2f} s p95 {c['ttft_p95']:.2f} s  per-stream {c['per_stream_median']:.1f}")
    for size, c in t.get("single", {}).items():
        print(f"  single {size:>5}: decode {c['decode_median']:.1f} {c['decode_rounds']}  prefill {c['prefill_median']:.0f} {c['prefill_rounds']}  TTFT {c['ttft_median']:.2f} s")
if base and base in table:
    for arm in table:
        if arm == base: continue
        print(f"== {arm} vs {base}")
        for s in sorted(k for k in table[arm] if k != "single" and k in table[base]):
            a, b = table[arm][s]["agg_rounds"], table[base][s]["agg_rounds"]
            ratios = [x / y for x, y in zip(a, b)]
            print(f"  {s:>2} streams: median of ratios {100 * (st.median(ratios) - 1):+.1f} %  ratio of medians {100 * (table[arm][s]['agg_median'] / table[base][s]['agg_median'] - 1):+.1f} %  rounds {[round(100 * (r - 1), 1) for r in ratios]}")
        for size in sorted(table[arm].get("single", {})):
            if size in table[base].get("single", {}):
                for key in ("decode", "prefill"):
                    a, b = table[arm]["single"][size][key + "_rounds"], table[base]["single"][size][key + "_rounds"]
                    print(f"  single {size:>5} {key}: median ratio {st.median(x / y for x, y in zip(a, b)):.3f}  ratio of medians {table[arm]['single'][size][key + '_median'] / table[base]['single'][size][key + '_median']:.3f}")
if out: json.dump(table, open(out, "w"), indent=1)
