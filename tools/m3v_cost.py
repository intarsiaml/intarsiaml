#!/usr/bin/env python3
"""M3 stage 0: reduce the cost slot (m3_verify_cost samples, g0b_bench attn samples, macmon clock) to one table.

  m3v_cost.py DIR [--min-mhz 1400] [--json OUT.json] [--md OUT.md]

DIR holds cost.csv, g0b-attn.csv and macmon.jsonl.gz (tools/m3v_cost_slot.sh). A sample is kept when the GPU clock of the macmon
sample covering it (the first sample stamped at or after the call) is >= --min-mhz; round 0 is warm-up and dropped.
"""
import argparse, bisect, collections, datetime, gzip, json, statistics, sys


def load_clock(path):
    ts, mhz = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                j = json.loads(line)
            except ValueError:
                continue
            t = datetime.datetime.fromisoformat(j["timestamp"]).timestamp() * 1000
            ts.append(t)
            mhz.append(j["gpu_usage"][0])
    return ts, mhz


def clock_at(ts, mhz, epoch_ms):
    i = bisect.bisect_left(ts, epoch_ms)
    return mhz[min(i, len(mhz) - 1)]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-mhz", type=float, default=1400)
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args()
    ts, mhz = load_clock(f"{a.dir}/macmon.jsonl.gz")
    cost = collections.defaultdict(list)
    cost_all = collections.defaultdict(list)
    clocks = collections.defaultdict(list)
    for line in open(f"{a.dir}/cost.csv"):
        if not line.startswith("data,"):
            continue
        _, ctx, kind, rnd, rep, ms, ep = line.strip().split(",")
        if int(rnd) == 0:
            continue
        c = clock_at(ts, mhz, float(ep))
        clocks[int(ctx)].append(c)
        cost_all[(int(ctx), kind)].append(float(ms))
        if c >= a.min_mhz:
            cost[(int(ctx), kind)].append(float(ms))
    out = {"min_mhz": a.min_mhz, "plain": {}, "verify": {}, "iter": {}, "clock": {}, "attn": {}}
    md = []
    for ctx in sorted(clocks):
        cl = clocks[ctx]
        out["clock"][ctx] = {"median": statistics.median(cl), "p10": q(cl, 0.1), "p90": q(cl, 0.9),
                             "kept_frac": sum(c >= a.min_mhz for c in cl) / len(cl)}
    kinds = sorted({k for _, k in cost_all}, key=lambda s: (s.rstrip("0123456789_k").split("_")[0], s))
    for ctx in sorted(clocks):
        for k in kinds:
            v, va = cost.get((ctx, k), []), cost_all.get((ctx, k), [])
            if not v:
                continue
            row = {"n": len(v), "n_all": len(va), "median": statistics.median(v), "p10": q(v, 0.1), "p90": q(v, 0.9),
                   "median_all": statistics.median(va)}
            grp = "plain" if k.startswith("plain") else "verify" if k.startswith("verify") else "iter"
            out[grp].setdefault(str(ctx), {})[k] = row
    # g0b attn: sec per dispatch, median over rounds >= 1 whose chain epoch has a good clock
    att = collections.defaultdict(list)
    for line in open(f"{a.dir}/g0b-attn.csv"):
        p = line.strip().split(",")
        if p[0] == "epoch_ms" or len(p) < 8 or p[1] != "attn":
            continue
        ep, _, ctx, cand, k, rnd, sec, nd = p[:8]
        if int(rnd) == 0:
            continue
        if clock_at(ts, mhz, float(ep)) < a.min_mhz:
            continue
        att[(int(ctx), cand)].append(float(sec) * 1e6)
    for (ctx, cand), v in att.items():
        out["attn"].setdefault(str(ctx), {})[cand] = {"n": len(v), "us": statistics.median(v)}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, sort_keys=True)
    md.append(f"GPU clock during cost samples (MHz, round >= 1): " + "; ".join(
        f"ctx {c}: median {v['median']:.0f} p10 {v['p10']:.0f} p90 {v['p90']:.0f}, kept {v['kept_frac']*100:.0f}% at >= {a.min_mhz:.0f}"
        for c, v in out["clock"].items()))
    for ctx in sorted(clocks):
        md.append(f"\n### ctx {ctx} (ms, median [p10-p90], n kept / n all)\n\n| pass | ms | n |\n|---|---:|---:|")
        for grp in ("plain", "verify", "iter"):
            for k, r in out[grp].get(str(ctx), {}).items():
                md.append(f"| {k} | {r['median']:.2f} [{r['p10']:.2f}-{r['p90']:.2f}] | {r['n']}/{r['n_all']} |")
    text = "\n".join(md)
    if a.md:
        open(a.md, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
