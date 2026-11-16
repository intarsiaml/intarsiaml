#!/usr/bin/env python3
"""itlcreep: server vs client inter-token latency over time, from one window's raw dir (scripts/itlcreep/window.sh).

usage: analyze.py RAW_DIR [--bin-min 4] [--json OUT.json]

Reads
  p1/*.jsonl        box-local client, whole soak2 traffic (phase 1)
  p2/*.jsonl        box-local client, worker 1's conversations (phase 2)
  p2-mac/*.jsonl    Mac client, worker 0's conversations, concurrent with p2 (phase 2)
  serve-S.log.gz    scheduler trace (INTARSIA_CB_TRACE=1): DecodePhase / Decode / Chunk events with wall-clock ms
  sample-S.tsv      engine counters every 30 s (engine ITL histogram)
  clock.txt         Mac clock minus box clock, bracketing phase 2 (Mac marks are moved to the box clock)
Prints markdown tables: per replay and per time bin, one column group per vantage, and the straight-line trends.

Quantities (all per token chunk or step; the box-local client is the server's SSE emit seen over loopback):
  w99   localeval's per-request ITL p99 (chunk gap / character-proportional token weight), median over requests: the soak2 number
  g99   unweighted gap between consecutive content chunks, p99 pooled over the requests of the replay / bin (ms)
  s99   the same after merging chunks closer than 3 ms into one engine step (an MTP burst arrives token by token), p99
  g150  chunk gaps above 150 ms per 1000 chunks
  srv   scheduler iteration interval (DecodePhase to the next DecodePhase, ms) while a sequence decodes, from the trace
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import re
import statistics as st
import sys
import time

BURST_S = 0.003


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def slope(xs, ys):
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - my - b * (x - mx) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in res) / (n - 2) / sxx)
    return b, se


def load_rows(pattern, shift=0.0):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if not r.get("ok"):
                continue
            r["marks"] = [[m[0] + shift, m[1]] for m in r.get("marks", [])]
            r["epoch_start"] = r["epoch_start"] + shift
            r["_seed"] = r["seed"]
            rows.append(r)
    return rows


def gaps_of(r):
    t = [m[0] for m in r["marks"]]
    gaps = [(b - a) * 1e3 for a, b in zip(t, t[1:])]
    when = t[1:]
    steps = [t[0]]
    stw = []
    sg = []
    for x in t[1:]:
        if x - steps[-1] >= BURST_S:
            sg.append((x - steps[-1]) * 1e3)
            stw.append(x)
            steps.append(x)
    return when, gaps, stw, sg


def pooled(rows):
    g, s, n = [], [], 0
    for r in rows:
        _, gg, _, sg = gaps_of(r)
        g += gg
        s += sg
        n += len(r["marks"])
    return g, s, n


def stats(rows):
    if not rows:
        return None
    g, s, n = pooled(rows)
    return {
        "n_req": len(rows),
        "ttft_sum": sum(r["ttft_s"] for r in rows if r["ttft_s"] is not None),
        "w99": st.median([r["itl_ms_p99"] for r in rows if r["itl_ms_p99"] is not None]),
        "w50": st.median([r["itl_ms_p50"] for r in rows if r["itl_ms_p50"] is not None]),
        "g99": pctl(g, .99), "s50": pctl(s, .5), "s99": pctl(s, .99),
        "g150": 1000 * sum(x > 150 for x in g) / max(1, n), "g250": 1000 * sum(x > 250 for x in g) / max(1, n),
        "tok_s": st.median([r["decode_tok_s"] for r in rows if r["decode_tok_s"]]),
    }


def fmt(x, d=0):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def load_trace(path):
    """-> list of (t_ms, kind, fields) for DecodePhase / Chunk / Admitted lines."""
    ev = []
    if not os.path.exists(path):
        return ev
    rx = re.compile(r"^cb-trace t=(\d+) (\w+) \{ (.*) \}")
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith("cb-trace"):
                continue
            m = rx.match(line)
            if not m:
                continue
            kind = m.group(2)
            if kind not in ("DecodePhase", "Chunk", "Admitted", "Finished"):
                continue
            d = dict(re.findall(r"(\w+): (\w+)", m.group(3)))
            ev.append((int(m.group(1)) / 1000.0, kind, d))
    return ev


def server_intervals(ev):
    """Scheduler iteration intervals while at least one sequence decodes: (t_end_s, ms, b_before, had_chunk)."""
    out, prev, chunk_since = [], None, False
    for t, kind, d in ev:
        if kind == "Chunk":
            chunk_since = True
        elif kind == "DecodePhase":
            it, b = int(d["iter"]), int(d["b"])
            if prev is not None and it == prev[0] + 1 and prev[2] >= 1:
                out.append((t, (t - prev[1]) * 1e3, prev[2], chunk_since))
            prev, chunk_since = (it, t, b), False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--bin-min", type=float, default=4.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    raw = a.raw

    shift = 0.0  # box clock = Mac clock - offset
    cf = os.path.join(raw, "clock.txt")
    offs = []
    if os.path.exists(cf):
        for line in open(cf):
            m = re.search(r"offset_mac_minus_box_s=([+-][\d.]+) rtt_s=([\d.]+)", line)
            if m:
                offs.append((float(m.group(1)), float(m.group(2))))
    if offs:
        shift = -st.mean(o for o, _ in offs)
        print(f"clock: Mac minus box offsets {[o for o, _ in offs]} s (rtt {[r for _, r in offs]}); Mac marks shifted by {shift:+.3f} s\n")

    V = {"box p1 (K=2, both workers)": load_rows(f"{raw}/p1/*-k2.jsonl"),
         "box p2 (worker 1)": load_rows(f"{raw}/p2/*-k2.jsonl"),
         "Mac p2 (worker 0)": load_rows(f"{raw}/p2-mac/*-k2.jsonl", shift)}
    allrows = [r for v in V.values() for r in v]
    if not allrows:
        sys.exit("no client rows")
    t0 = min(r["epoch_start"] for r in allrows)
    print(f"t0 = first request {time.strftime('%H:%M:%S', time.localtime(t0))} (box clock, local time of this Mac); minutes below are since t0\n")
    out = {"vantage": {}}

    # ---- per replay
    print("## Per replay\n")
    print("| vantage | seed | start min | req | TTFT sum s | w99 ms | w50 ms | g99 ms | s50 ms | s99 ms | g150 /1000 | tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    for name, rows in V.items():
        by = {}
        for r in rows:
            by.setdefault(r["_seed"], []).append(r)
        per[name] = []
        for seed in sorted(by):
            s = stats(by[seed])
            start = min(r["epoch_start"] for r in by[seed])
            mid = (start - t0) / 60
            per[name].append((mid, seed, s))
            print(f"| {name} | {seed} | {mid:.1f} | {s['n_req']} | {s['ttft_sum']:.1f} | {fmt(s['w99'])} | {fmt(s['w50'],1)} | {fmt(s['g99'])} | {fmt(s['s50'],1)} | {fmt(s['s99'])} | {fmt(s['g150'],1)} | {fmt(s['tok_s'],1)} |")
    out["per_replay"] = {n: [(m, sd, s) for m, sd, s in v] for n, v in per.items()}

    # ---- trends per vantage (per-replay medians vs start minute); only complete replays (>= 20 requests for the full ones, >= 10 for worker halves)
    print("\n## Trends (per replay, least squares on start minute; value per 10 min and its standard error)\n")
    print("| vantage | replays | w99 | g99 | s99 | g150 /1000 | TTFT sum (per request s) |")
    print("|---|---|---|---|---|---|---|")
    for name, v in per.items():
        full = [(m, s) for m, sd, s in v if s["n_req"] >= (40 if "p1" in name else 20)]
        row = [name, str(len(full))]
        for key in ("w99", "g99", "s99", "g150"):
            b, se = slope([m for m, s in full], [s[key] for m, s in full])
            row.append(f"{b*10:+.1f} +- {se*10:.1f}")
        b, se = slope([m for m, s in full], [s["ttft_sum"] / s["n_req"] for m, s in full])
        row.append(f"{b*10:+.3f} +- {se*10:.3f}")
        print("| " + " | ".join(row) + " |")

    # ---- time bins: clients, server trace, engine histogram
    ev = load_trace(os.path.join(raw, "serve-S.log.gz"))
    iv = server_intervals(ev)
    # engine ITL histogram deltas from the sampler (HH:MM:SS local time of the box; assumed same zone as this Mac)
    samp = []
    sf = os.path.join(raw, "sample-S.tsv")
    if os.path.exists(sf):
        lines = open(sf).read().splitlines()
        hdr = lines[0].split("\t")
        day = time.strftime("%Y-%m-%d", time.localtime(t0))
        for ln in lines[1:]:
            c = ln.split("\t")
            try:
                ts = time.mktime(time.strptime(day + " " + c[0], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            d = dict(zip(hdr, c))
            samp.append((ts, d))
    bw = a.bin_min * 60
    tmax = max(max(m[0] for r in allrows for m in r["marks"][-1:]), t0)
    nb = int((tmax - t0) // bw) + 1
    print(f"\n## Time bins of {a.bin_min:g} min (box clock)\n")
    print("| bin start min | box p1: w99 / g99 / s99 / g150 | box p2: same | Mac p2: same | srv iv p50 / p99 / >150 per 1000 | engine gaps >100 / >150 / >250 per 1000 tok |")
    print("|---|---|---|---|---|---|")
    binrows = []
    for i in range(nb):
        lo, hi = t0 + i * bw, t0 + (i + 1) * bw
        cells = []
        rec = {"bin_min": i * a.bin_min}
        for name, rows in V.items():
            sel = [r for r in rows if lo <= r["marks"][-1][0] < hi] if rows else []
            s = stats(sel) if len(sel) >= 3 else None
            cells.append("-" if s is None else f"{fmt(s['w99'])} / {fmt(s['g99'])} / {fmt(s['s99'])} / {fmt(s['g150'],1)}")
            rec[name] = s
        sv = [x for x in iv if lo <= x[0] < hi]
        if len(sv) > 50:
            ms = [x[1] for x in sv]
            sc = f"{pctl(ms,.5):.0f} / {pctl(ms,.99):.0f} / {1000*sum(m>150 for m in ms)/len(ms):.1f}"
            rec["srv"] = {"n": len(ms), "p50": pctl(ms, .5), "p99": pctl(ms, .99), "gt150_per_1000": 1000 * sum(m > 150 for m in ms) / len(ms)}
        else:
            sc = "-"
        e = [(ts, d) for ts, d in samp if lo - 1 <= ts <= hi + 1]
        ec = "-"
        if len(e) >= 2:
            f0, f1 = e[0][1], e[-1][1]
            def dd(k):
                try:
                    return float(f1[k]) - float(f0[k])
                except (KeyError, ValueError):
                    return float("nan")
            tot = dd("intarsia_batch_itl_seconds_count")
            if tot > 0:
                g100 = tot - dd('intarsia_batch_itl_seconds_bucket{le="0.1"}')
                g150 = tot - dd('intarsia_batch_itl_seconds_bucket{le="0.15"}')
                g250 = tot - dd('intarsia_batch_itl_seconds_bucket{le="0.25"}')
                ec = f"{1000*g100/tot:.1f} / {1000*g150/tot:.1f} / {1000*g250/tot:.2f}"
                rec["engine"] = {"tokens": tot, "gt100": g100, "gt150": g150, "gt250": g250}
        binrows.append(rec)
        print(f"| {i*a.bin_min:g} | " + " | ".join(cells) + f" | {sc} | {ec} |")
    out["bins"] = binrows

    # ---- server-only trend and concurrent Mac vs box in phase 2
    print("\n## Server scheduler interval trend (trace, while a sequence decodes)\n")
    xs = [(r.get("srv") or {}).get("p99") for r in binrows]
    ok = [(i * a.bin_min, r["srv"]["p99"], r["srv"]["gt150_per_1000"]) for i, r in enumerate(binrows) if "srv" in r]
    if len(ok) >= 4:
        b, se = slope([x for x, _, _ in ok], [y for _, y, _ in ok])
        b2, se2 = slope([x for x, _, _ in ok], [z for _, _, z in ok])
        print(f"p99 of the iteration interval: {b*10:+.1f} +- {se*10:.1f} ms per 10 min over {len(ok)} bins; share above 150 ms: {b2*10:+.2f} +- {se2*10:.2f} per 1000 per 10 min")
    print(f"trace events parsed: {len(ev)}; iteration intervals: {len(iv)}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
