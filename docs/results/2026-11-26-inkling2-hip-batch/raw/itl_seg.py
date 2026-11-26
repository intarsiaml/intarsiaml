"""Per stream count ITL / step-time percentiles of a serve visit (lane ink-hip-batch), from the server's own histograms.
  python3 itl_seg.py OUT_DIR VISIT
OUT_DIR/VISIT.poll: tatooine time, metric, value (serve_ctl.sh poll-start, every 0.2 s): intarsia_batch_itl_seconds_bucket (each gap
between two tokens of one request), intarsia_batch_step_seconds_bucket (a scheduler iteration that ran a decode step),
intarsia_batch_decode_rows_total{rows=N}. OUT_DIR/VISIT-rK-le.out: localeval speed's lines with the Mac's time in front. Each
concurrency level runs between the line before it (the size rep, or the previous level's "N streams: aggregate") and its own
"N streams: aggregate" line; the histograms' deltas over that span are the level's (clock offset from VISIT-clock.txt). Percentiles
interpolate linearly inside the server's buckets (1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 250, 500, 1000 ms), so they are bucket-coarse."""
import collections
import glob
import json
import os
import re
import sys

out, visit = sys.argv[1], sys.argv[2]
mac0, tat, mac1 = (float(x) for x in open(os.path.join(out, f"{visit}-clock.txt")).read().split()[:3])
off = tat - (mac0 + mac1) / 2  # tatooine minus Mac

samples = collections.OrderedDict()  # ts -> {metric: value}
for line in open(os.path.join(out, f"{visit}.poll")):
    p = line.split()
    if len(p) != 3:
        continue
    samples.setdefault(float(p[0]), {})[p[1]] = float(p[2])
ts = sorted(samples)


def at(t):
    """the last full sample at or before tatooine time t"""
    best = None
    for x in ts:
        if x <= t and len(samples[x]) > 10:
            best = x
    return samples[best] if best is not None else {}


def hist(s, name):
    b = []
    for k, v in s.items():
        m = re.match(name + r'_bucket\{le="([^"]+)"\}', k)
        if m:
            b.append((float("inf") if m.group(1) == "+Inf" else float(m.group(1)), v))
    return sorted(b)


def delta(a, b, name):
    ha, hb = dict(hist(a, name)), hist(b, name)
    return [(le, c - ha.get(le, 0)) for le, c in hb]


def pct(h, q):
    tot = h[-1][1] if h else 0
    if tot <= 0:
        return None
    want, prev_le, prev_c = q * tot, 0.0, 0.0
    for le, c in h:
        if c >= want:
            if le == float("inf"):
                return f">{prev_le * 1e3:.0f}"
            f = (want - prev_c) / (c - prev_c) if c > prev_c else 1.0
            return f"{(prev_le + f * (le - prev_le)) * 1e3:.0f}"
        prev_le, prev_c = le, c
    return None


def rows(a, b):
    r = {}
    for k, v in b.items():
        m = re.match(r'intarsia_batch_decode_rows_total\{rows="(\d+)"\}', k)
        if m:
            d = v - a.get(k, 0)
            if d > 0:
                r[int(m.group(1))] = int(d)
    return r


res = []
for le in sorted(glob.glob(os.path.join(out, f"{visit}-r*-le.out"))):
    rep = re.search(r"-r(\d+)-le\.out$", le).group(1)
    marks = []
    for line in open(le):
        p = line.split(" ", 1)
        if len(p) < 2:
            continue
        try:
            t = float(p[0])
        except ValueError:
            continue
        if re.search(r"tok rep \d+/\d+:", p[1]):
            marks.append((t, None, None))
        m = re.search(r"(\d+) streams: aggregate ([0-9.]+) tok/s", p[1])
        if m:
            marks.append((t, int(m.group(1)), float(m.group(2))))
    start = None
    for t, n, agg in marks:
        if n is None:
            start = t
            continue
        if start is None:
            start = t
            continue
        a, b = at(start + off), at(t + off)
        itl, step = delta(a, b, "intarsia_batch_itl_seconds"), delta(a, b, "intarsia_batch_step_seconds")
        res.append({"rep": int(rep), "streams": n, "aggregate_tok_s": agg, "span_s": round(t - start, 2),
                    "itl_ms_p50": pct(itl, 0.5), "itl_ms_p99": pct(itl, 0.99), "itl_gaps": int(itl[-1][1]) if itl else 0,
                    "step_ms_p50": pct(step, 0.5), "step_ms_p99": pct(step, 0.99), "decode_rows": rows(a, b)})
        start = t
for r in res:
    print(json.dumps(r))
