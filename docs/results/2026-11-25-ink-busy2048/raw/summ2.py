"""(ink-busy2048 copy of raw/b48/summ.py: no output-check column, peak RSS from rss.txt.) Busy-chunk A/B summary: per arm, localeval conc-plain-1k (aggregate tok/s median, TTFT median / p99 per stream count),
server ITL quantiles from the /metrics histogram delta (interpolated inside the bucket; whole run, all stream counts),
device memory in use at the end snapshot, output check. usage: summ.py BCDIR DEEPEVAL_RESULTS"""
import glob
import json
import re
import sys

bc, res = sys.argv[1], sys.argv[2]


def hist(path, name):
    b = {}
    for l in open(path):
        m = re.match(rf'^{name}_bucket{{le="([^"]+)"}} (\S+)', l)
        if m:
            b[float("inf") if m[1] == "+Inf" else float(m[1])] = float(m[2])
    return b


def q(h0, h1, p):
    ks = sorted(h1)
    d = [(k, h1[k] - h0.get(k, 0)) for k in ks]
    n = d[-1][1]
    if n <= 0:
        return None
    t, lo, prev = p * n, 0.0, 0.0
    for k, c in d:
        if c >= t:
            if k == float("inf"):
                return f">{lo * 1e3:.0f}"
            return f"{(lo + (k - lo) * (t - prev) / max(c - prev, 1)) * 1e3:.0f}"
        lo, prev = k, c
    return None


def top(h0, h1):
    ks = sorted(h1)
    last = 0.0
    for k in ks:
        if h1[k] - h0.get(k, 0) > 0 and k != float("inf"):
            pass
    lo = 0.0
    for k in ks:
        if h1[k] - h0.get(k, 0) >= h1[ks[-1]] - h0.get(ks[-1], 0):
            return f"<= {k * 1e3:.0f}" if k != float("inf") else f"> {lo * 1e3:.0f}"
        lo = k


def gauge(path, name):
    for l in open(path):
        if l.startswith(name + " "):
            return float(l.split()[1])


print("| arm | busy chunk | 1 str tok/s | 4 str | 16 str | TTFT p50 s (1/4/16) | TTFT p99 s (1/4/16) | ITL p50 / p99 / max ms "
      "| device mem used GiB | peak RSS GiB | rejected |")
print("|---|---:|---:|---:|---:|---|---|---|---:|---:|---|")
for a in sorted(glob.glob(f"{bc}/arm*-bc*"), key=lambda s: int(re.search(r"arm(\d+)", s)[1])):
    n, c = re.search(r"arm(\d+)-bc(\d+)", a).groups()
    if not glob.glob(f"{a}/metrics1.txt"):
        print(f"| {n} | {c} | (running) |")
        continue
    lg = open(f"{a}/localeval.log").read() if glob.glob(f"{a}/localeval.log") else ""
    m = re.search(r"results/(\S+)/result\.json", lg)
    rows = {}
    rej = "-"
    if m:
        j = json.load(open(f"{res}/{m[1]}/result.json"))
        rows = {r["streams"]: r for r in j["speed"]["concurrency"]}
        rj = [str(s) for s, r in rows.items() if r.get("rejected")]
        rej = ",".join(rj) + " (spread)" if rj else "none"
    g = lambda s, k: (rows[s][k]["median"] if isinstance(rows[s][k], dict) else rows[s][k]) if s in rows else None
    h0, h1 = hist(f"{a}/metrics0.txt", "intarsia_batch_itl_seconds"), hist(f"{a}/metrics1.txt", "intarsia_batch_itl_seconds")
    itl = f"{q(h0, h1, .5)} / {q(h0, h1, .99)} / {top(h0, h1)}"
    tot, free = gauge(f"{a}/metrics1.txt", "intarsia_device_memory_total_bytes"), gauge(f"{a}/metrics1.txt", "intarsia_device_memory_free_bytes")
    mem = f"{(tot - free) / 2**30:.1f}" if tot and free else "-"
    try:
        rss = f"{max(int(x) for x in open(f'{a}/rss.txt').read().split()) / 2**20:.2f}"
    except (OSError, ValueError):
        rss = "-"
    f = lambda x: "-" if x is None else f"{x:.1f}" if x > 5 else f"{x:.2f}"
    print(f"| {n} | {c} | {f(g(1, 'aggregate_tps'))} | {f(g(4, 'aggregate_tps'))} | {f(g(16, 'aggregate_tps'))} | "
          f"{f(g(1, 'ttft_s'))} / {f(g(4, 'ttft_s'))} / {f(g(16, 'ttft_s'))} | "
          f"{f(rows.get(1, {}).get('ttft_p99_s'))} / {f(rows.get(4, {}).get('ttft_p99_s'))} / {f(rows.get(16, {}).get('ttft_p99_s'))} | "
          f"{itl} | {mem} | {rss} | {rej} |")
