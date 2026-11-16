#!/usr/bin/env python3
"""Lane itl2s: tables from the client marks (cells/itl-*.json) and the server's `cb-trace-us` lines (serve-<visit>.log.gz).

analyze.py <raw dir> [--base OLD] [--detail VISIT]

Client side, per (visit, cell): chunks that arrive within 1 ms of each other are one arrival (a verified run is several SSE chunks
back to back); the gap between arrivals of a request is what a reader waits for the next text. Reported: p50 / p90 / p99 / max of
those gaps pooled over the cell's requests, and localeval's character-weighted `itl_ms_p99` (median over requests) for comparison.
Server side (Emit lines, one per scheduler iteration that handed tokens out): emission gaps per request and the per-token spread
(gap / tokens of the emission), wall time of the draft (`prepare`), verify and decode parts of each iteration, and which long gaps
had a prefill chunk in them (PrefillIter).
"""
import glob, gzip, json, os, re, statistics, sys
from collections import defaultdict

RAW = sys.argv[1]
BASE = "N1"
DETAIL = None
a = sys.argv[2:]
while a:
    if a[0] == "--base": BASE = a[1]; a = a[2:]
    elif a[0] == "--detail": DETAIL = a[1]; a = a[2:]
    else: a = a[1:]


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")


def burst_gaps(marks, eps=0.001):
    arr = []
    for t, _ in marks:
        if not arr or t - arr[-1] > eps:
            arr.append(t)
    return [(arr[i] - arr[i - 1]) * 1e3 for i in range(1, len(arr))], arr


def loc_itl_p99(marks, n_tokens):
    """localeval.client.itl_stats p99, character-weighted when chunks != tokens."""
    if len(marks) < 3 or not n_tokens: return None
    if len(marks) == n_tokens: w = [1.0] * len(marks)
    else:
        ch = sum(m[1] for m in marks)
        w = [n_tokens * m[1] / ch for m in marks]
    vals = [(marks[i][0] - marks[i - 1][0]) / w[i] for i in range(1, len(marks)) if w[i] > 0]
    return q(vals, 0.99) * 1e3 if len(vals) >= 2 else None


cells = defaultdict(list)  # (cell, arm) -> [(visit, data)]
for f in sorted(glob.glob(f"{RAW}/cells/itl-*.json")):
    d = json.load(open(f))
    cell = f"{d['set']}{'-nt' if d['thinking'] == 'off' else ''} T={d['temperature']:g} s{d['streams']}"
    cells[(cell, d["visit"].split(".")[0])].append((d["visit"], d))
cell_names = sorted({k[0] for k in cells}, key=lambda c: (c.split()[2], c.split()[1], c))
arms = sorted({k[1] for k in cells}, key=lambda x: (x != BASE, x))

print("## client: gaps between arrivals (ms), pooled over the cell's requests of all visits of the arm\n")
print("| cell | arm | visits | combined tok/s (median pair; per visit) | tok/s per stream | gap p50 | p90 | p99 | p99.9 | max | localeval ITL p99 (median of requests) | per-visit gap p99 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
summary = {}
for c in cell_names:
    for arm in arms:
        vs = cells.get((c, arm))
        if not vs: continue
        gaps, loc, rate, pv, comb = [], [], [], [], []
        for v, d in vs:
            vg = []
            for r in d["requests"]:
                if not r["ok"]: continue
                g, arr = burst_gaps(r["marks"])
                g = g[1:]  # the first arrival after the first token is the decode start, not steady state; kept separately below
                gaps += g; vg += g
                n = r["usage"].get("completion_tokens", 0)
                loc.append(loc_itl_p99(r["marks"], n) or float("nan"))
                if len(r["marks"]) > 2: rate.append((n - 1) / (r["marks"][-1][0] - r["marks"][0][0]))
            pv.append(q(vg, 0.99))
            grp = defaultdict(float)
            for r in d["requests"]:
                if r["ok"] and len(r["marks"]) > 2:
                    grp[(r["round"], r["group"])] += (r["usage"]["completion_tokens"] - 1) / (r["marks"][-1][0] - r["marks"][0][0])
            comb.append(statistics.median(grp.values()))
        summary[(c, arm)] = dict(p99=q(gaps, 0.99), p90=q(gaps, 0.9), p50=q(gaps, 0.5), max=max(gaps), rate=statistics.mean(rate), comb=comb, pv=pv, n=len(gaps))
        print(f"| {c} | {arm} | {len(vs)} | {statistics.mean(comb):.1f} ({' / '.join(f'{x:.0f}' for x in comb)}) | {statistics.mean(rate):.1f} | {q(gaps, .5):.1f} | {q(gaps, .9):.1f} | {q(gaps, .99):.1f} | {q(gaps, .999):.1f} | {max(gaps):.1f} | {statistics.median([x for x in loc if x == x]):.0f} | {' / '.join(f'{x:.0f}' for x in pv)} |")

# server side
EMIT = re.compile(r"cb-trace-us t=(\d+) Emit \{ iter: (\d+), tokens: \[(.*?)\], prepare_us: (\d+), verify_us: (\d+), verify_seqs: (\d+), verify_rows: (\d+), decode_us: (\d+), decode_rows: (\d+) \}")
PREF = re.compile(r"cb-trace-us t=(\d+) PrefillIter \{ iter: (\d+), prefill_us: (\d+), decoding: (\d+) \}")


def server(visit):
    p = f"{RAW}/serve-{visit}.log.gz"
    if not os.path.exists(p): return None
    em, pf = [], {}
    for line in gzip.open(p, "rt", errors="replace"):
        m = EMIT.search(line)
        if m:
            t, it, toks, pre, ver, vs, vr, dec, dr = m.groups()
            em.append(dict(t=int(t) / 1e6, iter=int(it), toks=[(int(a), int(b)) for a, b in re.findall(r"\((\d+), (\d+)\)", toks)], prepare=int(pre) / 1e3, verify=int(ver) / 1e3,
                           vseqs=int(vs), vrows=int(vr), decode=int(dec) / 1e3, drows=int(dr)))
            continue
        m = PREF.search(line)
        if m: pf[int(m.group(2))] = (int(m.group(1)) / 1e6, int(m.group(3)) / 1e3, int(m.group(4)))
    return em, pf


def cell_window(visit, cell_key):
    out = []
    for (c, arm), vs in cells.items():
        for v, d in vs:
            if v == visit:
                out.append((c, d["t_start"], d["t_end"]))
    return sorted(out, key=lambda x: x[1])



print("\n## server (Emit lines): per visit and cell\n")
print("| visit | cell | iterations | emission gap ms p50 / p99 / max | per-token spread p99 | 2-seq verify pass ms p50 / p99 | draft (prepare) ms p50 / p99 | decode step ms p50 / p99 | tokens per emission | dup iters |")
print("|---|---|---|---|---|---|---|---|---|---|")
RECS = defaultdict(list)   # visit -> gap records
for arm in arms:
    for visit in sorted({v for (c, ar), vs in cells.items() if ar == arm for v, _ in vs}):
        s_ = server(visit)
        if not s_: continue
        em, pf = s_
        seen = defaultdict(int)
        for e in em: seen[e["iter"]] += 1
        dup = sum(1 for v in seen.values() if v > 1)
        for c, t0, t1 in cell_window(visit, None):
            ee = [e for e in em if t0 <= e["t"] <= t1]
            if not ee: continue
            per = defaultdict(list)
            for e in ee:
                for rid, n in e["toks"]: per[rid].append((e["t"], n, e))
            gaps, spread, ntok, nem = [], [], 0, 0
            for rid, L in per.items():
                if sum(n for _, n, _ in L) < 100: continue  # the warm-up requests
                done = L[0][1]
                for k in range(1, len(L)):
                    g = (L[k][0] - L[k - 1][0]) * 1e3
                    e1, e0 = L[k][2], L[k - 1][2]
                    cls = ("V" if e1["vseqs"] and not e1["drows"] else "V+D" if e1["vseqs"] else "D")
                    pfin = [pf[i][1] for i in range(e0["iter"], e1["iter"]) if i in pf]
                    RECS[visit].append(dict(cell=c, gap=g, cls=cls, vrows=e1["vrows"], vseqs=e1["vseqs"], pf=sum(pfin), npf=len(pfin), pos=done, n=L[k][1],
                                            other=len(L) and sum(1 for r2, L2 in per.items() if r2 != rid and L2[0][0] <= L[k][0] <= L2[-1][0])))
                    gaps.append(g); spread.append(g / L[k][1]); ntok += L[k][1]; nem += 1
                    done += L[k][1]
            if not gaps: continue
            v2 = [e["verify"] for e in ee if e["vseqs"] == 2]
            dr = [e["prepare"] for e in ee if e["vseqs"] or e["prepare"]]
            dc = [e["decode"] for e in ee if e["drows"]]
            f = lambda v, a: (f"{q(v, a):.1f}" if v else "-")
            print(f"| {visit} | {c} | {len(ee)} | {q(gaps, .5):.1f} / {q(gaps, .99):.1f} / {max(gaps):.1f} | {q(spread, .99):.1f} | {f(v2, .5)} / {f(v2, .99)} (n={len(v2)}) | {f(dr, .5)} / {f(dr, .99)} | {f(dc, .5)} / {f(dc, .99)} | {ntok / nem:.2f} | {dup} |")

THR = float(os.environ.get("THR", "60"))
print(f"\n## server: what the gaps above {THR:.0f} ms (emission gaps, both streams running) were made of, per visit\n")
print("| visit | gaps | gaps > thr | class of the iteration that ended the gap (V / V+D / D) | prefill chunk inside | first 16 tokens of the reply | alone on the server | gap p99 over gaps with 2 streams running |")
print("|---|---|---|---|---|---|---|---|")
for visit, R in sorted(RECS.items()):
    big = [r for r in R if r["gap"] > THR]
    two = [r["gap"] for r in R if r["other"]]
    cl = {k: sum(1 for r in big if r["cls"] == k) for k in ("V", "V+D", "D")}
    print(f"| {visit} | {len(R)} | {len(big)} | {cl['V']} / {cl['V+D']} / {cl['D']} | {sum(1 for r in big if r['npf'])} | {sum(1 for r in big if r['pos'] < 16)} | {sum(1 for r in big if not r['other'])} | {q(two, .99):.1f} |")
print("\n## server: 2-seq verify pass wall time by row count (ms), per visit\n")
print("| visit | rows | passes | p50 | p99 | max |")
print("|---|---|---|---|---|---|")
for visit in sorted(RECS):
    s_ = server(visit)
    em, _ = s_
    by = defaultdict(list)
    for e in em:
        if e["vseqs"]: by[(e["vseqs"], e["vrows"])].append(e["verify"])
    for (sq, rw), v in sorted(by.items()):
        if len(v) >= 20: print(f"| {visit} | {sq} seqs / {rw} rows | {len(v)} | {q(v, .5):.1f} | {q(v, .99):.1f} | {max(v):.1f} |")
