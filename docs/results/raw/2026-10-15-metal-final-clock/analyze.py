#!/usr/bin/env python3
"""Clock-aware analysis of the 12-pass slot. Usage: analyze.py [fast_threshold_mhz]  (run in this directory)."""
import json, re, sys, glob, os, statistics as st
from datetime import datetime
FAST = float(sys.argv[1]) if len(sys.argv) > 1 else 1450.0
HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 40.6
ENG = ["safe", "cand", "mlx-lm", "llamacpp"]
NAME = {"safe": "SAFE", "cand": "CAND", "mlx-lm": "mlx-lm", "llamacpp": "llama.cpp"}

# macmon samples: (epoch, mhz, watts)
mm = []
for line in open(f"{HERE}/macmon.jsonl"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    t = datetime.fromisoformat(d["timestamp"]).timestamp()
    mm.append((t, d["gpu_usage"][0], d["gpu_power"], d["gpu_usage"][1]))
mm.sort()

def window(a, b):
    # a sample stamped t covers roughly (t-interval, t]; take samples with a < t <= b+0.1
    return [s for s in mm if a < s[0] <= b + 0.1]

def med(x):
    return st.median(x) if x else None

passes = []
for f in sorted(glob.glob(f"{HERE}/passes/clk-*-localeval.ts")):
    label = re.search(r"(clk-.*)-localeval", f).group(1)
    m = re.match(r"clk-(safe|cand|mlx-lm|llamacpp)-r(\d)", label)
    eng, rnd = m.group(1), int(m.group(2))
    txt = open(f).read()
    mrid = re.search(r"results/(\d{8}-\d{6}-adhoc-[^/]+)/result.json", txt)
    if not mrid:
        continue  # pass still running
    rid = mrid.group(1)
    res = json.load(open(f"/opt/code/deepeval/results/{rid}/result.json"))
    reps = []
    for line in txt.splitlines():
        ts, _, body = line.partition("\t")
        mr = re.search(r"\]\s+(\d+) tok rep (\d)/3: prompt=(\d+) ttft=([\d.]+)s prefill=\S+ tok/s decode=([\d.]+) tok/s.*out=(\d+)", body)
        if mr:
            size, rep, prompt, ttft, dec, out = int(mr[1]), int(mr[2]), int(mr[3]), float(mr[4]), float(mr[5]), int(mr[6])
            tend = float(ts)
            tdec = out / dec if dec else 0
            reps.append(dict(size=size, rep=rep, ttft=ttft, dec=dec, tend=tend,
                             pa=tend - tdec - ttft, pb=tend - tdec, da=tend - tdec, db=tend))
    rows = {r["target_prompt_tokens"]: r for r in res["speed"]["rows"]}
    p = dict(label=label, eng=eng, rnd=rnd, rid=rid, reps=reps,
             dec={s: rows[s]["decode_tok_s"] for s in rows}, mp={s: rows[s]["marginal_prefill_tok_s"] for s in rows},
             ttft={s: rows[s]["ttft_s"]["median"] for s in rows})
    g = open(f"{HERE}/passes/{label}-gpu-idle.txt").read()
    p["idle"] = re.search(r"FINAL median=(\d+)", g).group(1)
    p["zoom"] = "yes" if re.search(r"zoom before: \S", g) else "no"
    for k, sizes in (("pf", None), ("dc", None)):
        pass
    for size in (128, 1024, 4096):
        pw = [s for r in reps if r["size"] == size for s in window(r["pa"], r["pb"])]
        dw = [s for r in reps if r["size"] == size for s in window(r["da"], r["db"])]
        p[f"mhz_pf{size}"] = med([s[1] for s in pw]); p[f"w_pf{size}"] = med([s[2] for s in pw]); p[f"n_pf{size}"] = len(pw)
        p[f"mhz_dc{size}"] = med([s[1] for s in dw]); p[f"w_dc{size}"] = med([s[2] for s in dw])
    allpf = [s[1] for r in reps if r["size"] in (1024, 4096) for s in window(r["pa"], r["pb"])]
    p["mhz_pf"] = med(allpf)
    p["fast"] = p["mhz_pf4096"] is not None and p["mhz_pf4096"] >= FAST
    p["t0"] = reps[0]["pa"] if reps else 0
    passes.append(p)
passes.sort(key=lambda p: p["t0"])

def fmt(x, d=0):
    return "n/a" if x is None else f"{x:.{d}f}"
def rng(vals, d=0):
    v = [x for x in vals if x is not None]
    return "n/a" if not v else f"{st.median(v):.{d}f} ({min(v):.{d}f}-{max(v):.{d}f}, n={len(v)})"
def med_of(vals):
    v = [x for x in vals if x is not None]
    return st.median(v) if v else None

out = []
P = out.append
P(f"Fast-clock threshold: a pass is FAST when the median GPU MHz over its three 4k prefill request windows is >= {FAST:.0f} (the 1k windows are only 0.25-0.43 s long = 2-3 macmon samples each and include the clock ramp-up from the preceding decode, so they are shown but not used for the flag)\n")
P("## Per-pass table\n")
P("| pass | localeval run id | idle util % | zoom | GPU MHz 1k pf | GPU MHz 4k pf | pf W 4k | MHz decode 1k | fast? | dec 128 | dec 1k | dec 4k | marg pf 1k | marg pf 4k | TTFT 1k s | TTFT 4k s |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for p in passes:
    P(f"| {p['label']} | {p['rid']} | {p['idle']} | {p['zoom']} | {fmt(p['mhz_pf1024'])} | {fmt(p['mhz_pf4096'])} | {fmt(p['w_pf4096'])} | {fmt(p['mhz_dc1024'])} | "
      f"{'FAST' if p['fast'] else 'slow'} | {p['dec'][128]['median']:.1f} | {p['dec'][1024]['median']:.1f} | {p['dec'][4096]['median']:.1f} | "
      f"{fmt(p['mp'][1024])} | {fmt(p['mp'][4096])} | {p['ttft'][1024]:.3f} | {p['ttft'][4096]:.3f} |")

def stats(subset):
    d = {}
    for e in ENG:
        ps = [p for p in subset if p["eng"] == e]
        d[e] = dict(n=len(ps),
                    dec={s: [p["dec"][s]["median"] for p in ps] for s in (128, 1024, 4096)},
                    mp={s: [p["mp"][s] for p in ps] for s in (1024, 4096)})
    return d
S_all = stats(passes)
S_fast = stats([p for p in passes if p["fast"]])

def table(S, title):
    P(f"\n## {title}\n")
    P("| engine | passes | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k |")
    P("|---|---|---|---|---|---|---|")
    for e in ENG:
        s = S[e]
        if not s["n"]:
            P(f"| {NAME[e]} | 0 | - | - | - | - | - |"); continue
        P(f"| {NAME[e]} | {s['n']} | {rng(s['dec'][128],1)} | {rng(s['dec'][1024],1)} | {rng(s['dec'][4096],1)} | {rng(s['mp'][1024])} | {rng(s['mp'][4096])} |")
table(S_all, "All-pass medians (median, (min-max, n)); decode tok/s, prefill tok/s")
table(S_fast, f"Fast-clock passes only (pass median GPU >= {FAST:.0f} MHz)")

def ratios(S, title):
    P(f"\n## {title}\n")
    P("| metric | SAFE/mlx-lm | CAND/mlx-lm | SAFE/llama.cpp | CAND/llama.cpp | SAFE %floor | CAND %floor |")
    P("|---|---|---|---|---|---|---|")
    res = {}
    for kind, sizes in (("dec", (128, 1024, 4096)), ("mp", (1024, 4096))):
        for s in sizes:
            M = {e: med_of(S[e][kind][s]) for e in ENG}
            def r(a, b):
                return None if not M[a] or not M[b] else M[a] / M[b]
            row = [r("safe", "mlx-lm"), r("cand", "mlx-lm"), r("safe", "llamacpp"), r("cand", "llamacpp")]
            fl = [M["safe"] / FLOOR if kind == "dec" and M["safe"] else None, M["cand"] / FLOOR if kind == "dec" and M["cand"] else None]
            P(f"| {'decode' if kind=='dec' else 'marginal prefill'} {s} | " + " | ".join("n/a" if x is None else f"{x*100:.1f}%" for x in row + fl) + " |")
            res[(kind, s)] = (M, row, fl)
    return res
R_all = ratios(S_all, "Ratios, all-pass medians")
R_fast = ratios(S_fast, "Ratios, fast-clock passes only")

def verdicts(R, title):
    P(f"\n## Verdicts: {title}\n")
    P("| criterion | SAFE | CAND |")
    P("|---|---|---|")
    for s in (1024, 4096):
        row = R[("mp", s)][1]
        def v(x):
            if x is None: return "n/a"
            lv = "stretch (>=85%)" if x >= .85 else "target (>=65%)" if x >= .65 else "committed (>=50%)" if x >= .5 else "BELOW committed"
            return f"{x*100:.1f}% of mlx-lm: {lv}"
        P(f"| prefill {s} vs mlx-lm | {v(row[0])} | {v(row[1])} |")
    for s in (128, 1024, 4096):
        M, row, fl = R[("dec", s)]
        def ok(b): return "n/a" if b is None else ("PASS" if b else "FAIL")
        def c(e, i):
            if not M[e]: return "n/a"
            f = M[e] / FLOOR
            a = None if not M["mlx-lm"] else M[e] / M["mlx-lm"]
            l = None if not M["llamacpp"] else M[e] >= M["llamacpp"]
            return (f">=85% floor: {ok(f >= .85)} ({f*100:.1f}%); >=90% mlx-lm: {ok(None if a is None else a >= .9)} ({'n/a' if a is None else f'{a*100:.1f}%'}); "
                    f">= llama.cpp: {ok(l)} ({M[e]:.1f} vs {fmt(M['llamacpp'],1)})")
        P(f"| decode {s} | {c('safe',0)} | {c('cand',1)} |")
verdicts(R_all, "all-pass medians")
if all(S_fast[e]["n"] for e in ENG):
    verdicts(R_fast, "fast-clock passes only")
else:
    P("\n(Fast-only verdicts omitted: not every engine has a fast pass: " + ", ".join(f"{NAME[e]}={S_fast[e]['n']}" for e in ENG) + ")")

# state-matched pairwise: within each round, same-round engines are closest in time but not state-matched; show per-round ratio
P("\n## Per-round marginal prefill 4k (each round's engines ran minutes apart, in rotated order)\n")
P("| round | order | " + " | ".join(NAME[e] for e in ENG) + " |")
P("|---|---|---|---|---|---|")
for r in (1, 2, 3):
    ps = sorted([p for p in passes if p["rnd"] == r], key=lambda p: p["t0"])
    cells = []
    for e in ENG:
        q = [p for p in ps if p["eng"] == e]
        cells.append("n/a" if not q else f"{fmt(q[0]['mp'][4096])} @ {fmt(q[0]['mhz_pf4096'])} MHz")
    P(f"| {r} | {' > '.join(NAME[p['eng']] for p in ps)} | " + " | ".join(cells) + " |")

# freq vs prefill correlation within engine
P("\n## Clock vs 4k-prefill TTFT per engine (pass level)\n")
P("| engine | pairs (MHz, TTFT 4k s) |")
P("|---|---|")
for e in ENG:
    P(f"| {NAME[e]} | " + "; ".join(f"{p['mhz_pf4096']:.0f}, {p['ttft'][4096]:.2f}" for p in passes if p["eng"] == e and p["mhz_pf4096"]) + " |")
P("\n## Per-request (prefill window): pass, size, rep, TTFT, median MHz, median W, samples\n")
P("| pass | size | rep | TTFT s | MHz | W | n |")
P("|---|---|---|---|---|---|---|")
for p in passes:
    for r in p["reps"]:
        if r["size"] == 128: continue
        w = window(r["pa"], r["pb"])
        P(f"| {p['label']} | {r['size']} | {r['rep']} | {r['ttft']:.2f} | {fmt(med([x[1] for x in w]))} | {fmt(med([x[2] for x in w]))} | {len(w)} |")
print("\n".join(out))
