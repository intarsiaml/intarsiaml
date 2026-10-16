#!/usr/bin/env python3
"""g2_analyze.py [fast_mhz=1450]: per-pass table with GPU clock in the 4k prefill windows, ABBA summary with and without the
slow-clock passes, spec counters per prompt size from the /metrics timeline. Run with python3 -I in this directory."""
import json, re, glob, os, sys, statistics as st
from datetime import datetime
HERE = os.path.dirname(os.path.abspath(__file__)); FAST = float(sys.argv[1]) if len(sys.argv) > 1 else 1450.0
BASELINE = {128: 36.7, 1024: 36.6, 4096: 35.6}
mm = []
for line in open(HERE + "/macmon.jsonl"):
    try: d = json.loads(line)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_power"]))
mm.sort()
def window(a, b): return [s for s in mm if a < s[0] <= b + 0.1]
def med(x): return st.median(x) if x else None
def counters(label):
    out = []
    for line in open(f"{HERE}/passes/{label}-metrics.ts"):
        p = line.split()
        if not p: continue
        d = {"t": float(p[0])}
        for i in range(1, len(p) - 1, 2):
            try: d[p[i]] = float(p[i + 1])
            except ValueError: pass
        out.append(d)
    return out
passes = []
for f in sorted(glob.glob(HERE + "/passes/p*-localeval.ts")):
    label = re.search(r"(p\d-(?:off|on))-localeval", f)[1]; mode = label.split("-")[1]
    txt = open(f).read()
    mrid = re.search(r"results/(\d{8}-\d{6}-adhoc-[^/]+)/result.json", txt)
    if not mrid: continue
    rid = mrid[1]; res = json.load(open(f"/opt/code/deepeval/results/{rid}/result.json"))
    reps = []
    for line in txt.splitlines():
        ts, _, body = line.partition("\t")
        mr = re.search(r"\]\s+(\d+) tok rep (\d)/3: prompt=(\d+) ttft=([\d.]+)s prefill=\S+ tok/s decode=([\d.]+) tok/s.*out=(\d+)", body)
        if mr:
            size, rep, prompt, ttft, dec, out = int(mr[1]), int(mr[2]), int(mr[3]), float(mr[4]), float(mr[5]), int(mr[6])
            tend = float(ts); tdec = out / dec if dec else 0
            reps.append(dict(size=size, rep=rep, ttft=ttft, dec=dec, out=out, tend=tend, pa=tend - tdec - ttft, pb=tend - tdec, da=tend - tdec, db=tend))
    rows = {r["target_prompt_tokens"]: r for r in res["speed"]["rows"]}
    p = dict(label=label, mode=mode, rid=rid, reps=reps, dec={s: rows[s]["decode_tok_s"]["median"] for s in rows},
             decall={s: [r["dec"] for r in reps if r["size"] == s] for s in rows}, mp={s: rows[s]["marginal_prefill_tok_s"] for s in rows},
             pf={s: rows[s]["prefill_tok_s"]["median"] for s in rows})
    g = open(f"{HERE}/passes/{label}-gpu-idle.txt").read(); p["idle"] = re.search(r"FINAL median=(\d+)", g)[1]
    for size in (128, 1024, 4096):
        pw = [s for r in reps if r["size"] == size for s in window(r["pa"], r["pb"])]
        dw = [s for r in reps if r["size"] == size for s in window(r["da"], r["db"])]
        p[f"mhz_pf{size}"] = med([s[1] for s in pw]); p[f"mhz_dc{size}"] = med([s[1] for s in dw]); p[f"w_dc{size}"] = med([s[2] for s in dw])
    p["fast"] = p["mhz_pf4096"] is not None and p["mhz_pf4096"] >= FAST
    p["t0"] = reps[0]["pa"] if reps else 0
    # spec counters per size
    cs = counters(label); p["spec"] = {}
    if mode == "on" and cs:
        keys = [k for k in cs[-1] if k != "t"]
        def at(t):  # last sample at or before t
            c = [x for x in cs if x["t"] <= t]; return c[-1] if c else cs[0]
        for size in (128, 1024, 4096):
            rs = [r for r in reps if r["size"] == size]
            if not rs: continue
            a, b = at(rs[0]["pa"] - 0.5), at(rs[-1]["tend"] + 1.0)
            p["spec"][size] = {k.replace("intarsia_spec_", "").replace("_total", ""): b.get(k, 0) - a.get(k, 0) for k in keys}
    passes.append(p)
passes.sort(key=lambda p: p["t0"])
f0 = lambda x, d=0: "n/a" if x is None else f"{x:.{d}f}"
print(f"Fast-clock threshold: median GPU MHz over the three 4k prefill request windows >= {FAST:.0f}\n")
print("| pass | localeval run | idle util % | MHz 4k prefill | MHz decode 128/1k/4k | fast? | decode 128 | decode 1k | decode 4k (median of 3) | marginal prefill 1k / 4k tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in passes:
    print(f"| {p['label']} | {p['rid']} | {p['idle']} | {f0(p['mhz_pf4096'])} | {f0(p['mhz_dc128'])}/{f0(p['mhz_dc1024'])}/{f0(p['mhz_dc4096'])} | {'FAST' if p['fast'] else 'slow'} | "
          f"{p['dec'][128]:.1f} ({'/'.join(f'{x:.1f}' for x in p['decall'][128])}) | {p['dec'][1024]:.1f} ({'/'.join(f'{x:.1f}' for x in p['decall'][1024])}) | {p['dec'][4096]:.1f} ({'/'.join(f'{x:.1f}' for x in p['decall'][4096])}) | {f0(p['mp'][1024])} / {f0(p['mp'][4096])} |")
def summary(sel, title):
    print(f"\n### {title}\n")
    print("| size | spec off tok/s | spec on tok/s | on / off | spec off vs 36.7/36.6/35.6 | passes (off, on) used |")
    print("|---|---|---|---|---|---|")
    for s in (128, 1024, 4096):
        off = [x for p in sel if p["mode"] == "off" for x in p["decall"][s]]; on = [x for p in sel if p["mode"] == "on" for x in p["decall"][s]]
        if not off or not on: print(f"| {s} | n/a | n/a | n/a | | |"); continue
        mo, mn = st.median(off), st.median(on)
        print(f"| {s} | {mo:.1f} (n={len(off)}, {min(off):.1f}-{max(off):.1f}) | {mn:.1f} (n={len(on)}, {min(on):.1f}-{max(on):.1f}) | **{mn/mo:.2f}x** | {mo/BASELINE[s]*100-100:+.1f}% | {','.join(p['label'] for p in sel)} |")
summary(passes, "All passes (ABBA, no clock filter)")
fast = [p for p in passes if p["fast"]]
summary(fast, f"Passes with 4k-prefill clock >= {FAST:.0f} MHz only")
print("\n### spec counters from /metrics (spec-on passes, per prompt size: 3 reps x 256 tokens)\n")
print("| pass | size | verify passes | proposed | accepted | acceptance | tokens per pass (1 + accepted/passes) |")
print("|---|---|---|---|---|---|---|")
for p in passes:
    for s, c in p["spec"].items():
        sp, pr, ac = c.get("steps", 0), c.get("proposed_tokens", 0), c.get("accepted_tokens", 0)
        print(f"| {p['label']} | {s} | {sp:.0f} | {pr:.0f} | {ac:.0f} | {100*ac/pr if pr else 0:.1f}% | {1+ac/sp if sp else 0:.2f} |")
