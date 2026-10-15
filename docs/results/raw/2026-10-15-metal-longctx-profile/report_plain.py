#!/usr/bin/env python3
"""All plain/cb/kernel `generate` runs of the three slots with the GPU clock of their PREFILL window (busy samples, util > 0.5; window =
process end - decode time - 0.3 s, back by the prefill time, inner 90%). Writes analysis/runs.csv and prints per-context summaries.
Slot 1: ./clock/macmon.jsonl + ./runs.tsv; slot 2: s2/; slot 3: s3/."""
import json, os, re, glob, csv, statistics as st, datetime, collections
D = os.path.dirname(os.path.abspath(__file__))
def ts(s): return datetime.datetime.fromisoformat(s).timestamp()
def load_mm(p):
    out = []
    if not os.path.exists(p): return out
    for l in open(p):
        try:
            j = json.loads(l); out.append((ts(j["timestamp"]), j["gpu_usage"][0], j["gpu_usage"][1], j["gpu_power"], j["temp"]["gpu_temp_avg"]))
        except Exception: pass
    return out
rows = []
for slot, base in (("s1", D), ("s2", f"{D}/s2"), ("s3", f"{D}/s3")):
    mm = load_mm(f"{base}/clock/macmon.jsonl")
    rt = f"{base}/runs.tsv"
    if not os.path.exists(rt): continue
    runs = {}
    for l in list(open(rt))[1:]:
        a = l.rstrip("\n").split("\t")
        if len(a) == 3: runs[a[2]] = (float(a[0]), float(a[1]))
    for name, (a, b) in runs.items():
        for kind in ("plain", "cb", "kernel", "exp"):
            f = f"{base}/{kind}/{name}.err"
            if not os.path.exists(f): continue
            s = open(f).read()
            m = re.search(r"prefill (\d+) ms \(([\d.]+) tok/s\) \| decode (\d+) ms \(([\d.]+) tok/s\)", s)
            pt = re.search(r"prompt: (\d+) tokens", s)
            if not m or not pt: continue
            pf, dc = int(m.group(1)) / 1e3, int(m.group(3)) / 1e3
            pe = b - dc - 0.3
            w = [x for x in mm if pe - pf + 0.05 * pf <= x[0] <= pe - 0.05 * pf and x[2] > 0.5]
            wd = [x for x in mm if pe + 0.3 <= x[0] <= b - 0.5 and x[2] > 0.3]
            if len(w) < 3: continue
            fq = sorted(x[1] for x in w)
            ch = re.search(r"-c(\d+)", name); ctx = re.search(r"ctx(\d+)", name)
            rows.append(dict(slot=slot, kind=kind, name=name, prompt=int(pt.group(1)), chunk=int(ch.group(1)) if ch else None, prefill_ms=pf * 1e3, prefill_tps=float(m.group(2)),
                             decode_ms=dc * 1e3, decode_tps=float(m.group(4)) if dc > 1 else None, mhz_med=st.median(fq), mhz_p10=fq[len(fq)//10], pct_low=100 * sum(1 for x in fq if x < 1450) / len(fq),
                             watts=st.median(x[3] for x in w), tmax=max(x[4] for x in w), dec_mhz=st.median(x[1] for x in wd) if len(wd) > 2 else None))
os.makedirs(f"{D}/analysis", exist_ok=True)
with open(f"{D}/analysis/runs.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print("| context | kind | runs | prefill ms median (min-max) | prefill tok/s | decode tok/s median (min-max) | prefill MHz median | runs >= 1450 MHz |\n|---:|---|---:|---|---:|---|---:|---:|")
g = collections.defaultdict(list)
for r in rows:
    if r["chunk"] == 2048 and r["kind"] in ("plain", "cb"): g[(r["prompt"], r["kind"], "all")].append(r)
for (p, k, _), v in sorted(g.items()):
    for label, sub in (("all", v), ("clock>=1450", [r for r in v if r["mhz_med"] >= 1450])):
        if not sub: continue
        pf = [r["prefill_ms"] for r in sub]; dc = [r["decode_tps"] for r in sub if r["decode_tps"]]
        print(f"| {p} ({label}) | {k} | {len(sub)} | {st.median(pf):.0f} ({min(pf):.0f}-{max(pf):.0f}) | {p/st.median(pf)*1000:.0f} | " + (f"{st.median(dc):.2f} ({min(dc):.2f}-{max(dc):.2f})" if dc else "") + f" | {st.median(r['mhz_med'] for r in sub):.0f} | {sum(1 for r in sub if r['mhz_med']>=1450)} |")
