#!/usr/bin/env python3
"""Lane au1: markdown table of one window of scripts/au1/cat-window.sh: decode tok/s per category and arm (tools/dfh/cat_table.py),
auto vs MTP k2 and vs the faster of MTP k2 / DFlash2 k7, the rule's verdict per category, and the share of DFlash passes of the auto arm
(from its per-pass logs). usage: win_table.py <raw/W> [auto arm name, default auto]"""
import glob, re, subprocess, sys, os
raw = sys.argv[1].rstrip("/"); auto = sys.argv[2] if len(sys.argv) > 2 else "auto"
here = os.path.dirname(os.path.abspath(__file__))
out = subprocess.run([sys.executable, "-I", os.path.join(here, "..", "dfh", "cat_table.py"), raw], capture_output=True, text=True, env=dict(os.environ, PREC="3")).stdout
lines = out.splitlines()
print(lines[0], file=sys.stderr)
for l in lines[1:]:
    if "PROBLEM" in l: print(l, file=sys.stderr)
hdr = next(l for l in lines if l.startswith("category")).split()[1:]
data = {}
for l in lines:
    m = re.match(r"^((?:ho|rcp)-\S+)\s+(.*)$", l)
    if not m: continue
    cat, rest = m.groups()
    vals = re.findall(r"(\d+\.\d+)\s+\d+\.\d\dx(?:\s+(\d+\.\d\d)t/p)?", rest)
    data[cat] = {a: (float(v), float(t) if t else None) for a, (v, t) in zip(hdr, vals)}
share = {}
cats = [("ho-chat", 4), ("ho-code", 2), ("ho-prose", 4), ("ho-recall", 2), ("rcp-code", 8), ("rcp-prose", 8), ("rcp-recall", 8)]
D = {c: [0, 0] for c, _ in cats}
for f in sorted(glob.glob(f"{raw}/pass-{auto}-r*.log")):
    reqs, cur = [], []
    for ln in open(f, errors="replace"):
        if ln.startswith("spec pass "):
            cur.append(ln.split()[2])
        elif ln.startswith("spec auto:"):
            reqs.append(cur); cur = []
    i = 0
    for c, n in cats:
        for r in reqs[i:i + n]:
            D[c][0] += r.count("D"); D[c][1] += len(r)
        i += n
print(f"| category | mtp2 | dfl7 | auto | auto / mtp2 | auto / best | rule | DFlash passes of auto |")
print("|---|---:|---:|---:|---:|---:|---|---:|")
allok = True
for c, _ in cats:
    d = data[c]; m, f, a = d["mtp2"][0], d["dfl7"][0], d[auto][0]
    best = max(m, f); r1, r2 = a / m, a / best
    ok = r1 >= 0.99 and r2 >= 0.97; allok &= ok
    print(f"| {c} | {m:.1f} | {f:.1f} | **{a:.1f}** | {r1:.3f} | {r2:.3f} | {'pass' if ok else 'FAIL'} | {100 * D[c][0] / max(D[c][1], 1):.0f}% |")
print(f"\nrule (every category auto/mtp2 >= 0.99 and auto/best >= 0.97): {'MET' if allok else 'NOT MET'}", file=sys.stderr)
extra = [a for a in hdr if a not in ("mtp2", "dfl7", auto)]
if extra:
    print("\n| category | " + " | ".join(f"{a} tok/s (x mtp2)" for a in extra) + " |"); print("|---|" + "---:|" * len(extra))
    for c, _ in cats:
        print(f"| {c} | " + " | ".join(f"{data[c][a][0]:.1f} ({data[c][a][0] / data[c]['mtp2'][0]:.3f})" for a in extra) + " |")
