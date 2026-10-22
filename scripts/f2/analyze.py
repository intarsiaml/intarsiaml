#!/usr/bin/env python3
"""F2: summarise `accuracy --mode reuse --reuse-sweep` logs (SPLIT / ROW / SUM lines) per arm.
usage: analyze.py sweep-<arm>.log ...   (prints a markdown table plus per-split lines of the failing splits)"""
import math, re, sys

def parse(path):
    splits, rows = [], []
    for line in open(path, errors="replace"):
        if line.startswith("SPLIT "):
            m = re.match(r"SPLIT (\d+) turn2 (\d+) cached (\d+) cold_cached (\d+) agree (\d+)/(\d+) (ok|FAIL)(?: \((.*)\))?", line)
            if m:
                s, t2, ca, cc, ag, of, v, desc = m.groups()
                splits.append(dict(split=int(s), turn2=int(t2), cached=int(ca), cold_cached=int(cc), agree=int(ag), of=int(of), ok=v == "ok", desc=desc or ""))
        elif line.startswith("ROW "):
            m = re.match(r"ROW (\d+) (\d+) cold_top2 (\d+) (\d+) gap_cold (\S+) gap_warm (\S+) kl (\S+) flip (\d)", line)
            if m:
                s, j, a, b, gc, gw, kl, fl = m.groups()
                rows.append(dict(split=int(s), j=int(j), gc=float(gc), gw=float(gw), kl=float(kl), flip=int(fl)))
    return splits, rows

def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

def main(paths):
    print("| arm | splits | diverged | gate-fail | rows | argmax flips | median/p90/max abs(dgap) | rms dgap | mean KL | rows gap<0.25 / <0.5 / <1.0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    fails = []
    for p in paths:
        sp, rw = parse(p)
        arm = re.sub(r"^sweep-|\.log$", "", p.split("/")[-1])
        d = [abs(r["gw"] - r["gc"]) for r in rw]
        rms = math.sqrt(sum(x * x for x in d) / len(d)) if d else float("nan")
        div = sum(1 for s in sp if s["agree"] < s["of"])
        fail = [s for s in sp if not s["ok"] or s["cached"] < s["split"] or s["cold_cached"] != 0]
        for s in fail:
            fails.append((arm, s))
        g = [r["gc"] for r in rw]
        kl = sum(r["kl"] for r in rw) / max(len(rw), 1)
        print(f"| {arm} | {len(sp)} | {div} | {len(fail)} | {len(rw)} | {sum(r['flip'] for r in rw)} | {q(d,.5):.3f} / {q(d,.9):.3f} / {max(d, default=float('nan')):.3f} | {rms:.3f} | {kl:.2e} | {sum(x<.25 for x in g)} / {sum(x<.5 for x in g)} / {sum(x<1 for x in g)} |")
    print()
    print("Diverging splits (agree < of):")
    for p in paths:
        sp, _ = parse(p)
        arm = re.sub(r"^sweep-|\.log$", "", p.split("/")[-1])
        for s in sp:
            if s["agree"] < s["of"]:
                print(f"- {arm} split {s['split']}: agree {s['agree']}/{s['of']} {'ok' if s['ok'] else 'FAIL'} {s['desc']}")

main(sys.argv[1:])
