#!/usr/bin/env python3
"""Lane f185: per-round and pooled prefill / TTFT / decode ratios (mtp over off) from the localeval speed files of scripts/mpf/longctx.sh runs.
usage: python3 tools/mpf/pool.py <dir>=<tag> [<dir>=<tag> ...]   (files <arm>-r<N>.speed.txt; arms off and mtp; rounds are tagged <tag>-r<N>)"""
import glob, os, re, sys, statistics as st
def load(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.speed.txt"))):
        arm, rnd = re.match(r"(.+)-r(\d+)\.speed\.txt", os.path.basename(f)).groups()
        for line in open(f):
            m = re.search(r"prompt=(\d+).*?prefill=(\d+).*?decode=([\d.]+)", line)
            if not m: continue
            n = int(m.group(1)); b = "4k" if n < 8000 else "16k" if n < 25000 else "34k"
            out.setdefault((b, arm, f"{tag}-r{rnd}"), []).append((n, float(m.group(2)), float(m.group(3))))
    return out
allr = {}
for spec in sys.argv[1:]:
    d, tag = spec.split("=")
    allr.update(load(d, tag))
rounds = sorted({k[2] for k in allr})
print("rounds pooled:", rounds)
for b in ("4k", "16k", "34k"):
    pr, dr, offp, mtpp, offd, mtpd, offt, mtpt = [], [], [], [], [], [], [], []
    for r in rounds:
        if (b, "off", r) in allr and (b, "mtp", r) in allr:
            o, m = allr[(b, "off", r)], allr[(b, "mtp", r)]
            po, pm = st.median(x[1] for x in o), st.median(x[1] for x in m)
            do, dm = st.median(x[2] for x in o), st.median(x[2] for x in m)
            pr.append(pm / po); dr.append(dm / do)
            offp += [x[1] for x in o]; mtpp += [x[1] for x in m]; offd += [x[2] for x in o]; mtpd += [x[2] for x in m]
            offt += [x[0] / x[1] for x in o]; mtpt += [x[0] / x[1] for x in m]
            print(f"  {b:3} {r:6} prefill off {po:.0f} mtp {pm:.0f} x {pm/po:.3f} | decode off {do:.1f} mtp {dm:.1f} x {dm/do:.3f}")
    print(f"{b}: rounds {len(pr)}; pooled prefill off {st.median(offp):.0f} mtp {st.median(mtpp):.0f} x {st.median(mtpp)/st.median(offp):.4f}; per-round x min {min(pr):.3f} max {max(pr):.3f} median {st.median(pr):.3f}; "
          f"TTFT off {st.median(offt):.3f}s mtp {st.median(mtpt):.3f}s ({st.median(mtpt)/st.median(offt):.3f}); decode off {st.median(offd):.1f} mtp {st.median(mtpd):.1f} x {st.median(mtpd)/st.median(offd):.3f}; per-round decode x min {min(dr):.3f} max {max(dr):.3f}")
