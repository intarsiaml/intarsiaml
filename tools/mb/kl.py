#!/usr/bin/env python3
"""Lane mb: per-category KL(mb law || target) and total variation per emitted token, from the INTARSIA_SPEC_MB_AUDIT lines
a dt1 window saved (raw/<window>/mbaudit-<arm>-<cat>-r<round>.cum, cumulative per server: a category's passes are the lines
past the previous file's count, files in the window's order).
   tools/mb/kl.py <window> <arm>..."""
import glob, os, re, sys

W, arms = sys.argv[1], sys.argv[2:]
RAW = f"docs/results/2026-11-06-27b-dflash-hip-tp1/raw/{W}"
PAT = re.compile(r"mb-audit rows (\d+) emitted (\d+) kl (\S+) tv (\S+) rescued (\d+)")
print("| arm | category | passes | emitted tokens | KL per token (nats) | TV per token | rescued per token |")
print("|---|---|---|---|---|---|---|")
for arm in arms:
    files = sorted(glob.glob(f"{RAW}/mbaudit-{arm}-*.cum"), key=os.path.getmtime)
    seen, tot = 0, [0, 0, 0.0, 0.0, 0]
    for f in files:
        rows = [PAT.search(l) for l in open(f, errors="replace")]
        rows = [m for m in rows if m]
        new = rows[seen:]
        seen = len(rows)
        cat = re.search(rf"mbaudit-{arm}-(.+)-r\d+\.cum$", f).group(1)
        e = sum(int(m.group(2)) for m in new)
        kl = sum(float(m.group(3)) for m in new)
        tv = sum(float(m.group(4)) for m in new)
        rs = sum(int(m.group(5)) for m in new)
        for i, v in enumerate([len(new), e, kl, tv, rs]):
            tot[i] += v
        print(f"| {arm} | {cat} | {len(new)} | {e} | {kl / e:.4f} | {tv / e:.4f} | {rs / e:.3f} |")
    n, e, kl, tv, rs = tot
    print(f"| {arm} | **all** | {n} | {e} | **{kl / e:.4f}** | {tv / e:.4f} | {rs / e:.3f} |")
